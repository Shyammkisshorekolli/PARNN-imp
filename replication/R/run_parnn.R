# One (dataset, horizon) experiment: tune (p,q) on validation -> refit on train+val
# -> forecast the test window -> metrics (+ prediction interval). Resumable.

task_seed <- function(name, horizon, cfg = PARNN_CONFIG) {
  # deterministic, distinct per task, no hand-picked numbers
  as.integer((cfg$seed + 1000L * match(name, names(DATASETS)) + 100L * match(horizon, names(cfg$horizons$weekly))) %% 2^30)
}

dataset_period <- function(name, cfg = PARNN_CONFIG) {
  # Detected ONCE per dataset on the data that precedes the longest test window,
  # so no horizon's test data is ever used and ST/MT/LT share one S.
  y <- load_series(name, cfg); n <- length(y)
  h <- max(horizon_lengths(DATASETS[[name]]$unit, n, cfg))
  r <- resolve_period(name, y[seq_len(n - h)], cfg)
  data.frame(Dataset = name, n = n, S = r$period, reason = r$reason,
             best_candidate = if (is.null(r$best_p)) NA else r$best_p,
             best_score = if (is.null(r$best_score)) NA else r$best_score,
             gate = if (is.null(r$gate)) NA else r$gate)
}

run_parnn_task <- function(name, horizon, S, cfg = PARNN_CONFIG, force = FALSE) {
  dir.create(file.path(cfg$out_root, "parnn"), recursive = TRUE, showWarnings = FALSE)
  stem <- file.path(cfg$out_root, "parnn", paste0(name, "_", horizon))
  if (!force && file.exists(paste0(stem, "_metrics.csv")))
    return(read.csv(paste0(stem, "_metrics.csv")))

  t0 <- Sys.time()
  y <- load_series(name, cfg); n <- length(y)
  h <- horizon_lengths(DATASETS[[name]]$unit, n, cfg)[[horizon]]
  v <- cfg$validation_multiple * h
  tr_n <- n - v - h
  if (tr_n < 30L) stop(name, " ", horizon, ": training window too short (", tr_n, ")")
  train <- y[seq_len(tr_n)]
  val   <- y[tr_n + seq_len(v)]
  test  <- y[tr_n + v + seq_len(h)]
  seed  <- task_seed(name, horizon, cfg)
  cat(sprintf("[%s %s] n=%d h=%d train=%d val=%d S=%d\n", name, horizon, n, h, tr_n, v, S))

  tuning <- parnn_tune(train, val, S, seed, cfg)
  write.csv(tuning, paste0(stem, "_tuning.csv"), row.names = FALSE)
  best <- tuning[1L, ]
  if (!is.finite(best$val_MASE)) stop("All (p,q) fits failed for ", name, " ", horizon)

  in_sample <- c(train, val)
  do_pi <- horizon %in% cfg$pi$horizons
  rows <- vector("list", cfg$repetitions); fc1 <- NULL; arima1 <- NULL
  for (r in seq_len(cfg$repetitions)) {
    fit <- parnn_fit(in_sample, S, best$p, best$q, cfg$final_repeats, seed + 99999L + r, cfg, arima = arima1)
    arima1 <- fit$arima                                   # ARIMA is deterministic: fit once
    set.seed(seed + 7L + r)
    fc <- parnn_forecast(fit, h, cfg, paths = if (do_pi && r == 1L) cfg$pi$paths else 0L,
                         level = cfg$pi$level, nets = cfg$pi$nets)
    if (r == 1L) fc1 <- fc
    rows[[r]] <- cbind(Dataset = name, Horizon = horizon, Model = "PARNN", Rep = r,
                       all_metrics(test, fc$mean, in_sample, S))
  }
  reps <- do.call(rbind, rows)
  write.csv(reps, paste0(stem, "_reps.csv"), row.names = FALSE)
  mm <- colMeans(reps[, c("MASE", "RMSE", "SMAPE", "MASE_insample")])
  ss <- apply(reps[, c("MASE", "RMSE", "SMAPE")], 2, stats::sd)
  out <- cbind(Dataset = name, Horizon = horizon, h = h, S = S, p = best$p, k = best$k, q = best$q,
               ARIMA = arima1$order, AIC = arima1$aic, val_MASE = best$val_MASE,
               as.data.frame(t(mm)), MASE_sd = ss[["MASE"]], RMSE_sd = ss[["RMSE"]], SMAPE_sd = ss[["SMAPE"]],
               seconds = round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1))
  write.csv(data.frame(Step = seq_len(h), Actual = test, PARNN = fc1$mean,
                       Lower = if (do_pi) fc1$lower else NA, Upper = if (do_pi) fc1$upper else NA),
            paste0(stem, "_forecast.csv"), row.names = FALSE)
  write.csv(out, paste0(stem, "_metrics.csv"), row.names = FALSE)
  out
}

# Table-2 style summary: tuned (m, k, l) per dataset and horizon, built from results.
summarise_parnn <- function(cfg = PARNN_CONFIG) {
  fs <- list.files(file.path(cfg$out_root, "parnn"), pattern = "_metrics\\.csv$", full.names = TRUE)
  if (!length(fs)) return(invisible(NULL))
  all <- do.call(rbind, lapply(fs, read.csv))
  all <- all[order(match(all$Dataset, names(DATASETS)), match(all$Horizon, c("ST", "MT", "LT"))), ]
  write.csv(all, file.path(cfg$out_root, "parnn_all_metrics.csv"), row.names = FALSE)
  all$mkl <- sprintf("(%d,%d,%d)", all$p, all$k, all$q)
  t2 <- reshape(all[, c("Dataset", "Horizon", "mkl")], idvar = "Dataset", timevar = "Horizon", direction = "wide")
  write.csv(t2, file.path(cfg$out_root, "table2_tuned_hyperparameters.csv"), row.names = FALSE)
  invisible(all)
}
