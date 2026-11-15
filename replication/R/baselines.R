# ============================================================================
# Statistical / ML baselines of the paper (Sec. 4.2), fitted on the in-sample
# window (train + validation = everything before the test window), as in the
# authors' Statistical_Machine_Learning_Model_Implementation.R.
# ts frequency is the DETECTED period S when S <= arima_max_seasonal, else 1.
# ============================================================================
ts_freq <- function(S, cfg = PARNN_CONFIG) if (S > 1L && S <= cfg$arima_max_seasonal) as.integer(S) else 1L
.ts <- function(y, S, cfg) stats::ts(y, frequency = ts_freq(S, cfg))
.num <- function(x) as.numeric(x)

# ARIMA is shared by ARIMA / Hybrid-1 / Hybrid-2 (and exported for Hybrid-3 in Python)
get_arima <- function(y, S, cfg) {
  f <- ts_freq(S, cfg)
  suppressWarnings(forecast::auto.arima(stats::ts(y, frequency = f), seasonal = f > 1L))
}

# --- deterministic baselines --------------------------------------------------
bl_RWD   <- function(y, h, S, cfg, seed, ctx) .num(forecast::rwf(stats::ts(y), h = h, drift = TRUE)$mean)
bl_ETS   <- function(y, h, S, cfg, seed, ctx) .num(forecast::forecast(forecast::ets(.ts(y, S, cfg)), h = h)$mean)
bl_ARIMA <- function(y, h, S, cfg, seed, ctx) .num(forecast::forecast(ctx$arima, h = h)$mean)
bl_TBATS <- function(y, h, S, cfg, seed, ctx) {
  x <- if (S > 1L) stats::ts(y, frequency = S) else stats::ts(y)
  .num(forecast::forecast(suppressWarnings(forecast::tbats(x, use.parallel = FALSE)), h = h)$mean)
}

# --- stochastic baselines -----------------------------------------------------
bl_ARNN <- function(y, h, S, cfg, seed, ctx) {
  set.seed(seed)
  .num(forecast::forecast(forecast::nnetar(.ts(y, S, cfg)), h = h)$mean)
}

# Recursive AR-type MLP on a (standardised) series: ensemble of nnet nets.
.lag_mlp_forecast <- function(x, h, L, size, reps, nn, seed) {
  mu <- mean(x); sdv <- stats::sd(x); if (!is.finite(sdv) || sdv <= 0) sdv <- 1
  z <- (x - mu) / sdv; n <- length(z); idx <- (L + 1L):n
  X <- do.call(cbind, lapply(seq_len(L), function(i) z[idx - i]))
  E <- ens_fit(X, z[idx], size, reps, nn, seed)
  lag <- rev(utils::tail(z, L)); out <- numeric(h)
  for (s in seq_len(h)) { out[s] <- ens_predict(E, matrix(lag, nrow = 1L)); lag <- c(out[s], lag)[seq_len(L)] }
  list(fc = out * sdv + mu, E = E, X = X, y = z[idx])
}

# MLP: nnfor::mlp (the paper's MLP) when installed; otherwise a documented stand-in.
bl_MLP <- function(y, h, S, cfg, seed, ctx) {
  if (requireNamespace("nnfor", quietly = TRUE)) {
    set.seed(seed)
    fit <- nnfor::mlp(.ts(y, S, cfg))
    return(.num(forecast::forecast(fit, h = h)$mean))
  }
  # STAND-IN (nnfor not installed): 20-net averaged MLP on L lags of the series,
  # L = max(detected period, 5) capped at 24, hidden = ceil((L+1)/2).
  L <- min(max(ts_freq(S, cfg), 5L), 24L)
  .lag_mlp_forecast(y, h, L, ceiling((L + 1) / 2), 20L, cfg$nnet, seed)$fc
}
bl_MLP_is_standin <- function() !requireNamespace("nnfor", quietly = TRUE)

# Hybrid-1 (Zhang 2003, ARIMA-ANN): ARIMA forecast + an MLP trained on the ARIMA
# residuals. Residual lag count L and hidden size chosen on the last 20% of residuals.
bl_Hybrid1 <- function(y, h, S, cfg, seed, ctx) {
  res <- as.numeric(stats::residuals(ctx$arima)); res[!is.finite(res)] <- 0
  n <- length(res); cut <- floor(0.8 * n)
  best <- c(Inf, 1L, 1L)
  for (L in 1:5) for (sz in 1:4) {
    sc <- tryCatch({
      f <- .lag_mlp_forecast(res[seq_len(cut)], n - cut, L, sz, 5L, cfg$nnet, seed + 100L * L + sz)$fc
      mean((res[(cut + 1L):n] - f)^2)
    }, error = function(e) Inf)
    if (is.finite(sc) && sc < best[1]) best <- c(sc, L, sz)
  }
  rf <- .lag_mlp_forecast(res, h, best[2], best[3], 20L, cfg$nnet, seed)$fc
  .num(forecast::forecast(ctx$arima, h = h)$mean) + rf
}

# Hybrid-2 (ARIMA-ARNN): ARIMA forecast + nnetar fitted on the ARIMA residuals (as in the repo)
bl_Hybrid2 <- function(y, h, S, cfg, seed, ctx) {
  set.seed(seed)
  res <- stats::ts(as.numeric(stats::residuals(ctx$arima)), frequency = ts_freq(S, cfg))
  res[!is.finite(res)] <- 0
  .num(forecast::forecast(ctx$arima, h = h)$mean) +
    .num(forecast::forecast(forecast::nnetar(res), h = h)$mean)
}

BASELINES <- list(
  RWD = list(f = bl_RWD, stochastic = FALSE), ETS = list(f = bl_ETS, stochastic = FALSE),
  ARIMA = list(f = bl_ARIMA, stochastic = FALSE), TBATS = list(f = bl_TBATS, stochastic = FALSE),
  MLP = list(f = bl_MLP, stochastic = TRUE), ARNN = list(f = bl_ARNN, stochastic = TRUE),
  `Hybrid-1` = list(f = bl_Hybrid1, stochastic = TRUE), `Hybrid-2` = list(f = bl_Hybrid2, stochastic = TRUE)
)

# ---- task table (also consumed by the Python deep-learning script) -----------
task_table <- function(cfg = PARNN_CONFIG) {
  do.call(rbind, lapply(names(DATASETS), function(nm) {
    y <- load_series(nm, cfg); n <- length(y); hs <- horizon_lengths(DATASETS[[nm]]$unit, n, cfg)
    data.frame(Dataset = nm, Horizon = names(hs), h = as.integer(hs), n = n, in_sample_n = n - as.integer(hs),
               path = file.path(cfg$data_root, DATASETS[[nm]]$path), target = DATASETS[[nm]]$target,
               unit = DATASETS[[nm]]$unit, row.names = NULL)
  }))
}

run_baseline_task <- function(name, horizon, S, models, cfg = PARNN_CONFIG, force = FALSE) {
  dir.create(file.path(cfg$out_root, "baselines"), recursive = TRUE, showWarnings = FALSE)
  dir.create(file.path(cfg$out_root, "arima_export"), recursive = TRUE, showWarnings = FALSE)
  y <- load_series(name, cfg); n <- length(y)
  h <- horizon_lengths(DATASETS[[name]]$unit, n, cfg)[[horizon]]
  ins <- y[seq_len(n - h)]; test <- y[(n - h + 1L):n]
  ctx <- list(arima = get_arima(ins, S, cfg))
  # export ARIMA residuals + forecast for Hybrid-3 (ARIMA-LSTM, Python)
  ex <- file.path(cfg$out_root, "arima_export", paste0(name, "_", horizon))
  write.csv(data.frame(resid = as.numeric(stats::residuals(ctx$arima))), paste0(ex, "_resid.csv"), row.names = FALSE)
  write.csv(data.frame(fc = as.numeric(forecast::forecast(ctx$arima, h = h)$mean)), paste0(ex, "_fc.csv"), row.names = FALSE)
  for (m in models) {
    fn <- file.path(cfg$out_root, "baselines", paste0(name, "_", horizon, "_", m, "_reps.csv"))
    if (!force && file.exists(fn)) next
    spec <- BASELINES[[m]]; reps <- if (spec$stochastic) cfg$repetitions else 1L
    t0 <- Sys.time(); rows <- list()
    for (r in seq_len(reps)) {
      pr <- tryCatch(spec$f(ins, h, S, cfg, cfg$seed + 1000L * r, ctx), error = function(e) { message("  ", m, " failed: ", conditionMessage(e)); NULL })
      if (is.null(pr) || length(pr) != h || anyNA(pr)) { rows <- list(); break }
      rows[[r]] <- cbind(Dataset = name, Horizon = horizon, Model = m, Rep = r, all_metrics(test, pr, ins, S))
    }
    if (length(rows)) {
      write.csv(do.call(rbind, rows), fn, row.names = FALSE)
      cat(sprintf("  [%s %s] %-9s MASE=%.3f (%.0fs)\n", name, horizon, m, mean(sapply(rows, function(z) z$MASE)),
                  as.numeric(difftime(Sys.time(), t0, units = "secs"))))
    }
  }
  invisible(NULL)
}
