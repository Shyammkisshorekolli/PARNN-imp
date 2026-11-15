# ============================================================================
# Aggregation: mean (sd) tables per horizon (paper Tables 3-5 layout),
# Friedman / Iman-Davenport test (Table 6) and MCB (Nemenyi) mean-rank plots (Fig. 3).
# Everything is computed from the result files; no paper values are used here.
# ============================================================================
MODEL_ORDER <- c("PARNN", "RWD", "ETS", "ARIMA", "TBATS", "MLP", "ARNN",
                 "Hybrid-1", "Hybrid-2", "Hybrid-3", "NBeats", "DeepAR", "TCN", "Transformer")
METRICS <- c("MASE", "RMSE", "SMAPE")

load_all_reps <- function(cfg = PARNN_CONFIG) {
  fs <- unlist(lapply(c("parnn", "baselines", "deep"), function(d)
    list.files(file.path(cfg$out_root, d), pattern = "_reps\\.csv$", full.names = TRUE)))
  if (!length(fs)) stop("No *_reps.csv found under ", cfg$out_root)
  cols <- c("Dataset", "Horizon", "Model", "Rep", METRICS)
  all <- do.call(rbind, lapply(fs, function(f) read.csv(f, check.names = FALSE)[, cols]))
  all$Model <- factor(all$Model, levels = c(MODEL_ORDER, setdiff(unique(all$Model), MODEL_ORDER)))
  all$Dataset <- factor(all$Dataset, levels = names(DATASETS))
  all$Horizon <- factor(all$Horizon, levels = c("ST", "MT", "LT"))
  droplevels(all)
}

fmt_ms <- function(m, s) ifelse(is.na(s), sprintf("%.3f", m), sprintf("%.3f (%.3f)", m, s))

mean_sd_tables <- function(all, outdir) {
  for (hz in levels(all$Horizon)) {
    d <- all[all$Horizon == hz, ]; if (!nrow(d)) next
    rows <- list()
    for (met in METRICS) {
      agg <- aggregate(d[[met]], list(Dataset = d$Dataset, Model = d$Model), function(v) c(m = mean(v), s = if (length(v) > 1) sd(v) else NA))
      agg <- data.frame(Dataset = agg$Dataset, Model = agg$Model, m = agg$x[, "m"], s = agg$x[, "s"])
      agg$txt <- fmt_ms(agg$m, agg$s)
      best <- ave(agg$m, agg$Dataset, FUN = function(v) min(v, na.rm = TRUE))
      agg$txt <- ifelse(agg$m == best, paste0("*", agg$txt, "*"), agg$txt)      # *best per dataset*
      w <- reshape(agg[, c("Dataset", "Model", "txt")], idvar = "Dataset", timevar = "Model", direction = "wide")
      names(w) <- sub("^txt\\.", "", names(w)); w <- cbind(Metric = met, w)
      rows[[met]] <- w
    }
    all_cols <- unique(unlist(lapply(rows, names)))
    out <- do.call(rbind, lapply(rows, function(w) { for (cn in setdiff(all_cols, names(w))) w[[cn]] <- NA; w[, all_cols] }))
    write.csv(out, file.path(outdir, paste0("table_", hz, "_mean_sd.csv")), row.names = FALSE)
  }
}

friedman_mcb <- function(all, outdir, alpha = 0.05) {
  res <- list()
  for (met in METRICS) {
    mm <- aggregate(all[[met]], list(Dataset = all$Dataset, Horizon = all$Horizon, Model = all$Model), mean)
    W <- reshape(mm, idvar = c("Dataset", "Horizon"), timevar = "Model", direction = "wide")
    names(W) <- sub("^x\\.", "", names(W)); M <- as.matrix(W[, -(1:2), drop = FALSE])
    keep <- colSums(is.na(M)) == 0
    if (sum(keep) < 3 || nrow(M) < 2) { message("Friedman (", met, "): not enough complete data, skipped"); next }
    dropped <- colnames(M)[!keep]; M <- M[, keep, drop = FALSE]
    n <- nrow(M); k <- ncol(M)
    R <- t(apply(M, 1, rank, ties.method = "average")); rbar <- colMeans(R)
    chi <- 12 * n / (k * (k + 1)) * sum((rbar - (k + 1) / 2)^2)
    Ff  <- (n - 1) * chi / (n * (k - 1) - chi)
    p   <- stats::pf(Ff, k - 1, (k - 1) * (n - 1), lower.tail = FALSE)
    cd  <- stats::qtukey(1 - alpha, k, Inf) / sqrt(2) * sqrt(k * (k + 1) / (6 * n))
    res[[met]] <- data.frame(Metric = met, models = k, tasks = n, chi2 = chi, F = Ff,
                             df1 = k - 1, df2 = (k - 1) * (n - 1), p_value = p, CD_nemenyi = cd,
                             dropped_models = paste(dropped, collapse = ";"))
    rk <- data.frame(Model = names(rbar), MeanRank = as.numeric(rbar))[order(rbar), ]
    write.csv(rk, file.path(outdir, paste0("mean_ranks_", met, ".csv")), row.names = FALSE)
    png(file.path(outdir, paste0("mcb_", met, ".png")), width = 900, height = 150 + 32 * k, res = 110)
    par(mar = c(4, 9, 3, 1)); y <- rev(seq_len(k))
    plot(rk$MeanRank, y, xlim = range(c(rk$MeanRank - cd / 2, rk$MeanRank + cd / 2)), yaxt = "n", ylab = "",
         xlab = "Mean rank (lower = better)", pch = 19,
         main = sprintf("MCB / Nemenyi, %s (%d tasks, CD = %.2f)", met, n, cd))
    axis(2, at = y, labels = rk$Model, las = 1)
    segments(rk$MeanRank - cd / 2, y, rk$MeanRank + cd / 2, y)
    abline(v = c(rk$MeanRank[1] - cd / 2, rk$MeanRank[1] + cd / 2), lty = 2, col = "grey40")
    dev.off()
  }
  out <- do.call(rbind, res); write.csv(out, file.path(outdir, "friedman_test.csv"), row.names = FALSE); out
}
