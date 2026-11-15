# Sanity check of data, splits, horizons and the MASE convention against the paper.
# Recomputes the random-walk-with-drift (RWD) baseline for the medium-term horizon and
# compares it with the values published in Table 4 (validation/paper_reference_rwd_mt.csv).
# Run from the project folder:  Rscript validation/check_rwd_against_paper.R
suppressPackageStartupMessages(library(forecast))
for (f in c("config", "data_metrics")) source(file.path("R", paste0(f, ".R")))
ref <- read.csv("validation/paper_reference_rwd_mt.csv")
out <- do.call(rbind, lapply(seq_len(nrow(ref)), function(i) {
  nm <- ref$Dataset[i]; y <- load_series(nm); n <- length(y)
  h  <- horizon_lengths(DATASETS[[nm]]$unit, n)[[ref$Horizon[i]]]
  tr <- y[seq_len(n - h)]; te <- y[(n - h + 1):n]
  f  <- rwf(ts(tr), h = h, drift = TRUE)$mean
  data.frame(Dataset = nm, h = h, RMSE_ours = rmse(te, f), RMSE_paper = ref$RMSE[i],
             MASE_ours = mase_test(te, f), MASE_paper = ref$MASE[i],
             MASE_insample_S1 = mase_insample(te, f, tr, 1L))
}))
print(out, digits = 4, row.names = FALSE)
cat("\nMax relative RMSE gap:", format(max(abs(out$RMSE_ours / out$RMSE_paper - 1)), digits = 2),
    "| max relative MASE gap:", format(max(abs(out$MASE_ours / out$MASE_paper - 1)), digits = 2), "\n")
