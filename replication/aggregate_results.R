# Builds mean (sd) tables, Friedman test and MCB plots from everything under results/.
#   Rscript aggregate_results.R [--quick]
args <- commandArgs(trailingOnly = TRUE)
for (f in c("config.R", "data_metrics.R", "aggregate.R")) source(file.path("R", f))
cfg <- PARNN_CONFIG; if (any(args == "--quick")) cfg$out_root <- "results_quick"
outdir <- file.path(cfg$out_root, "aggregate"); dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
all <- load_all_reps(cfg)
write.csv(all, file.path(outdir, "all_repetitions_long.csv"), row.names = FALSE)
mean_sd_tables(all, outdir)
fr <- friedman_mcb(all, outdir)
print(fr)
cat("Wrote tables and plots to", outdir, "\n")
