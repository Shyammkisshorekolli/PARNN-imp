# Statistical / ML baselines (RWD, ETS, ARIMA, TBATS, MLP, ARNN, Hybrid-1, Hybrid-2).
# Also writes results/task_table.csv and results/arima_export/ for the Python script.
#   Rscript run_baselines.R [--datasets=A,B] [--horizons=ST,MT,LT] [--models=ARIMA,ARNN] [--quick] [--force]
suppressPackageStartupMessages({ library(forecast); library(nnet) })
args <- commandArgs(trailingOnly = TRUE)
opt <- function(key, default = NULL) { a <- grep(paste0("^--", key, "="), args, value = TRUE)
  if (length(a)) sub(paste0("^--", key, "="), "", a[1]) else default }
files <- file.path(getwd(), "R", c("config.R", "data_metrics.R", "seasonality.R", "parnn.R", "run_parnn.R", "baselines.R"))
options(parnn.files = files); for (f in files) source(f)

cfg <- PARNN_CONFIG
if (any(args == "--quick")) { cfg$repetitions <- 2L; cfg$seasonality$n_surrogates <- 30L; cfg$out_root <- "results_quick" }
datasets <- strsplit(opt("datasets", paste(names(DATASETS), collapse = ",")), ",")[[1]]
horizons <- strsplit(opt("horizons", "ST,MT,LT"), ",")[[1]]
models   <- strsplit(opt("models", paste(names(BASELINES), collapse = ",")), ",")[[1]]
force    <- any(args == "--force")
if (bl_MLP_is_standin()) message("NOTE: package 'nnfor' not installed -> MLP uses an internal stand-in (see R/baselines.R).")

dir.create(cfg$out_root, showWarnings = FALSE, recursive = TRUE)
write.csv(task_table(cfg), file.path(cfg$out_root, "task_table.csv"), row.names = FALSE)
season <- do.call(rbind, lapply(datasets, dataset_period, cfg = cfg))
write.csv(season, file.path(cfg$out_root, "seasonality_detected.csv"), row.names = FALSE)
for (d in datasets) for (hz in horizons) {
  cat(sprintf("[%s %s]\n", d, hz))
  tryCatch(run_baseline_task(d, hz, season$S[season$Dataset == d], models, cfg, force),
           error = function(e) message("FAILED ", d, " ", hz, ": ", conditionMessage(e)))
}
cat("Done.\n")
