# Usage (from the project folder):
#   Rscript run_parnn.R                                  # all 12 datasets x ST/MT/LT, full settings
#   Rscript run_parnn.R --datasets=Tourism,UK_Unemployment --horizons=ST,MT
#   Rscript run_parnn.R --cores=4                        # parallel grid search
#   Rscript run_parnn.R --quick                          # tiny settings, smoke test only
#   Rscript run_parnn.R --force                          # recompute even if results exist
# Requires R packages: forecast, nnet.  Results go to ./results/
suppressPackageStartupMessages({ library(forecast); library(nnet) })

args <- commandArgs(trailingOnly = TRUE)
opt <- function(key, default = NULL) {
  a <- grep(paste0("^--", key, "="), args, value = TRUE)
  if (length(a)) sub(paste0("^--", key, "="), "", a[1]) else default
}
flag <- function(key) any(args == paste0("--", key))

files <- file.path(getwd(), "R", c("config.R", "data_metrics.R", "seasonality.R", "parnn.R", "run_parnn.R"))
options(parnn.files = files)
for (f in files) source(f)

cfg <- PARNN_CONFIG
cfg$cores <- as.integer(opt("cores", cfg$cores))
if (flag("quick")) {                       # smoke-test settings (NOT for reporting results)
  cfg$p_grid <- 1:3; cfg$q_grid <- 1:3
  cfg$tuning_repeats <- 3L; cfg$final_repeats <- 20L; cfg$repetitions <- 2L
  cfg$seasonality$n_surrogates <- 30L; cfg$pi$paths <- 200L; cfg$pi$nets <- 20L
  cfg$out_root <- "results_quick"
}
datasets <- strsplit(opt("datasets", paste(names(DATASETS), collapse = ",")), ",")[[1]]
horizons <- strsplit(opt("horizons", "ST,MT,LT"), ",")[[1]]
force    <- flag("force")

dir.create(cfg$out_root, showWarnings = FALSE, recursive = TRUE)
season <- do.call(rbind, lapply(datasets, dataset_period, cfg = cfg))
write.csv(season, file.path(cfg$out_root, "seasonality_detected.csv"), row.names = FALSE)
print(season)

for (d in datasets) for (hz in horizons) {
  S <- season$S[season$Dataset == d]
  res <- tryCatch(run_parnn_task(d, hz, S, cfg, force),
                  error = function(e) { message("FAILED ", d, " ", hz, ": ", conditionMessage(e)); NULL })
  if (!is.null(res)) print(res)
}
summarise_parnn(cfg)
cat("\nDone. See", cfg$out_root, "\n")
