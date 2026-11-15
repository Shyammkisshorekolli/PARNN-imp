# Plot observed history, PARNN forecast, and test-period ground truth.
# This is an add-on script; it does not modify the replication code or results.
# Run from the PARNN_replication project folder:
#   Rscript plot_parnn_actuals.R
#   Rscript plot_parnn_actuals.R --horizon=ST --separate
#   Rscript plot_parnn_actuals.R --datasets=AMZN,Tourism --context=150

args <- commandArgs(trailingOnly = TRUE)
opt <- function(key, default = NULL) {
  a <- grep(paste0("^--", key, "="), args, value = TRUE)
  if (length(a)) sub(paste0("^--", key, "="), "", a[1]) else default
}
flag <- function(key) any(args == paste0("--", key))

for (f in c("config.R", "data_metrics.R")) source(file.path("R", f))
cfg <- PARNN_CONFIG
if (flag("quick")) cfg$out_root <- "results_quick"

hz <- opt("horizon", "MT")
if (!hz %in% c("ST", "MT", "LT")) stop("--horizon must be ST, MT, or LT")
context <- opt("context", NA_character_)
context <- if (is.na(context)) NA_integer_ else as.integer(context)
if (!is.na(context) && (is.na(context) || context < 1L)) stop("--context must be a positive integer")
datasets <- strsplit(opt("datasets", paste(names(DATASETS), collapse = ",")), ",", fixed = TRUE)[[1]]
unknown <- setdiff(datasets, names(DATASETS))
if (length(unknown)) stop("Unknown dataset(s): ", paste(unknown, collapse = ", "))

pretty <- c(AMZN = "AMZN Stock", GOOG = "GOOG Stock", MSFT = "MSFT Stock",
            Births = "Births", Colombia_Dengue = "Colombia Dengue",
            Colombia_Malaria = "Colombia Malaria", Venezuela_Dengue = "Venezuela Dengue",
            Venezuela_Malaria = "Venezuela Malaria", US_EPU_Index = "US EPU Index",
            UK_Unemployment = "UK Unemployment", Russia_Exchange = "Russia Exchange",
            Tourism = "Tourism")
unit_label <- c(daily = "Daily", weekly = "Weekly", monthly = "Monthly", quarterly = "Quarterly")

draw_panel <- function(ds) {
  info <- DATASETS[[ds]]
  path <- file.path(cfg$out_root, "parnn", paste0(ds, "_", hz, "_forecast.csv"))
  if (!file.exists(path)) {
    plot.new()
    title(main = pretty[[ds]])
    text(0.5, 0.5, paste0("No ", hz, " forecast yet"), col = "grey40")
    return(FALSE)
  }

  y <- load_series(ds, cfg)
  h <- horizon_lengths(info$unit, length(y), cfg)[[hz]]
  fc <- read.csv(path, check.names = FALSE)
  required <- c("Actual", "PARNN")
  if (!all(required %in% names(fc))) stop("Expected Actual and PARNN columns in ", path)
  if (nrow(fc) != h) stop("Forecast length mismatch in ", path, ": expected ", h, ", found ", nrow(fc))
  if (length(y) <= h) stop("Series is shorter than its forecast horizon: ", ds)

  origin <- length(y) - h
  first <- if (is.na(context)) 1L else max(1L, origin - context + 1L)
  x_history <- first:origin
  x_test <- (origin + 1L):length(y)
  values <- c(y[x_history], fc$Actual, fc$PARNN)
  yrange <- range(values[is.finite(values)], na.rm = TRUE)
  if (!all(is.finite(yrange)) || diff(yrange) == 0) yrange <- yrange + c(-1, 1)

  plot(NA, xlim = c(first, length(y)), ylim = yrange,
       xlab = unit_label[[info$unit]], ylab = info$target,
       main = pretty[[ds]], bty = "l", cex.main = 1.0, cex.lab = 0.9, cex.axis = 0.8)
  lines(x_history, y[x_history], col = "black", lwd = 1.0)
  lines(c(origin, x_test), c(y[origin], fc$Actual), col = "blue", lwd = 1.5)
  lines(c(origin, x_test), c(y[origin], fc$PARNN), col = "red", lwd = 1.8)
  abline(v = origin, lty = 3, col = "darkorange")
  TRUE
}

plot_dir <- file.path(cfg$out_root, "plots")
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)
nr <- ceiling(length(datasets) / 4)
out <- file.path(plot_dir, paste0("PARNN_vs_ground_truth_", hz, ".png"))
grDevices::png(out, width = 2400, height = 520 * nr + 140, res = 200)
oldpar <- par(mfrow = c(nr, 4), mar = c(3.5, 4.0, 2.0, 0.6), oma = c(2.5, 0, 0, 0))
shown <- vapply(datasets, draw_panel, logical(1))
par(oldpar)
par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0), new = TRUE)
plot(0, 0, type = "n", bty = "n", xaxt = "n", yaxt = "n", xlab = "", ylab = "")
legend("bottom", horiz = TRUE, bty = "n", cex = 1.0, inset = 0.005,
       legend = c("Observed history", "PARNN forecast", "Ground truth", "Forecast start"),
       col = c("black", "red", "blue", "darkorange"),
       lty = c(1, 1, 1, 3), lwd = c(1.5, 2, 1.5, 1))
grDevices::dev.off()
cat("Wrote ", normalizePath(out, winslash = "/", mustWork = TRUE),
    " (", sum(shown), " of ", length(datasets), " datasets have forecast results)\n", sep = "")

if (flag("separate")) {
  for (ds in datasets[shown]) {
    path <- file.path(plot_dir, paste0("PARNN_vs_ground_truth_", ds, "_", hz, ".png"))
    grDevices::png(path, width = 1500, height = 950, res = 160)
    par(mar = c(4, 4.5, 2.5, 1))
    draw_panel(ds)
    legend("topleft", bty = "n", lty = c(1, 1, 1, 3), lwd = c(1.5, 2, 1.5, 1),
           col = c("black", "red", "blue", "darkorange"),
           legend = c("Observed history", "PARNN forecast", "Ground truth", "Forecast start"))
    grDevices::dev.off()
  }
}
