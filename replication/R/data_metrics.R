# The 12 datasets used in the paper (Table 1). Only file / column / sampling
# unit are declared; length, seasonality and split sizes are derived from data.
DATASETS <- list(
  AMZN              = list(path = "Daily/AMZN.csv",               target = "Adj Close",  unit = "daily"),
  GOOG              = list(path = "Daily/GOOG.csv",               target = "Adj Close",  unit = "daily"),
  MSFT              = list(path = "Daily/MSFT.csv",               target = "Adj Close",  unit = "daily"),
  Births            = list(path = "Daily/Births.csv",             target = "x",          unit = "daily"),
  Colombia_Dengue   = list(path = "Weekly/Columbia_Dengue.csv",   target = "Cases",      unit = "weekly"),
  Colombia_Malaria  = list(path = "Weekly/Colombia_Malaria.csv",  target = "Cases",      unit = "weekly"),
  Venezuela_Dengue  = list(path = "Weekly/Venezuela_Dengue.csv",  target = "Cases",      unit = "weekly"),
  Venezuela_Malaria = list(path = "Weekly/Venezuela_Malaria.csv", target = "Cases",      unit = "weekly"),
  US_EPU_Index      = list(path = "Monthly/US_EPU_Index.csv",     target = "USEPUINDXM", unit = "monthly"),
  UK_Unemployment   = list(path = "Monthly/UK_Unemployment.csv",  target = "AURUKM",     unit = "monthly"),
  Russia_Exchange   = list(path = "Monthly/Russia_Exchange.csv",  target = "Exchange",   unit = "monthly"),
  Tourism           = list(path = "Quarterly/Aus_tourism.csv",    target = "Trips",      unit = "quarterly")
)

load_series <- function(name, cfg = PARNN_CONFIG) {
  d <- DATASETS[[name]]
  if (is.null(d)) stop("Unknown dataset: ", name)
  f <- file.path(cfg$data_root, d$path)
  if (!file.exists(f)) stop("Dataset file not found: ", f)
  raw <- read.csv(f, check.names = FALSE)
  if (!(d$target %in% names(raw)))
    stop(name, ": column '", d$target, "' missing. Have: ", paste(names(raw), collapse = ", "))
  y <- suppressWarnings(as.numeric(raw[[d$target]]))
  if (anyNA(y)) stop(name, ": NA / non-numeric values in target column")
  y
}

horizon_lengths <- function(unit, n, cfg = PARNN_CONFIG) {
  r <- cfg$horizons[[unit]]
  if (is.null(r)) stop("No horizon rule for unit ", unit)
  h <- if (unit == "daily") floor(r * n) else r
  stats::setNames(as.integer(h), names(r))
}

# ---- Metrics ---------------------------------------------------------------
# MASE (as REPORTED in the paper's Tables 3-5): mean|y - yhat| divided by the mean
# lag-1 naive error of the evaluated window itself, i.e. Metrics::mase(actual, predicted).
# Verified: this reproduces the paper's random-walk-with-drift MASE on all 12 datasets
# (see validation/check_rwd_against_paper.R). The denominator does not depend on the
# model, so it never changes which (p, q) wins on the validation window.
mase_test <- function(actual, pred) {
  den <- mean(abs(diff(actual)))
  if (!is.finite(den) || den == 0) den <- 1
  mean(abs(actual - pred)) / den
}
# MASE exactly as printed in Sec. 4.3 (in-sample seasonal-naive scaling with period S).
# Kept as a secondary column only; the paper's own numbers do not use it.
mase_insample <- function(actual, pred, train, S = 1L) {
  S <- max(1L, as.integer(S))
  den <- mean(abs(train[(S + 1L):length(train)] - train[1L:(length(train) - S)]))
  if (!is.finite(den) || den == 0) den <- 1
  mean(abs(actual - pred)) / den
}
rmse  <- function(actual, pred) sqrt(mean((actual - pred)^2))
smape <- function(actual, pred) {
  mean(abs(actual - pred) / pmax((abs(actual) + abs(pred)) / 2, .Machine$double.eps)) * 100
}
all_metrics <- function(actual, pred, train, S) {
  data.frame(MASE = mase_test(actual, pred), RMSE = rmse(actual, pred), SMAPE = smape(actual, pred),
             MASE_insample = mase_insample(actual, pred, train, S))
}
