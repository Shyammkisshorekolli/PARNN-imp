# ============================================================================
# Data-driven seasonal-period detection (no period is hardcoded per dataset).
#
# For each candidate period p (2 .. n / min_cycles):
#   1. remove the trend with a centred moving average of length p,
#   2. average the detrended values by phase (position mod p),
#   3. score(p) = adjusted R^2 of that phase-mean model (penalises large p).
# Significance gate: the best observed score must exceed the (1 - alpha)
# quantile of the best score obtained on surrogate series simulated from a
# NON-seasonal ARIMA fitted to the same data (same trend / short memory,
# no seasonality). If it does not, S = 1 (non-seasonal).
# Period choice: the SHORTEST p whose score >= keep_ratio * best score and that
# also beats the gate. (Multiples of the true period score almost equally.)
# ============================================================================
.season_score <- function(y, p) {
  w <- if (p %% 2L == 0L) c(0.5, rep(1, p - 1L), 0.5) / p else rep(1, p) / p
  tr <- as.numeric(stats::filter(y, w, sides = 2))
  ok <- which(!is.na(tr))
  d  <- (y - tr)[ok]
  m  <- length(d)
  if (m - p < 2L) return(NA_real_)
  g   <- (ok - 1L) %% p
  sst <- sum((d - mean(d))^2)
  if (sst <= 0) return(0)
  gm  <- tapply(d, g, mean)
  gn  <- as.numeric(table(g))
  ssb <- sum(gn * (as.numeric(gm) - mean(d))^2)
  r2  <- ssb / sst
  1 - (1 - r2) * (m - 1) / (m - p)
}

.season_scan <- function(y, periods) vapply(periods, function(p) .season_score(y, p), numeric(1))

.surrogate <- function(y, fit) {
  s <- tryCatch(as.numeric(stats::simulate(fit, nsim = length(y), future = FALSE)),
                error = function(e) NULL)
  if (is.null(s) || anyNA(s)) {                      # fallback: shuffled increments
    s <- y[1] + c(0, cumsum(sample(diff(y))))
  }
  s
}

detect_period <- function(y, scfg = PARNN_CONFIG$seasonality, seed = 1L) {
  n    <- length(y)
  pmax <- min(scfg$max_period, floor(n / scfg$min_cycles))
  if (pmax < 2L) return(list(period = 1L, reason = "series too short", best_p = NA, best_score = NA, gate = NA))
  periods <- 2L:pmax
  sc      <- .season_scan(y, periods)
  best    <- max(sc, na.rm = TRUE)

  set.seed(seed)
  fit <- suppressWarnings(forecast::auto.arima(stats::ts(y), seasonal = FALSE,
                                               stepwise = TRUE, approximation = TRUE))
  sur_max <- vapply(seq_len(scfg$n_surrogates), function(i)
    max(.season_scan(.surrogate(y, fit), periods), na.rm = TRUE), numeric(1))
  gate <- as.numeric(stats::quantile(sur_max, 1 - scfg$alpha))

  best_p <- periods[which.max(sc)]
  if (best <= gate)
    return(list(period = 1L, reason = "not significant", best_p = best_p, best_score = best, gate = gate))
  ok <- which(sc >= scfg$keep_ratio * best & sc > gate)
  list(period = as.integer(min(periods[ok])), reason = "significant",
       best_p = best_p, best_score = best, gate = gate)
}

# Period for one (dataset, in-sample) pair, honouring config override.
resolve_period <- function(name, in_sample, cfg = PARNN_CONFIG, seed = cfg$seed) {
  ov <- cfg$seasonality$override[[name]]
  if (!is.null(ov)) return(list(period = as.integer(ov), reason = "override"))
  detect_period(in_sample, cfg$seasonality, seed)
}
