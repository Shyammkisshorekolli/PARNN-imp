# ============================================================================
# Central configuration. Every tunable choice lives here; nothing is hardcoded
# elsewhere. Items marked [paper] come from Panja et al. (arXiv:2204.09640);
# [repo] from the authors' Models/PARNN.R; [assumption] where both are silent.
# ============================================================================
PARNN_CONFIG <- list(
  data_root = "Datasets",
  out_root  = "results",
  seed      = 42L,

  # [paper] grid for m (= p, lags of y) and l (= q, lags of ARIMA residuals)
  p_grid = 1:10,
  q_grid = 1:10,

  # [paper] validation window = 2 x test horizon; final forecast = mean of 500 nets
  validation_multiple = 2L,
  final_repeats       = 500L,
  # [assumption] the paper does not state the repeats used during grid search
  tuning_repeats      = 20L,

  # [repo] nnet defaults used by the authors' avnnet(): maxit=100, rang=0.5, decay=0
  nnet = list(maxit = 100L, decay = 0, rang = 0.5, MaxNWts = 5000L),

  # [repo] Box-Cox(0.5) on the series and on ARIMA residuals, then standardise.
  # Set boxcox_lambda = NULL to disable (the paper text only mentions standardisation).
  boxcox_lambda = 0.5,
  boxcox_resid  = TRUE,

  # [repo] future ARIMA feedback errors are fed as 0 in standardised-residual space
  future_resid = "zero_scaled",

  # [paper] forecast horizons per sampling unit; daily = fractions of n
  horizons = list(
    daily     = c(ST = 0.05, MT = 0.10, LT = 0.20),
    weekly    = c(ST = 13L,  MT = 26L,  LT = 52L),
    monthly   = c(ST = 6L,   MT = 12L,  LT = 24L),
    quarterly = c(ST = 4L,   MT = 8L,   LT = 12L)
  ),

  # [assumption] auto.arima defaults (AIC). Seasonal ARIMA is used only if the
  # detected period S <= this cap; forecast::auto.arima is impractical for long periods.
  arima_max_seasonal = 24L,

  # Data-driven seasonality detection (see R/seasonality.R). override = NULL -> auto.
  seasonality = list(
    min_cycles   = 8L,    # candidate period p needs >= this many full cycles
    max_period   = 400L,
    n_surrogates = 99L,   # non-seasonal ARIMA surrogates for the significance gate
    alpha        = 0.05,
    keep_ratio   = 0.9,   # choose the SHORTEST period scoring >= keep_ratio * best
    override     = NULL   # e.g. list(Tourism = 4L) forces a value for that dataset
  ),

  # [paper] 80% prediction interval from 1000 simulated paths
  pi = list(horizons = "MT", paths = 1000L, level = 80, nets = 100L),

  # [paper] stochastic models are repeated and reported as mean (sd)
  repetitions = 10L,

  cores = 1L
)
