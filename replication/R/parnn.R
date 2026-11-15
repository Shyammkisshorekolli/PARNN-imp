# ============================================================================
# PARNN(m, k, l)  --  Panja, Chakraborty, Kumar, Hadid (arXiv:2204.09640)
#   y_t = f(y_{t-1..t-m}, e_{t-1..t-l}),  e_t = ARIMA one-step residuals,
#   f = single-hidden-layer network, k = ceil((m + l + 1) / 2) hidden units,
#   final forecast = mean of `repeats` networks, multi-step forecasts recursive.
# Here m = p (lags of the series) and l = q (lags of ARIMA residuals).
# ============================================================================

# ---- Box-Cox (sign-aware so that ARIMA residuals, which can be <= 0, work) --
bc_fwd <- function(x, l) {
  if (is.null(l)) return(x)
  if (l == 0) return(log(x))
  (sign(x) * abs(x)^l - 1) / l
}
bc_inv <- function(z, l) {
  if (is.null(l)) return(z)
  if (l == 0) return(exp(z))
  pmax(l * z + 1, 1e-8)^(1 / l)
}

# ---- Stage 1: ARIMA (auto.arima, AIC) and its in-sample residuals -----------
parnn_arima <- function(y, S, cfg = PARNN_CONFIG) {
  S_use <- if (S > 1L && S <= cfg$arima_max_seasonal) as.integer(S) else 1L
  fit <- suppressWarnings(forecast::auto.arima(stats::ts(y, frequency = S_use),
                                               seasonal = S_use > 1L))
  e <- as.numeric(stats::residuals(fit))
  e[!is.finite(e)] <- 0
  ord <- forecast::arimaorder(fit)
  list(resid = e, aic = fit$aic, S_used = S_use,
       order = sprintf("(%d,%d,%d)%s", ord[1], ord[2], ord[3],
                       if (length(ord) >= 7 && S_use > 1L)
                         sprintf("(%d,%d,%d)[%d]", ord[4], ord[5], ord[6], S_use) else ""))
}

# ---- Transform + standardise (statistics from the fitting window only) -------
parnn_scale <- function(y, e, cfg = PARNN_CONFIG) {
  lam <- cfg$boxcox_lambda
  if (!is.null(lam) && any(y <= 0)) lam <- NULL            # Box-Cox needs y > 0
  yt <- bc_fwd(y, lam)
  et <- if (isTRUE(cfg$boxcox_resid)) bc_fwd(e, lam) else e
  sdv <- function(v) { s <- stats::sd(v); if (!is.finite(s) || s <= 0) 1 else s }
  cy <- mean(yt); sy <- sdv(yt); ce <- mean(et); se <- sdv(et)
  list(ys = (yt - cy) / sy, es = (et - ce) / se, cy = cy, sy = sy, ce = ce, se = se, lambda = lam,
       e_zero_scaled = if (identical(cfg$future_resid, "zero_raw"))
         (bc_fwd(0, if (isTRUE(cfg$boxcox_resid)) lam else NULL) - ce) / se else 0)
}

# ---- Lagged design matrix: [y_{t-1..t-p}, e_{t-1..t-q}] -> y_t ---------------
parnn_xy <- function(ys, es, p, q) {
  n <- length(ys); L <- max(p, q); idx <- (L + 1L):n
  X <- cbind(do.call(cbind, lapply(seq_len(p), function(i) ys[idx - i])),
             do.call(cbind, lapply(seq_len(q), function(i) es[idx - i])))
  list(X = X, y = ys[idx])
}

# ---- Ensemble of `repeats` networks, stored as stacked weight matrices ------
ens_fit <- function(X, y, k, repeats, nn, seed) {
  set.seed(seed)
  d <- ncol(X)
  W1 <- vector("list", repeats); W2 <- vector("list", repeats)
  for (i in seq_len(repeats)) {
    m <- nnet::nnet(x = X, y = y, size = k, linout = TRUE, trace = FALSE,
                    maxit = nn$maxit, decay = nn$decay, rang = nn$rang, MaxNWts = nn$MaxNWts)
    w <- m$wts
    W1[[i]] <- matrix(w[1:(k * (d + 1L))], nrow = k, byrow = TRUE)   # k x (d+1): bias, inputs
    W2[[i]] <- w[(k * (d + 1L) + 1L):(k * (d + 1L) + k + 1L)]         # bias, hidden
  }
  list(W1 = do.call(rbind, W1), W2 = do.call(rbind, W2), R = repeats, k = k, d = d)
}

# Mean prediction of the ensemble for each row of X (logistic hidden, linear output).
# Equals rowMeans(sapply(nets, predict, X)) because the output layer is linear.
ens_predict <- function(E, X) {
  X <- matrix(X, ncol = E$d)
  H <- stats::plogis(cbind(1, X) %*% t(E$W1))
  w2 <- as.vector(t(E$W2[, -1, drop = FALSE]))
  as.numeric((H %*% w2 + sum(E$W2[, 1])) / E$R)
}
ens_subset <- function(E, nets) {
  nets <- min(nets, E$R)
  list(W1 = E$W1[seq_len(nets * E$k), , drop = FALSE], W2 = E$W2[seq_len(nets), , drop = FALSE],
       R = nets, k = E$k, d = E$d)
}

# ---- Fit PARNN(p, q) ---------------------------------------------------------
parnn_fit <- function(y, S, p, q, repeats, seed, cfg = PARNN_CONFIG, arima = NULL) {
  if (is.null(arima)) arima <- parnn_arima(y, S, cfg)
  sc <- parnn_scale(y, arima$resid, cfg)
  xy <- parnn_xy(sc$ys, sc$es, p, q)
  k  <- as.integer(ceiling((p + q + 1) / 2))
  E  <- ens_fit(xy$X, xy$y, k, repeats, cfg$nnet, seed)
  list(E = E, p = p, q = q, k = k, sc = sc, arima = arima,
       sigma = stats::sd(xy$y - ens_predict(E, xy$X)))
}

# ---- Recursive multi-step forecast (+ optional simulated prediction interval)
parnn_forecast <- function(fit, h, cfg = PARNN_CONFIG, paths = 0L, level = 80, nets = 100L) {
  p <- fit$p; q <- fit$q; sc <- fit$sc
  ylag <- rev(utils::tail(sc$ys, p)); elag <- rev(utils::tail(sc$es, q))
  f0 <- sc$e_zero_scaled
  pred <- numeric(h)
  for (s in seq_len(h)) {
    pred[s] <- ens_predict(fit$E, matrix(c(ylag, elag), nrow = 1L))
    ylag <- c(pred[s], ylag)[seq_len(p)]
    elag <- c(f0, elag)[seq_len(q)]               # future ARIMA feedback error = 0 (repo)
  }
  back <- function(v) bc_inv(v * sc$sy + sc$cy, sc$lambda)
  out <- list(mean = back(pred))
  if (paths > 0L) {
    Es <- ens_subset(fit$E, nets)
    Y <- matrix(rep(rev(utils::tail(sc$ys, p)), each = paths), nrow = paths)
    Ee <- matrix(rep(rev(utils::tail(sc$es, q)), each = paths), nrow = paths)
    sim <- matrix(NA_real_, paths, h)
    for (s in seq_len(h)) {
      yy <- ens_predict(Es, cbind(Y, Ee)) + stats::rnorm(paths, 0, fit$sigma)   # Gaussian eps_t
      sim[, s] <- yy
      Y  <- cbind(yy, Y)[, seq_len(p), drop = FALSE]
      Ee <- cbind(f0, Ee)[, seq_len(q), drop = FALSE]
    }
    sim <- back(sim)
    a <- (100 - level) / 200
    out$lower <- apply(sim, 2, stats::quantile, probs = a,     type = 8)
    out$upper <- apply(sim, 2, stats::quantile, probs = 1 - a, type = 8)
  }
  out
}

# ---- Parallel map that also works on Windows (PSOCK) -------------------------
par_map <- function(X, FUN, cores = 1L) {
  if (cores <= 1L || length(X) < 2L) return(lapply(X, FUN))
  if (.Platform$OS.type == "windows") {
    cl <- parallel::makeCluster(cores); on.exit(parallel::stopCluster(cl), add = TRUE)
    parallel::clusterCall(cl, function(wd, files) {
      setwd(wd); suppressMessages({library(forecast); library(nnet)})
      for (f in files) source(f)
    }, getwd(), getOption("parnn.files"))
    parallel::parLapply(cl, X, FUN)
  } else parallel::mclapply(X, FUN, mc.cores = cores)
}

# One grid point: fit on training window, forecast the validation window, score by MASE.
.tune_one <- function(job) {
  r <- tryCatch({
    fit <- parnn_fit(job$train, job$S, job$p, job$q, job$repeats, job$seed, job$cfg, job$arima)
    pr  <- parnn_forecast(fit, length(job$val), job$cfg)$mean
    sc  <- mase_test(job$val, pr)
    if (!is.finite(sc)) Inf else sc
  }, error = function(e) Inf)
  data.frame(p = job$p, q = job$q, k = as.integer(ceiling((job$p + job$q + 1) / 2)), val_MASE = r)
}

parnn_tune <- function(train, val, S, seed, cfg = PARNN_CONFIG, verbose = TRUE) {
  arima <- parnn_arima(train, S, cfg)
  grid  <- expand.grid(p = cfg$p_grid, q = cfg$q_grid)
  grid  <- grid[length(train) - pmax(grid$p, grid$q) >= 20L, ]   # need >= 20 training rows
  if (nrow(grid) == 0L) stop("Training window too short for the (p, q) grid")
  jobs <- lapply(seq_len(nrow(grid)), function(i)
    list(p = grid$p[i], q = grid$q[i], train = train, val = val, S = S, arima = arima,
         repeats = cfg$tuning_repeats, seed = seed + i, cfg = cfg))
  res <- do.call(rbind, par_map(jobs, .tune_one, cfg$cores))
  res <- res[order(res$val_MASE, res$p, res$q), ]
  rownames(res) <- NULL
  res
}
