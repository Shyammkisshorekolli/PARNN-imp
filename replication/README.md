# PARNN replication (R) — stage 1: the PARNN model itself

Paper: Panja, Chakraborty, Kumar, Hadid, *Probabilistic AutoRegressive Neural Networks for
Accurate Long-range Forecasting* (arXiv:2204.09640). Authors' code: `Models/PARNN.R` in the
original repo (not included here; this is a from-scratch re-implementation).

## Run

```
Rscript run_parnn.R --quick --datasets=Tourism --horizons=ST   # smoke test (~seconds)
Rscript run_parnn.R --cores=4                                  # full: 12 datasets x ST/MT/LT
Rscript run_parnn.R --datasets=AMZN,GOOG --horizons=MT --force
Rscript validation/check_rwd_against_paper.R                   # data/split/metric sanity check
```
Needs R packages `forecast` and `nnet`. Results are written per task to `results/parnn/`
(`*_metrics.csv`, `*_tuning.csv`, `*_forecast.csv` with 80% interval for MT) and summarised in
`results/parnn_all_metrics.csv` and `results/table2_tuned_hyperparameters.csv`.
Runs are resumable (finished tasks are skipped). `--quick` output is for smoke tests only.

Everything tunable is in `R/config.R`. Nothing from the paper's Table 2 or Tables 3-5 is used
by the model; (m, k, l) are searched on the validation window.

## What the paper does vs. what its numbers show

I checked the paper against the authors' repo and recomputed published numbers:

| Item | Paper prose | What reproduces the paper's numbers | Used here |
|---|---|---|---|
| Splits / horizons | train, validation = 2h, test = h | RWD RMSE matches Table 4 on all 12 datasets (<0.4%) | as paper |
| MASE | Sec. 4.3 in-sample seasonal-naive scaling | test-window lag-1 scaling (`Metrics::mase(actual, pred)`); matches Table 4 RWD MASE on all 12 (<0.12%) | `MASE` column = this; `MASE_insample` = Sec. 4.3 formula |
| ARIMA input | "standardised series" | raw series, non-seasonal for weekly data: Colombia Malaria MT gives ARIMA(1,1,2), AIC 8168.85 exactly as quoted, on the n-h window | raw series |
| Hidden units | ceil((m+l+1)/2) | Table 2 is consistent with `ceiling`; authors' code uses `round` | `ceiling` |
| Future ARIMA residual inputs | not stated | authors' code feeds 0 in standardised-residual space | same |
| Box-Cox 0.5 | only in the prediction-interval section | authors' code applies it to y and residuals for the whole fit | same (switch in config) |
| nnet settings | not stated | authors' code uses nnet defaults (maxit=100, rang=0.5, decay=0) | same |
| Tuning repeats | not stated | not stated | 20 (assumption, configurable) |
| Stock column | "adjusted close" | — | `Adj Close` |

Because the MASE denominator is the same for every (p, q) on a given window, the seasonal period
cannot change which (p, q) is selected.

## Seasonality (detected, not hardcoded)

`R/seasonality.R`: for each candidate period, detrend with a centred moving average, average by
phase, score by adjusted R^2; the best score must beat the 95% quantile of the same score on
99 non-seasonal-ARIMA surrogate series. Detected once per dataset on the data before the longest test
window. It only decides whether `auto.arima` is seasonal (and only if S <= `arima_max_seasonal`) and
the `MASE_insample` column. Outputs: `results/seasonality_detected.csv`. Override per dataset in config.

## Stages 2-3: baselines, deep models, aggregation

Run order (all from the project folder):

```
Rscript run_parnn.R --cores=4              # PARNN, 10 repetitions per task -> results/parnn/
Rscript run_baselines.R                    # RWD, ETS, ARIMA, TBATS, MLP, ARNN, Hybrid-1, Hybrid-2
                                           #   also writes results/task_table.csv + results/arima_export/
pip install "u8darts[torch]"
python python/deep_models.py               # NBeats, DeepAR, TCN, Transformer, Hybrid-3 (needs the two files above)
Rscript aggregate_results.R                # results/aggregate/: mean (sd) tables, Friedman, MCB plots
```
Add `--quick` to any step for a smoke test (writes to `results_quick/`; numbers are meaningless).
All steps are resumable. 14 models x 12 datasets x 3 horizons = 36 tasks gives the paper's
F(13, 455) Friedman reference distribution.

Notes and limits
* Baselines are fitted on the in-sample window with library defaults, as in the authors' script.
  Stochastic models (MLP, ARNN, Hybrid-1/2, all deep models, PARNN) are repeated 10 times; mean (sd) reported.
* MLP uses `nnfor::mlp` if installed, otherwise an internal 20-net stand-in (printed as a NOTE).
  `nnfor` could not be installed in my sandbox, so the `nnfor` path is untested.
* Hybrid-1 (Zhang ARIMA-ANN) is my implementation (residual MLP, lags/size picked on the last 20% of residuals).
* Deep models use the authors' hyper-parameters and no input scaling (`--scale` adds MinMax scaling).
  Checkpointing/TensorBoard options were dropped. The 300-epoch DeepAR on Births is slow on CPU.
* Friedman / Iman-Davenport statistic, Nemenyi critical distance and MCB plots are computed per metric
  from the result files; only models with results on every task are included.
* No paper table values are used anywhere except `validation/` (RWD, Table 4) as a sanity check.
