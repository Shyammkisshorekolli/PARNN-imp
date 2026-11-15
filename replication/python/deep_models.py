"""
Deep-learning baselines of the paper: NBeats, DeepAR (darts RNNModel/LSTM), TCN,
Transformer, plus Hybrid-3 (ARIMA + LSTM on ARIMA residuals).

Hyper-parameters are those of the authors' Deep_Learning_Models_Implementation.py.
Same protocol as the R code: fit on the in-sample window (everything before the test
window), forecast h steps, score on the test window (MASE with test-window lag-1
scaling, RMSE, SMAPE %), repeat with different seeds.

Needs results/task_table.csv and results/arima_export/ (written by run_baselines.R).

  python python/deep_models.py [--datasets A,B] [--horizons ST,MT,LT]
                               [--models NBeats,DeepAR,TCN,Transformer,Hybrid-3]
                               [--reps 10] [--epoch-scale 1.0] [--out results] [--quick] [--force]

Series are NOT rescaled by default (as in the authors' script). --scale applies a
MinMax scaler fitted on the in-sample window, which is usually more stable.
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

from darts import TimeSeries
from darts.models import NBEATSModel, RNNModel, TCNModel, TransformerModel
try:
    from darts.utils.likelihood_models import GaussianLikelihood
except Exception:  # newer darts layouts
    from darts.utils.likelihood_models.torch import GaussianLikelihood
from darts.dataprocessing.transformers import Scaler

PL = {"enable_progress_bar": False, "enable_model_summary": False,
      "enable_checkpointing": False, "logger": False}


def metrics(actual, pred):
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    den = np.mean(np.abs(np.diff(actual)))
    den = den if np.isfinite(den) and den > 0 else 1.0
    mase = np.mean(np.abs(actual - pred)) / den
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    smape = float(np.mean(np.abs(actual - pred) /
                          np.maximum((np.abs(actual) + np.abs(pred)) / 2, np.finfo(float).eps)) * 100)
    return mase, rmse, smape


def ep(base, scale):
    return max(1, int(round(base * scale)))


# ---- model factories: values copied from the authors' script -----------------
def make_nbeats(seed, es):
    return NBEATSModel(input_chunk_length=15, output_chunk_length=7, random_state=seed,
                       n_epochs=ep(100, es), pl_trainer_kwargs=PL)

def make_transformer(seed, es):
    return TransformerModel(input_chunk_length=15, output_chunk_length=7, batch_size=32,
                            n_epochs=ep(100, es), d_model=16, nhead=8, num_encoder_layers=2,
                            num_decoder_layers=2, dim_feedforward=128, dropout=0.1,
                            activation="relu", random_state=seed, pl_trainer_kwargs=PL)

def make_tcn(seed, es):
    return TCNModel(input_chunk_length=15, output_chunk_length=7, kernel_size=2, num_filters=4,
                    dilation_base=2, dropout=0, random_state=seed, likelihood=GaussianLikelihood(),
                    n_epochs=ep(100, es), pl_trainer_kwargs=PL)

def make_rnn(seed, es):  # "DeepAR" in the authors' script (LSTM RNNModel, no likelihood)
    return RNNModel(model="LSTM", hidden_dim=20, dropout=0, batch_size=16, n_epochs=ep(300, es),
                    optimizer_kwargs={"lr": 1e-3}, random_state=seed, training_length=20,
                    input_chunk_length=14, force_reset=True, pl_trainer_kwargs=PL)

FACTORY = {"NBeats": make_nbeats, "Transformer": make_transformer, "TCN": make_tcn,
           "DeepAR": make_rnn, "Hybrid-3": make_rnn}


def forecast_series(model_name, values, h, seed, es, scale):
    ts = TimeSeries.from_values(np.asarray(values, float).reshape(-1, 1))
    sc = None
    if scale:
        sc = Scaler(); ts = sc.fit_transform(ts)
    m = FACTORY[model_name](seed, es)
    m.fit(ts, verbose=False) if model_name != "NBeats" else m.fit([ts], verbose=False)
    pr = m.predict(h)
    if sc is not None:
        pr = sc.inverse_transform(pr)
    return pr.values().reshape(-1)[:h]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--horizons", default="ST,MT,LT")
    ap.add_argument("--models", default="NBeats,DeepAR,TCN,Transformer,Hybrid-3")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--epoch-scale", type=float, default=1.0)
    ap.add_argument("--scale", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.reps, a.epoch_scale = 2, 0.05
    tasks = pd.read_csv(a.tasks or os.path.join(a.out, "task_table.csv"))
    if a.datasets:
        tasks = tasks[tasks.Dataset.isin(a.datasets.split(","))]
    tasks = tasks[tasks.Horizon.isin(a.horizons.split(","))]
    os.makedirs(os.path.join(a.out, "deep"), exist_ok=True)

    for _, t in tasks.iterrows():
        y = pd.read_csv(t.path)[t.target].astype(float).values
        ins, test = y[: int(t.in_sample_n)], y[int(t.in_sample_n):]
        assert len(test) == t.h
        for mname in a.models.split(","):
            fn = os.path.join(a.out, "deep", f"{t.Dataset}_{t.Horizon}_{mname}_reps.csv")
            if os.path.exists(fn) and not a.force:
                continue
            t0, rows = time.time(), []
            try:
                if mname == "Hybrid-3":
                    ex = os.path.join(a.out, "arima_export", f"{t.Dataset}_{t.Horizon}")
                    res = pd.read_csv(ex + "_resid.csv")["resid"].values
                    afc = pd.read_csv(ex + "_fc.csv")["fc"].values
                for r in range(a.reps):
                    seed = 1000 + r
                    if mname == "Hybrid-3":
                        pr = afc + forecast_series("Hybrid-3", res, t.h, seed, a.epoch_scale, a.scale)
                    else:
                        pr = forecast_series(mname, ins, t.h, seed, a.epoch_scale, a.scale)
                    if not np.all(np.isfinite(pr)):
                        raise ValueError("non-finite forecast")
                    mase, rmse, smape = metrics(test, pr)
                    rows.append(dict(Dataset=t.Dataset, Horizon=t.Horizon, Model=mname, Rep=r + 1,
                                     MASE=mase, RMSE=rmse, SMAPE=smape))
            except Exception as e:  # keep going; the task is simply missing from the tables
                print(f"  FAILED {t.Dataset} {t.Horizon} {mname}: {e}")
                continue
            pd.DataFrame(rows).to_csv(fn, index=False)
            print(f"  [{t.Dataset} {t.Horizon}] {mname:11s} MASE={np.mean([r['MASE'] for r in rows]):.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
