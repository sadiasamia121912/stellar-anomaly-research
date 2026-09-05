"""
External Baseline 2 -- Webb et al. (2020)-style hand-crafted-feature baseline,
positioned against Model B (the project's own 13-dim fingerprint), per Step 7.

Genuine hand-crafted-feature baseline: tsfresh's generic feature-extraction
library (independent of the project's own curated 13 features -- that
independence is the point of a baseline) run on the SAME common preprocessed
representation every model in this project starts from (R3): the train-fit
z-scored `norm_flux` (GLOBAL_NORM_MEAN/STD from Notebook 2), trimmed to each
star's own valid (non-padded) span -- not a baseline-specific rescaling. A
Random Forest classifier (not Webb et al.'s original HDBSCAN+isolation-forest,
which is architecturally a cluster-then-rank pipeline, not a direct classifier
-- see Literature_Notes_Summary.md's reproduction note for #2, Webb et al.:
comparison matters more at the feature-representation level here) is trained
on TRAIN, tuned on VAL, evaluated once on TEST, matching R8's equal-treatment
rule and R4's validation-only tuning rule.

Feature set: tsfresh MinimalFCParameters (fast, ~10 features/series) by default
-- swap to EfficientFCParameters for a larger, still-tractable set once this
runs cleanly end-to-end; ComprehensiveFCParameters is not used (impractical at
this per-curve length and dataset size on CPU).

Usage: python src/baseline2_webb.py
"""

from __future__ import annotations

import json
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

LOCAL_BASE = Path(os.environ.get("STELLAR_LOCAL_DATA", r"C:\Users\User\stellar-anomaly-data-local"))
OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "baseline2_webb"
SEED = 42

TRANSIT_DURATION_DAYS = 0.125
TRANSIT_RAMP_FRAC = 0.10
FLARE_RISE_TAU_DAYS = 0.01
FLARE_DECAY_TAU_DAYS = 0.05


def inject_transit(time, flux, depth, t0, duration=TRANSIT_DURATION_DAYS, ramp_frac=TRANSIT_RAMP_FRAC):
    flux = flux.copy()
    ramp = duration * ramp_frac
    t_end = t0 + duration
    in_flat = (time >= t0 + ramp) & (time <= t_end - ramp)
    in_ingress = (time >= t0) & (time < t0 + ramp)
    in_egress = (time > t_end - ramp) & (time <= t_end)
    flux[in_flat] -= depth
    if ramp > 0:
        frac_in = (time[in_ingress] - t0) / ramp
        flux[in_ingress] -= depth * frac_in
        frac_out = (t_end - time[in_egress]) / ramp
        flux[in_egress] -= depth * frac_out
    return flux


def inject_flare(time, flux, amplitude, t_peak, rise_tau=FLARE_RISE_TAU_DAYS, decay_tau=FLARE_DECAY_TAU_DAYS):
    flux = flux.copy()
    dt = time - t_peak
    rise_mask = dt < 0
    decay_mask = dt >= 0
    flux[rise_mask] += amplitude * np.exp(dt[rise_mask] / rise_tau)
    flux[decay_mask] += amplitude * np.exp(-dt[decay_mask] / decay_tau)
    return flux


def inject_periodic(time, flux, amplitude, period_days, phase):
    flux = flux.copy()
    flux += amplitude * np.sin(2 * np.pi * time / period_days + phase)
    return flux


INJECTION_FUNCS = {"transit": inject_transit, "flare": inject_flare, "periodic": inject_periodic}


def load_common_preprocessed(tic_id: str, cache: dict) -> tuple[np.ndarray, np.ndarray]:
    """The one common pipeline's output (R3): detrended/gap-handled flux, pre-injection."""
    if tic_id not in cache:
        npz_path = LOCAL_BASE / "processed" / "general" / f"TIC{tic_id}_processed.npz"
        with np.load(npz_path) as d:
            cache[tic_id] = (d["time"].copy(), d["flux"].copy())
    return cache[tic_id]


def build_series(row: pd.Series, cache: dict, norm_mean: float, norm_std: float) -> np.ndarray | None:
    tic_id = str(row["tic_id"])
    time, flux = load_common_preprocessed(tic_id, cache)
    if row["is_injected"]:
        params = json.loads(row["params_json"])
        flux = INJECTION_FUNCS[row["anomaly_type"]](time, flux, **params)
    norm_flux = (flux - norm_mean) / norm_std
    if len(norm_flux) < 20:
        return None
    return norm_flux


def extract_tsfresh_features(rows: pd.DataFrame, cache: dict, norm_mean: float, norm_std: float,
                              fc_parameters) -> tuple[pd.DataFrame, np.ndarray, list]:
    from tsfresh import extract_features
    from tsfresh.utilities.dataframe_functions import impute

    long_rows = []
    kept_instance_ids = []
    labels = []
    for _, row in rows.iterrows():
        series = build_series(row, cache, norm_mean, norm_std)
        if series is None:
            continue
        inst_id = int(row["instance_id"])
        # Subsample long series for tractable CPU runtime -- tsfresh's per-series
        # cost scales with length; a few thousand points is ample for these
        # feature calculators (autocorrelation, FFT coefficients, entropy, etc.)
        if len(series) > 4000:
            idx = np.linspace(0, len(series) - 1, 4000).astype(int)
            series = series[idx]
        for t, v in enumerate(series):
            long_rows.append({"id": inst_id, "time": t, "value": v})
        kept_instance_ids.append(inst_id)
        labels.append(bool(row["is_injected"]))

    if not long_rows:
        return pd.DataFrame(), np.array([]), []

    long_df = pd.DataFrame(long_rows)
    features = extract_features(
        long_df, column_id="id", column_sort="time", column_value="value",
        default_fc_parameters=fc_parameters, disable_progressbar=True, n_jobs=0,
    )
    features = features.loc[kept_instance_ids]  # preserve label order
    impute(features)
    return features, np.array(labels), kept_instance_ids


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    injection_manifest = pd.read_csv(LOCAL_BASE / "injections" / "injection_manifest.csv")
    with open(LOCAL_BASE / "provenance" / "normalization_stats.json") as f:
        norm_stats = json.load(f)
    norm_mean, norm_std = norm_stats["global_norm_mean"], norm_stats["global_norm_std"]

    processed_dir = LOCAL_BASE / "processed" / "general"
    available = {p.name.replace("TIC", "").replace("_processed.npz", "")
                 for p in processed_dir.glob("*.npz")}

    from tsfresh.feature_extraction import MinimalFCParameters
    fc_parameters = MinimalFCParameters()

    cache: dict = {}
    splits = {}
    for split_name in ["train", "val", "test"]:
        rows = injection_manifest[
            (injection_manifest["split"] == split_name)
            & (injection_manifest["tic_id"].astype(str).isin(available))
        ]
        print(f"[{split_name}] extracting tsfresh features for {len(rows)} manifest rows...")
        X, y, ids = extract_tsfresh_features(rows, cache, norm_mean, norm_std, fc_parameters)
        print(f"[{split_name}] feature matrix: {X.shape}, positives: {int(y.sum())}/{len(y)}")
        splits[split_name] = (X, y, ids)

    X_train, y_train, _ = splits["train"]
    X_val, y_val, _ = splits["val"]
    X_test, y_test, _ = splits["test"]
    if len(X_train) == 0 or len(X_test) == 0:
        print("Not enough local data synced yet to fit/evaluate -- aborting.")
        return

    feature_cols = X_train.columns
    X_val = X_val.reindex(columns=feature_cols, fill_value=0.0)
    X_test = X_test.reindex(columns=feature_cols, fill_value=0.0)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import average_precision_score, precision_recall_curve

    # R4: hyperparameter tuning on validation only. Small, fixed, pre-specified
    # grid (equal-budget spirit of R8 -- this is a lightweight classical model,
    # not a deep net, so "equal tuning trials" is a short grid, not epochs/GPU-hours).
    grid = [
        {"n_estimators": 200, "max_depth": None},
        {"n_estimators": 200, "max_depth": 10},
        {"n_estimators": 500, "max_depth": None},
    ]
    best_auc_pr, best_params, best_clf = -1.0, None, None
    for params in grid:
        clf = RandomForestClassifier(random_state=SEED, n_jobs=-1, **params)
        clf.fit(X_train, y_train)
        val_scores = clf.predict_proba(X_val)[:, 1]
        auc_pr = average_precision_score(y_val, val_scores) if len(set(y_val)) > 1 else float("nan")
        print(f"  val AUC-PR for {params}: {auc_pr:.4f}")
        if auc_pr > best_auc_pr:
            best_auc_pr, best_params, best_clf = auc_pr, params, clf

    test_scores = best_clf.predict_proba(X_test)[:, 1]
    test_auc_pr = average_precision_score(y_test, test_scores)
    precision, recall, thresholds = precision_recall_curve(y_test, test_scores)

    print(f"\nSelected config (validation-only): {best_params} (val AUC-PR={best_auc_pr:.4f})")
    print(f"TEST AUC-PR: {test_auc_pr:.4f}")

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": "External Baseline 2 -- Webb-style (tsfresh hand-crafted features + Random Forest)",
        "feature_set": "tsfresh MinimalFCParameters",
        "n_features": int(X_train.shape[1]),
        "selected_hyperparameters": best_params,
        "val_auc_pr_at_selection": None if np.isnan(best_auc_pr) else float(best_auc_pr),
        "test_auc_pr": float(test_auc_pr),
        "n_train": int(len(y_train)), "n_val": int(len(y_val)), "n_test": int(len(y_test)),
        "note": "PRELIMINARY -- run against however much of the general pool was locally "
                "synced at run time, not necessarily the full split. Re-run once the full "
                "local mirror + real S2/S8-excluded split is confirmed complete.",
    }
    with open(OUT_DIR / "baseline2_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {OUT_DIR / 'baseline2_result.json'}")


if __name__ == "__main__":
    main()
