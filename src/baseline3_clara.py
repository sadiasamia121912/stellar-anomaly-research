"""
External Baseline 3 -- CLARA (Dasgupta 2025)-style Unsupervised Random Forest,
positioned against Model D (the full proposed pipeline), per Step 7 / Part C.

Reproduces the URF anomaly-scoring mechanism common to Crake & Martinez-Galarza
(2023) and CLARA: (1) build a feature vector per light curve as flux samples
stacked with a Lomb-Scargle periodogram; (2) train a RandomForestClassifier to
distinguish REAL curves from a SYNTHETIC contrastive class built by
independently shuffling each feature dimension across the real set (destroys
cross-feature correlations while preserving each feature's marginal
distribution -- the standard URF synthetic-set trick, not CLARA's own
batman-based URF4 variant, which needs a transit-modeling dependency not
otherwise used in this project; documented here as a scoped simplification of
the full CLARA method, consistent with "simplified... baseline" language in
Step 7); (3) score any curve's anomalousness via the terminal-node population
heuristic (Baron & Poznanski 2017): similarity = mean fraction of REAL
training points sharing a leaf across all trees; anomaly score = 1 - similarity.

Evaluated on the SAME Track A injection-recovery ground truth as every other
model/baseline (R5/R8): AUC-PR of the anomaly score against the is_injected
label on the test split. The reference-library cosine-similarity descriptive
score (CLARA's other half) is added separately once processed/reference is
fully synced locally.

Usage: python src/baseline3_clara.py
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
OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "baseline3_clara"
SEED = 42
N_FLUX_SAMPLES = 3000   # per Crake & Martinez-Galarza / CLARA's 3000+1000 stack
N_LS_FREQS = 1000

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
    if tic_id not in cache:
        npz_path = LOCAL_BASE / "processed" / "general" / f"TIC{tic_id}_processed.npz"
        with np.load(npz_path) as d:
            cache[tic_id] = (d["time"].copy(), d["flux"].copy())
    return cache[tic_id]


LS_MAX_INPUT_POINTS = 3000  # subsample before LS -- astropy's LombScargle cost
                            # scales with input length; full-resolution (tens of
                            # thousands of points/star) is intractable on CPU at
                            # this dataset size. 3000 evenly-spaced points still
                            # comfortably resolves the periods this project's
                            # injected/target signals span (see Step 6's period
                            # ranges), at a small, documented resolution cost.


def flux_ls_vector(time: np.ndarray, flux: np.ndarray) -> np.ndarray | None:
    from astropy.timeseries import LombScargle

    if len(time) < 20:
        return None
    idx = np.linspace(0, len(flux) - 1, N_FLUX_SAMPLES).astype(int)
    flux_stack = flux[idx]
    if len(time) > LS_MAX_INPUT_POINTS:
        ls_input_idx = np.linspace(0, len(time) - 1, LS_MAX_INPUT_POINTS).astype(int)
        ls_time, ls_flux = time[ls_input_idx], flux[ls_input_idx]
    else:
        ls_time, ls_flux = time, flux
    try:
        freq, power = LombScargle(ls_time, ls_flux).autopower(
            maximum_frequency=1.0 / (2 * np.median(np.diff(ls_time))), method="fast",
        )
    except Exception:
        return None
    if len(power) < N_LS_FREQS:
        power = np.pad(power, (0, N_LS_FREQS - len(power)))
    else:
        ls_idx = np.linspace(0, len(power) - 1, N_LS_FREQS).astype(int)
        power = power[ls_idx]
    return np.concatenate([flux_stack, power]).astype(np.float32)


def build_vector(row: pd.Series, cache: dict) -> np.ndarray | None:
    tic_id = str(row["tic_id"])
    time, flux = load_common_preprocessed(tic_id, cache)
    if row["is_injected"]:
        params = json.loads(row["params_json"])
        flux = INJECTION_FUNCS[row["anomaly_type"]](time, flux, **params)
    return flux_ls_vector(time, flux)


def terminal_node_similarity(forest, leaves_real: np.ndarray, leaves_query: np.ndarray) -> np.ndarray:
    """Mean, across trees, of the fraction of REAL training points sharing each query point's leaf."""
    n_trees = leaves_real.shape[1]
    n_query = leaves_query.shape[0]
    sim = np.zeros(n_query, dtype=np.float64)
    for t in range(n_trees):
        real_leaves_t = leaves_real[:, t]
        counts = pd.Series(real_leaves_t).value_counts()
        n_real = len(real_leaves_t)
        sim += leaves_query[:, t].astype(np.int64).map(lambda leaf: counts.get(leaf, 0) / n_real) \
            if False else np.array([counts.get(leaf, 0) / n_real for leaf in leaves_query[:, t]])
    return sim / n_trees


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    injection_manifest = pd.read_csv(LOCAL_BASE / "injections" / "injection_manifest.csv")
    processed_dir = LOCAL_BASE / "processed" / "general"
    available = {p.name.replace("TIC", "").replace("_processed.npz", "")
                 for p in processed_dir.glob("*.npz")}

    cache: dict = {}

    def build_split(split_name: str, cap: int | None = None):
        rows = injection_manifest[
            (injection_manifest["split"] == split_name)
            & (injection_manifest["tic_id"].astype(str).isin(available))
        ]
        if cap:
            rows = rows.sample(n=min(cap, len(rows)), random_state=SEED)
        X, y = [], []
        for _, row in rows.iterrows():
            vec = build_vector(row, cache)
            if vec is None or not np.all(np.isfinite(vec)):
                continue
            X.append(vec)
            y.append(bool(row["is_injected"]))
        return np.array(X), np.array(y)

    # Capped for a fast preliminary signal (Lomb-Scargle is the bottleneck even
    # after the LS_MAX_INPUT_POINTS speedup); re-run with cap=None for the final,
    # full-split numbers once this preliminary pass is reviewed.
    PRELIMINARY_CAP = 1500
    print(f"Building TRAIN feature vectors (flux + Lomb-Scargle stack), capped at {PRELIMINARY_CAP}...")
    X_train, y_train = build_split("train", cap=PRELIMINARY_CAP)
    print(f"  train: {X_train.shape}, {int(y_train.sum())} injected / {len(y_train)}")
    print(f"Building TEST feature vectors, capped at {PRELIMINARY_CAP}...")
    X_test, y_test = build_split("test", cap=PRELIMINARY_CAP)
    print(f"  test: {X_test.shape}, {int(y_test.sum())} injected / {len(y_test)}")

    if len(X_train) < 20 or len(X_test) < 20:
        print("Not enough local data synced yet -- aborting.")
        return

    # URF training set: REAL = train-split clean (non-injected) curves only, so
    # the "normal" population the forest learns is not itself contaminated by
    # injected anomalies (mirrors CLARA/Crake's real-vs-synthetic design intent).
    real_mask = ~y_train
    X_real = X_train[real_mask]
    print(f"Real (non-injected) training curves: {len(X_real)}")

    rng = np.random.RandomState(SEED)
    X_synthetic = np.column_stack([rng.permutation(X_real[:, j]) for j in range(X_real.shape[1])])

    X_urf = np.vstack([X_real, X_synthetic])
    y_urf = np.concatenate([np.ones(len(X_real)), np.zeros(len(X_synthetic))])

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score

    print("Training Unsupervised Random Forest (real vs. shuffled-synthetic)...")
    urf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    urf.fit(X_urf, y_urf)
    urf_train_auc = roc_auc_score(y_urf, urf.predict_proba(X_urf)[:, 1])
    print(f"  URF real-vs-synthetic training AUC (sanity check): {urf_train_auc:.4f}")

    print("Scoring TEST instances via terminal-node similarity to REAL training curves...")
    leaves_real = urf.apply(X_real)
    leaves_test = urf.apply(X_test)
    similarity = terminal_node_similarity(urf, leaves_real, leaves_test)
    anomaly_score = 1.0 - similarity

    test_auc_pr = average_precision_score(y_test, anomaly_score)
    test_auc_roc = roc_auc_score(y_test, anomaly_score)
    print(f"\nTEST AUC-PR (anomaly_score vs. is_injected): {test_auc_pr:.4f}")
    print(f"TEST AUC-ROC: {test_auc_roc:.4f}")

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": "External Baseline 3 -- CLARA-style (URF + terminal-node similarity)",
        "simplification_note": "Synthetic class built via per-feature independent shuffling "
                                "of real training curves (standard URF trick), not CLARA's "
                                "own batman-based URF4 variant -- see module docstring.",
        "feature_vector": f"{N_FLUX_SAMPLES} flux samples + {N_LS_FREQS} Lomb-Scargle power values",
        "n_real_train_curves": int(len(X_real)),
        "urf_train_auc_sanity_check": float(urf_train_auc),
        "test_auc_pr": float(test_auc_pr),
        "test_auc_roc": float(test_auc_roc),
        "n_test": int(len(y_test)),
        "note": "PRELIMINARY -- reference-library cosine-similarity descriptive score "
                "(CLARA's other evaluation half) not yet added; needs processed/reference "
                "fully synced. Reflects whatever local data was available at run time.",
    }
    with open(OUT_DIR / "baseline3_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {OUT_DIR / 'baseline3_result.json'}")


if __name__ == "__main__":
    main()
