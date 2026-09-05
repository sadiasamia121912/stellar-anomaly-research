"""
Notebook 3, Cells 6-7 equivalent: candidate feature extraction -> TRAIN-SPLIT-ONLY
importance ranking -> frozen feature set (M3).

Hard rule (M3, Step 8): feature discovery/candidate generation/selection uses the
TRAIN split only. Validation may choose among a small number of pre-defined
candidate sets if needed, but never discovers/ranks new features. The test
split's injection labels are NEVER used for feature selection, under any
circumstance. This script enforces that by construction: it never even loads
val/test injection labels.

Preprocessing note (new decision, needs sign-off / logging in the pre-registration
memo alongside the project's other locked decisions): the 13 candidate features
(src/features.py) were validated by the original pilot against per-curve,
min-max-to-[0,1]-normalized flux -- Dim_Frac/Bright_Frac's fixed 0.3 threshold is
only meaningful on that scale. The VAE pipeline's z-scored (GLOBAL_NORM_MEAN/STD)
flux is a DIFFERENT, global normalization used for a different purpose (Notebook
3's reconstruction-error model). Per-curve min-max scaling is therefore applied
here as a feature-extraction-specific step, downstream of the same common
detrended/gap-handled `flux` field every model starts from (R3's "one common
pipeline" governs detrending/gap-handling/outlier-clipping upstream of this;
which *representation* a given model's own feature layer computes on top of that
is a model-specific, documented choice -- exactly as the VAE's own binning step
is specific to the VAE). Flagged explicitly rather than silently assumed.

Usage: python src/feature_importance_freeze.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURE_NAMES, FEATURE_RATIONALE, morphological_fingerprint  # noqa: E402

LOCAL_BASE = Path(os.environ.get("STELLAR_LOCAL_DATA", r"C:\Users\User\stellar-anomaly-data-local"))
OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "feature_importance"
N_INTERP_POINTS = 1000  # matches the original pilot's fixed-length interpolation


def load_raw_star(tic_id: str, cache: dict) -> tuple[np.ndarray, np.ndarray]:
    if tic_id not in cache:
        npz_path = LOCAL_BASE / "processed" / "general" / f"TIC{tic_id}_processed.npz"
        with np.load(npz_path) as d:
            cache[tic_id] = (d["time"].copy(), d["flux"].copy())
    return cache[tic_id]


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


def to_feature_scale(time: np.ndarray, flux: np.ndarray, n_points: int = N_INTERP_POINTS) -> np.ndarray | None:
    """Uniform-time interpolation + per-curve min-max to [0, 1] (see module docstring)."""
    if len(time) < 10:
        return None
    fmin, fmax = np.min(flux), np.max(flux)
    if (fmax - fmin) < 1e-10:
        return None
    flux_norm = (flux - fmin) / (fmax - fmin)
    t_uniform = np.linspace(time.min(), time.max(), n_points)
    interp = np.interp(t_uniform, time, flux_norm)
    return np.clip(interp, 0, 1)


def extract_row_features(row: pd.Series, raw_cache: dict) -> np.ndarray | None:
    tic_id = str(row["tic_id"])
    time, flux = load_raw_star(tic_id, raw_cache)
    if row["is_injected"]:
        params = json.loads(row["params_json"])
        flux = INJECTION_FUNCS[row["anomaly_type"]](time, flux, **params)
    curve = to_feature_scale(time, flux)
    if curve is None:
        return None
    return morphological_fingerprint(curve)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    injection_manifest = pd.read_csv(LOCAL_BASE / "injections" / "injection_manifest.csv")
    train_rows = injection_manifest[injection_manifest["split"] == "train"].reset_index(drop=True)
    print(f"Train-split manifest rows: {len(train_rows)} (val/test rows are never loaded by this script)")

    processed_dir = LOCAL_BASE / "processed" / "general"
    available = {os.path.basename(p).replace("TIC", "").replace("_processed.npz", "")
                 for p in processed_dir.glob("*.npz")} if processed_dir.exists() else set()
    train_rows = train_rows[train_rows["tic_id"].astype(str).isin(available)].reset_index(drop=True)
    print(f"Train-split rows with local .npz available: {len(train_rows)}")
    if len(train_rows) == 0:
        print("No local data available yet -- run again once the Drive sync completes.")
        return

    raw_cache: dict = {}
    X, y, kept_idx = [], [], []
    for i, row in train_rows.iterrows():
        fp = extract_row_features(row, raw_cache)
        if fp is None or not np.all(np.isfinite(fp)):
            continue
        X.append(fp)
        y.append(bool(row["is_injected"]))
        kept_idx.append(i)
    X = np.array(X)
    y = np.array(y)
    print(f"Extracted features for {len(X)}/{len(train_rows)} rows "
          f"({int(y.sum())} injected, {int((~y).sum())} clean).")
    if len(X) < 20:
        print("Too few rows extracted to rank importance meaningfully yet -- aborting.")
        return

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    # Train-split-internal CV only -- this never touches val/test (M3).
    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    print(f"5-fold CV AUC (train-split internal, sanity check only): "
          f"{cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    clf.fit(X, y)
    perm = permutation_importance(clf, X, y, n_repeats=20, random_state=42, n_jobs=-1)

    ranking = sorted(
        zip(FEATURE_NAMES, perm.importances_mean, perm.importances_std, clf.feature_importances_),
        key=lambda t: t[1], reverse=True,
    )
    print("\nPermutation importance (train-split only), ranked:")
    for name, mean_imp, std_imp, gini_imp in ranking:
        print(f"  {name:16s} perm={mean_imp:+.4f} (+/-{std_imp:.4f})  gini={gini_imp:.4f}")

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_train_rows_used": int(len(X)),
        "n_injected": int(y.sum()),
        "n_clean": int((~y).sum()),
        "cv_auc_mean": float(cv_scores.mean()),
        "cv_auc_std": float(cv_scores.std()),
        "feature_extraction_scale_note": (
            "Per-curve min-max-to-[0,1], uniform 1000-point interpolation -- "
            "distinct from the VAE's global z-score normalization; see module "
            "docstring. NEEDS SIGN-OFF: log as an amendment in "
            "docs/pre_registration_memo.md before treating this ranking as final."
        ),
        "ranking": [
            {
                "feature": name,
                "permutation_importance_mean": float(mean_imp),
                "permutation_importance_std": float(std_imp),
                "gini_importance": float(gini_imp),
                "rationale": FEATURE_RATIONALE[name],
            }
            for name, mean_imp, std_imp, gini_imp in ranking
        ],
        "m3_compliance_note": (
            "Only train-split rows were loaded at any point in this script "
            "(val/test injection labels were never read) -- feature discovery/"
            "ranking is therefore train-split-only per M3."
        ),
        "frozen": False,
    }
    out_path = OUT_DIR / "feature_importance_ranking.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved ranking to {out_path}")
    print("NOT auto-frozen: review the ranking + the feature-extraction-scale note, "
          "then explicitly decide the final retained feature set (all 13, or a "
          "trimmed subset) and log that decision before Model B is trained.")


if __name__ == "__main__":
    main()
