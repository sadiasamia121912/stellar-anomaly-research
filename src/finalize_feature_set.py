"""
Finalizes the frozen hand-crafted feature set -- the decision queued in
feature_importance_freeze.py's output, now actually made rather than left open.

Two checks, not one, because Track A importance alone is the wrong sole basis
for this decision:

  (a) M3-compliant validation comparison: train each CANDIDATE feature set on
      TRAIN, compare on VAL only (never test), pick by Track A (injection-
      recovery) AUC-PR.
  (b) Track B relevance check: the four features that scored near-zero for
      Track A (Dom_Freq, Rise_Fall_Ratio, N_Peaks, N_Troughs) are exactly the
      periodicity-shaped features -- and Track A's ground truth is injection
      recovery against synthetic transit/flare/periodic signals, NOT
      reference-class separation. A feature can be useless for "is this curve
      injected" while being essential for "which known class does this
      resemble" (pulsators/rotational variables/eclipsing binaries are
      defined almost entirely by periodicity). Trimming on Track A evidence
      alone risks silently crippling Track B (M4's Track A/B separation rule
      says exactly this: injection-recovery evidence must never be used alone
      to support -- or, by the same logic, to cut against -- a Track B
      claim). So this script also tests each candidate feature set's
      contribution to discriminating the 5 reference-library classes, and the
      final decision requires a feature to be dispensable on BOTH counts
      before it is dropped.

Usage: python src/finalize_feature_set.py
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
from feature_importance_freeze import extract_row_features, to_feature_scale, LOCAL_BASE  # noqa: E402
from features import FEATURE_NAMES, FEATURE_RATIONALE, morphological_fingerprint  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "feature_importance"

# Candidate sets, pre-defined (M3: "validation may choose among a small
# number of pre-defined candidate sets" -- not an open-ended search).
WEAKEST_4 = ["Dom_Freq", "Rise_Fall_Ratio", "N_Peaks", "N_Troughs"]
NEXT_WEAKEST_2 = ["Skewness", "Kurtosis"]
CANDIDATES = {
    "all_13": FEATURE_NAMES,
    "top_9_drop_weakest_4": [f for f in FEATURE_NAMES if f not in WEAKEST_4],
    "top_7_drop_weakest_6": [f for f in FEATURE_NAMES if f not in WEAKEST_4 + NEXT_WEAKEST_2],
}


def build_track_a_matrix(split_name: str):
    injection_manifest = pd.read_csv(LOCAL_BASE / "injections" / "injection_manifest.csv")
    processed_dir = LOCAL_BASE / "processed" / "general"
    available = {p.name.replace("TIC", "").replace("_processed.npz", "")
                 for p in processed_dir.glob("*.npz")}
    rows = injection_manifest[
        (injection_manifest["split"] == split_name) & (injection_manifest["tic_id"].astype(str).isin(available))
    ]
    raw_cache: dict = {}
    X, y = [], []
    for _, row in rows.iterrows():
        fp = extract_row_features(row, raw_cache)
        if fp is None or not np.all(np.isfinite(fp)):
            continue
        X.append(fp)
        y.append(bool(row["is_injected"]))
    return pd.DataFrame(X, columns=FEATURE_NAMES), np.array(y)


def build_track_b_matrix():
    """13-dim fingerprint + class label for every reference-library star with local data."""
    manifest = pd.read_csv(LOCAL_BASE / "reference_library" / "reference_library_FINAL_manifest.csv")
    ref_dir = LOCAL_BASE / "processed" / "reference"
    X, y = [], []
    for _, row in manifest.iterrows():
        tic_id = str(row["TIC_ID"])
        npz_path = ref_dir / f"TIC{tic_id}_processed.npz"
        if not npz_path.exists():
            continue
        with np.load(npz_path) as d:
            time, flux = d["time"], d["flux"]
        curve = to_feature_scale(time, flux)
        if curve is None:
            continue
        fp = morphological_fingerprint(curve)
        if not np.all(np.isfinite(fp)):
            continue
        X.append(fp)
        y.append(row["class"])
    return pd.DataFrame(X, columns=FEATURE_NAMES), np.array(y)


def main():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    print("=== Check (a): Track A -- validation-only comparison of candidate feature sets ===")
    X_train, y_train = build_track_a_matrix("train")
    X_val, y_val = build_track_a_matrix("val")
    print(f"Train rows: {len(X_train)}, Val rows: {len(X_val)}")

    track_a_results = {}
    for name, cols in CANDIDATES.items():
        clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        clf.fit(X_train[cols], y_train)
        val_score = average_precision_score(y_val, clf.predict_proba(X_val[cols])[:, 1])
        track_a_results[name] = val_score
        print(f"  {name:22s} (n={len(cols)}): VAL AUC-PR = {val_score:.4f}")

    print("\n=== Check (b): Track B -- do the weak-for-Track-A features matter for class separation? ===")
    X_ref, y_ref = build_track_b_matrix()
    print(f"Reference-library rows with local data: {len(X_ref)}, classes: {sorted(set(y_ref))}")

    track_b_results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, cols in CANDIDATES.items():
        clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        scores = cross_val_score(clf, X_ref[cols], y_ref, cv=cv, scoring="accuracy")
        track_b_results[name] = float(scores.mean())
        print(f"  {name:22s} (n={len(cols)}): 5-fold CV class-accuracy = {scores.mean():.4f} (+/-{scores.std():.4f})")

    # Per-feature importance specifically for the Track B (class-separation) task,
    # to see whether the "Track-A-useless" features are pulling weight here.
    clf_full = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    clf_full.fit(X_ref[FEATURE_NAMES], y_ref)
    from sklearn.inspection import permutation_importance
    perm_b = permutation_importance(clf_full, X_ref[FEATURE_NAMES], y_ref, n_repeats=20, random_state=42, n_jobs=-1)
    track_b_importance = dict(zip(FEATURE_NAMES, perm_b.importances_mean.tolist()))
    print("\nPer-feature Track B (class-separation) permutation importance:")
    for f in FEATURE_NAMES:
        flag = " <-- near-zero for Track A" if f in WEAKEST_4 else ""
        print(f"  {f:16s} {track_b_importance[f]:+.4f}{flag}")

    # --- Decision logic ---
    # A feature is dropped only if it is weak for BOTH tracks. If Track B shows
    # real importance for any of the "Track-A-weak" features, they are kept --
    # this is the whole point of running check (b) at all.
    track_b_weak_threshold = 0.01
    genuinely_weak = [f for f in WEAKEST_4 if track_b_importance[f] < track_b_weak_threshold]
    keep_for_track_b = [f for f in WEAKEST_4 if f not in genuinely_weak]

    if keep_for_track_b:
        decision = "all_13"
        reason = (
            f"Track A alone suggested dropping {WEAKEST_4}, but Track B's class-separation "
            f"importance shows {keep_for_track_b} still carry real signal for distinguishing "
            f"reference classes (periodicity/shape features matter for pulsators/EBs/rotational "
            f"variables even though they don't help catch these specific synthetic injections). "
            f"Per M4 (Track A results are never used alone as evidence for a Track B claim -- "
            f"symmetrically, they must not be used alone to justify removing a feature Track B "
            f"needs), the full 13-feature set is retained."
        )
    else:
        # both checks agree the 4 are dead weight -- confirm val performance doesn't drop
        best_a = max(track_a_results, key=track_a_results.get)
        if track_a_results["top_9_drop_weakest_4"] >= track_a_results["all_13"] - 0.01:
            decision = "top_9_drop_weakest_4"
            reason = (
                f"Both Track A (VAL AUC-PR) and Track B (class-separation importance) agree "
                f"{WEAKEST_4} carry negligible signal; dropping them costs no validation "
                f"performance ({track_a_results['top_9_drop_weakest_4']:.4f} vs "
                f"{track_a_results['all_13']:.4f} for all 13) and gives cleaner, more reliable "
                f"contrastive explanations (a feature that's mostly noise should not be eligible "
                f"to be named as 'the differing feature')."
            )
        else:
            decision = "all_13"
            reason = "Track B agreed the 4 were weak, but trimming cost validation AUC-PR -- kept all 13."

    frozen_features = CANDIDATES[decision]
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision_made_by": "Claude, on the user's explicit delegation -- decision logged here "
                             "in full so it can be audited/reversed if it turns out wrong",
        "track_a_val_auc_pr_by_candidate": track_a_results,
        "track_b_class_accuracy_by_candidate": track_b_results,
        "track_b_permutation_importance_all_13": track_b_importance,
        "frozen_feature_set_name": decision,
        "frozen_features": frozen_features,
        "n_features": len(frozen_features),
        "reasoning": reason,
        "m3_compliance_note": "Selection used TRAIN (fit) + VAL (compare candidates) only; "
                               "test-split labels were never loaded by this script.",
        "frozen": True,
    }
    out_path = OUT_DIR / "frozen_feature_set.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== DECISION: {decision} ({len(frozen_features)} features) ===")
    print(reason)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
