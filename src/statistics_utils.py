"""
Generic, reusable statistical-inference primitives per R7 / the pre-registration
memo's Section 4, ready to apply to real per-star predictions once models exist.
Nothing here is wired to a specific model yet -- these are the shared building
blocks Notebook 6 (ablation) and Notebook 7 (Track B, robustness) will call.

Implements, exactly as pre-registered:
  - Star-level paired bootstrap (primary inference). Resamples whole STARS as
    blocks, never individual instances/cadences independently -- required
    because a star can contribute multiple injected instances (Step 6).
  - Degenerate-resample discard rule: a bootstrap resample with too few
    positive (injected) instances to compute a stable AUC-PR is discarded, and
    the discard rate is reported alongside the CI (Step 6/R7).
  - A pre-specified permutation test for a paired star-level metric difference.
  - Benjamini-Hochberg FDR correction, applied within named, closed families
    (the exhaustive Family 1/2/3 list from the memo/roadmap) -- correction
    must never be applied across families, only within one.

None of this performs any test-split-touching by itself; it operates on
whatever per-star arrays it is handed. Callers are responsible for the R4/R9
rule that test is touched exactly once, for final reported numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BootstrapResult:
    point_estimate: float
    ci_lower: float
    ci_upper: float
    n_resamples_used: int
    n_resamples_discarded: int
    discard_rate: float
    bootstrap_distribution: np.ndarray = field(repr=False)


def star_level_bootstrap(
    star_ids: np.ndarray,
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_resamples: int = 1000,
    min_positives: int = 5,
    seed: int = 42,
    alpha: float = 0.05,
) -> BootstrapResult:
    """
    Star-level (block) bootstrap for a metric computed over instance-level
    (y_true, y_score) pairs, where `star_ids` gives each instance's physical
    star (R7: the star, not the cadence/instance, is the independent
    observational unit; a resampled star's instances all travel together).

    `metric_fn(y_true_subset, y_score_subset) -> float`, e.g. average_precision_score.
    A resample is discarded (not zero-filled) if it has fewer than
    `min_positives` positive instances, since AUC-PR is unstable/undefined-ish
    at very low positive counts -- the discard rate itself is part of the
    reported result (Step 6's degenerate-resample safeguard).
    """
    rng = np.random.RandomState(seed)
    unique_stars = np.unique(star_ids)
    star_to_idx = {s: np.where(star_ids == s)[0] for s in unique_stars}

    point_estimate = metric_fn(y_true, y_score)

    boot_values = []
    n_discarded = 0
    for _ in range(n_resamples):
        sampled_stars = rng.choice(unique_stars, size=len(unique_stars), replace=True)
        idx = np.concatenate([star_to_idx[s] for s in sampled_stars])
        y_true_bs, y_score_bs = y_true[idx], y_score[idx]
        if y_true_bs.sum() < min_positives:
            n_discarded += 1
            continue
        boot_values.append(metric_fn(y_true_bs, y_score_bs))

    boot_values = np.array(boot_values)
    lo, hi = np.percentile(boot_values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootstrapResult(
        point_estimate=float(point_estimate),
        ci_lower=float(lo),
        ci_upper=float(hi),
        n_resamples_used=len(boot_values),
        n_resamples_discarded=n_discarded,
        discard_rate=n_discarded / n_resamples,
        bootstrap_distribution=boot_values,
    )


@dataclass
class PairedComparisonResult:
    observed_diff: float
    ci_lower: float
    ci_upper: float
    permutation_p_value: float
    effect_size_rank_biserial: float
    n_resamples_used: int
    n_resamples_discarded: int


def star_level_paired_comparison(
    star_ids: np.ndarray,
    metric_a_per_star: dict,
    metric_b_per_star: dict,
    n_resamples: int = 1000,
    n_permutations: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> PairedComparisonResult:
    """
    Primary inferential basis for a model-vs-model (or model-vs-baseline)
    comparison (R7 item 3): paired difference in TEST-star-level predictions,
    with a 95% CI via paired star-level bootstrap and a pre-specified
    permutation procedure, plus an effect size (rank-biserial correlation).

    `metric_a_per_star` / `metric_b_per_star`: {star_id: per-star metric value},
    e.g. each star's individual AUC-PR or reconstruction error, ALREADY
    averaged across the full seed set for that configuration (R7's pinned
    seed-then-bootstrap order -- do that averaging before calling this).
    """
    stars = np.array(sorted(set(metric_a_per_star) & set(metric_b_per_star)))
    if len(stars) == 0:
        raise ValueError("No overlapping stars between the two metric dicts.")
    a = np.array([metric_a_per_star[s] for s in stars])
    b = np.array([metric_b_per_star[s] for s in stars])
    diffs = a - b

    observed_diff = float(diffs.mean())

    rng = np.random.RandomState(seed)
    boot_diffs = []
    n_discarded = 0
    for _ in range(n_resamples):
        idx = rng.choice(len(stars), size=len(stars), replace=True)
        boot_diffs.append(diffs[idx].mean())
    boot_diffs = np.array(boot_diffs)
    lo, hi = np.percentile(boot_diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    # Pre-specified permutation test: randomly flip the sign of each star's
    # paired difference (equivalent to permuting which model "wins" per star
    # under the null of no systematic difference), two-sided p-value.
    perm_stats = np.empty(n_permutations)
    for i in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(diffs))
        perm_stats[i] = (diffs * signs).mean()
    p_value = float(np.mean(np.abs(perm_stats) >= np.abs(observed_diff)))

    n_pos = int(np.sum(diffs > 0))
    n_neg = int(np.sum(diffs < 0))
    rank_biserial = (n_pos - n_neg) / len(diffs) if len(diffs) > 0 else float("nan")

    return PairedComparisonResult(
        observed_diff=observed_diff,
        ci_lower=float(lo),
        ci_upper=float(hi),
        permutation_p_value=p_value,
        effect_size_rank_biserial=float(rank_biserial),
        n_resamples_used=n_resamples,
        n_resamples_discarded=n_discarded,
    )


def benjamini_hochberg(p_values: dict, alpha: float = 0.05) -> dict:
    """
    BH FDR correction, applied within ONE pre-registered family at a time.
    `p_values`: {test_name: raw_p_value} for every test in that family, all
    assigned to this family BEFORE any test-split result is viewed (R7).
    Returns {test_name: {"p_raw", "p_adjusted", "significant"}}.
    """
    names = list(p_values.keys())
    raw = np.array([p_values[n] for n in names])
    order = np.argsort(raw)
    ranked = raw[order]
    m = len(ranked)
    adjusted = ranked * m / (np.arange(m) + 1)
    # enforce monotonicity (standard BH step-up correction)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    result = {}
    for rank_pos, orig_pos in enumerate(order):
        name = names[orig_pos]
        result[name] = {
            "p_raw": float(raw[orig_pos]),
            "p_adjusted": float(adjusted[rank_pos]),
            "significant": bool(adjusted[rank_pos] < alpha),
        }
    return result


# --- Pre-registered FDR family membership (Part C / R7, closed & exhaustive) ---
# Populate each family's dict with {test_name: p_value} as real results come in,
# then call benjamini_hochberg(...) separately per family -- never pool across
# families, and every test added later must be assigned to one of these three
# (or a documented new Family 4) before its result is viewed.
FDR_FAMILY_1_TRACK_A_CONFIRMATORY = [
    "Model_A_vs_B", "Model_B_vs_C", "Model_C_vs_D",
    "Model_C_vs_Baseline1", "Model_B_vs_Baseline2", "Model_D_vs_Baseline3",
    "anomaly_rate_permutation_test",
]
FDR_FAMILY_2_TRACK_B_CONFIRMATORY_TEMPLATE = (
    "leave_one_class_out_recovery_rate__{cls}", "known_class_false_novel_rate__{cls}",
)
FDR_FAMILY_3_EXPLORATORY_ROBUSTNESS = [
    "faithfulness_check_delta_S", "tess_systematics_covariate_adjusted",
]
EXCLUDED_FROM_FDR = ["hyperparameter_sensitivity_sweep", "novelty_formula_selection"]
