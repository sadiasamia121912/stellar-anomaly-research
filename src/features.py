"""
13-dimensional hand-crafted morphological fingerprint.

Ported and documented from the original pilot (Master.md, Cell 5) for reuse in
Notebook 3 (roadmap Cells 5-7: candidate extractor -> train-split-only importance
ranking -> frozen feature set) and in External Baseline 2 (Webb-style hand-crafted
features). Kept as one plain-numpy module (no notebook-only globals) so it can be
imported identically from a local script, a baseline implementation, and a Colab
notebook cell.

Each feature below carries a one-line astrophysical rationale, per Step 8's
requirement that every retained hand-crafted feature be justified, not just listed.
Whether a given feature survives training-split importance ranking (and thus
belongs in the FROZEN set actually used in Models B/D) is decided later, in
Notebook 3, using train-split injection labels only (M3) -- this module only
defines the full 13-candidate extractor.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.stats import entropy, kurtosis, skew

FEATURE_NAMES = [
    "Symmetry", "N_Peaks", "N_Troughs", "Rise_Fall_Ratio",
    "Dom_Freq", "Spec_Entropy", "HF_Power",
    "Dim_Frac", "Bright_Frac", "Flux_Std",
    "Skewness", "Kurtosis", "RMS",
]

FEATURE_RATIONALE = {
    "Symmetry": "Transits/eclipses are near-symmetric about mid-event; flares and "
                "asymmetric pulsators are not -- captures ingress/egress vs. fast-rise "
                "slow-decay morphology.",
    "N_Peaks": "Counts local maxima -- separates single-event (transit/flare) curves "
               "from multi-cycle periodic variables (pulsators, rotational variables).",
    "N_Troughs": "Counts local minima -- complements N_Peaks; e.g. eclipsing binaries "
                 "can show two unequal-depth troughs (primary/secondary eclipse) per cycle.",
    "Rise_Fall_Ratio": "Ratio of pre-peak rise slope to post-peak fall slope -- the "
                       "canonical flare signature (fast rise, slow exponential decay) "
                       "has this far from 1; symmetric events are near 1.",
    "Dom_Freq": "Dominant periodogram frequency -- the direct period estimator for "
                "periodic phenomena (pulsators, rotational variables, EBs); near-zero "
                "for single, aperiodic events (an isolated transit or flare).",
    "Spec_Entropy": "Shannon entropy of the normalized power spectrum -- low for a "
                    "narrowband periodic signal concentrated in one frequency, high "
                    "for broadband/aperiodic or noise-dominated curves.",
    "HF_Power": "Fraction of spectral power in the top 7/8 of the Nyquist band -- "
                "flags short-timescale structure (flares, fast transit ingress/egress, "
                "short-period pulsators) that low-frequency features miss.",
    "Dim_Frac": "Fraction of points below a low-flux threshold -- measures how much "
                "of the curve spends time in a dimmed state (transit/eclipse duration "
                "fraction, or a long-duration dimming event).",
    "Bright_Frac": "Fraction of points above a high-flux threshold -- the complementary "
                   "measure; large for flare spikes, near-baseline curves, or the "
                   "out-of-eclipse phase of an EB.",
    "Flux_Std": "Overall flux standard deviation -- the coarsest variability amplitude "
               "measure; distinguishes photometrically quiet stars from any active/"
               "variable/anomalous ones regardless of morphology.",
    "Skewness": "Third moment of the flux distribution -- transits/eclipses (brief "
               "dips below a flat baseline) skew negative; flares (brief spikes above "
               "a flat baseline) skew positive.",
    "Kurtosis": "Fourth moment (excess kurtosis) of the flux distribution -- high for "
               "curves dominated by rare, brief, extreme excursions (an isolated flare "
               "or transit against a flat baseline) vs. smoothly-varying pulsators.",
    "RMS": "Root-mean-square flux -- a scale-sensitive companion to Flux_Std, sensitive "
          "to the DC/baseline level as well as spread (useful once curves are z-scored "
          "rather than min-max normalized to [0, 1]).",
}


def compute_symmetry(curve: np.ndarray) -> float:
    mid = len(curve) // 2
    left = curve[:mid]
    right = curve[mid:mid + len(left)][::-1]
    return 1.0 - float(np.mean(np.abs(left - right)))


def count_peaks(curve: np.ndarray, prominence: float = 0.05) -> tuple[int, int]:
    peaks, _ = find_peaks(curve, prominence=prominence, distance=20)
    troughs, _ = find_peaks(-curve, prominence=prominence, distance=20)
    return len(peaks), len(troughs)


def rise_fall_ratio(curve: np.ndarray) -> float:
    peak_idx = int(np.argmax(curve))
    if peak_idx == 0 or peak_idx == len(curve) - 1:
        return 1.0
    rise = (curve[peak_idx] - curve[0]) / (peak_idx + 1e-8)
    fall = (curve[peak_idx] - curve[-1]) / (len(curve) - peak_idx + 1e-8)
    return float(rise / (fall + 1e-8))


def compute_fft_features(curve: np.ndarray) -> tuple[float, float, float]:
    n = len(curve)
    fft_vals = np.abs(fft(curve - np.mean(curve)))[: n // 2]
    freqs = fftfreq(n)[: n // 2]
    power = fft_vals ** 2
    total = np.sum(power) + 1e-10
    power_norm = power / total
    dom_idx = int(np.argmax(power[1:])) + 1 if n > 2 else 0
    return float(freqs[dom_idx]), float(entropy(power_norm[1:] + 1e-10)), float(np.sum(power[n // 8:]) / total)


def compute_duration_features(curve: np.ndarray, threshold: float = 0.3) -> tuple[float, float]:
    return (
        float(np.sum(curve < threshold) / len(curve)),
        float(np.sum(curve > (1 - threshold)) / len(curve)),
    )


def compute_variability(curve: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(np.std(curve)),
        float(skew(curve)),
        float(kurtosis(curve)),
        float(np.sqrt(np.mean(curve ** 2))),
    )


def morphological_fingerprint(curve: np.ndarray) -> np.ndarray:
    """
    Compute the full 13-candidate hand-crafted fingerprint for one light curve.

    `curve` is expected to be a 1D array on a bounded, roughly [0, 1]-normalized
    scale (matching the pilot's min-max normalization) for Dim_Frac/Bright_Frac's
    fixed threshold=0.3 to be meaningful; if the pipeline's normalization changes
    (e.g. to z-scored flux, per Notebook 2/3's GLOBAL_NORM_MEAN/STD), the duration
    thresholds must be re-derived, not reused as-is -- flagged here rather than
    silently producing meaningless Dim_Frac/Bright_Frac values on the wrong scale.
    """
    s = compute_symmetry(curve)
    n_peaks, n_troughs = count_peaks(curve)
    rfr = rise_fall_ratio(curve)
    dom_freq, spec_entropy, hf_power = compute_fft_features(curve)
    dim_frac, bright_frac = compute_duration_features(curve)
    flux_std, sk, ku, rms = compute_variability(curve)
    return np.array([
        s, n_peaks, n_troughs, rfr, dom_freq, spec_entropy, hf_power,
        dim_frac, bright_frac, flux_std, sk, ku, rms,
    ])
