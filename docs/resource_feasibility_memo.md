# Step 2.5 — Resource & Feasibility Memo
**Project:** Beyond Detection: Morphological Fingerprinting and Contrastive 
Characterization of Novel Stellar Anomalies in TESS Data
**Date:** 2026-08-16
**Status:** Pre-registered estimate — to be recalibrated against empirical 
timing from Notebook 3, Cell 1's first real training run. This memo must be 
updated with actual figures once available; nothing below is measured data.
**Governs:** CP0 (Part E) — must be complete before Step 3 / Notebook 1 begins.

---

## 1. Hardware / Compute Environment

- **Primary:** Google Colab Free Tier, T4 GPU (16 GB VRAM).
- **Known constraints:** ~12 hr hard session cap, ~90 min idle disconnect, 
  and a rolling weekly GPU-hour quota that Colab enforces dynamically (not a 
  fixed published number) — treated as the binding constraint, not the 12 hr 
  session cap.
- **Fallback trigger:** if free-tier quota is exhausted before the ≥10-seed 
  headline runs (Models C, D, Baselines 1–3) complete, upgrade to Colab Pro 
  for the remainder of that notebook only, then drop back to free tier. 
  Decision point: re-evaluate after Notebook 3's first three seeds report 
  actual per-seed wall-clock time.
- **Storage backend:** Google Drive, mounted per-session (Notebook 1, Cell 1 
  onward) — required for cross-session persistence between the 8 separate 
  notebooks.
- **Checkpointing requirement (risk mitigation):** given the disconnect risk, 
  every seeded training run must checkpoint to Drive after completion, not 
  just at notebook end — a lost session mid-run should cost one seed, not 
  the whole notebook.

---

## 2. Expected Training Time Per Model Configuration (pre-registered estimate)

| Component | Est. time/seed (T4) | Basis |
|---|---|---|
| VAE (shared backbone, Models A/C/D) | ~8 min | small 1D architecture, ~500–700 curves |
| Hand-crafted 13-feature classifier (Model B, Baseline 2) | ~3 min | classical ML, not GPU-bound |
| CNN + Attention fingerprint (adds to VAE, Models C/D) | +15 min | larger architecture, attention overhead |
| Contrastive explanation engine overhead (Model D only) | +5 min | template-based, lightweight |
| Baseline 1 — VRAE + isolation forest | ~12 min | comparable to VAE + classical detector |
| Baseline 3 — Unsupervised RF + cosine similarity | ~5 min | RF is CPU-bound, fast |

Per Part C's seed schedule (R6): Model C's single ≥10-seed run is reused for 
both its ≥5-seed-equivalent and ≥10-seed headline comparisons — not double-counted.

| Config | Seeds | Time/seed | Subtotal |
|---|---|---|---|
| Model A — VAE only | 5 | 8 min | 40 min |
| Model B — hand-crafted fingerprint | 5 | 3 min | 15 min |
| Model C — VAE + CNN | 10 | 23 min | 230 min |
| Model D — VAE + CNN + contrastive (proposed) | 10 | 28 min | 280 min |
| Baseline 1 — Villar-style | 10 | 12 min | 120 min |
| Baseline 2 — Webb-style | 10 | 3 min | 30 min |
| Baseline 3 — CLARA-style | 10 | 5 min | 50 min |
| **Core training subtotal** | | | **~765 min ≈ 12.75 GPU-hr** |

Evaluation-only procedures that **reuse** already-trained models (not additive 
training time): leave-one-class-out (Model D reference-library rotation), 
novelty-formula selection, faithfulness check, injection-recovery curves, 
bootstrap CI, permutation test, error analysis — all CPU-bound post-processing, 
estimated **≤3 hr combined**.

Hyperparameter sensitivity sweep (Recommended, not Core): ~20–30 lightweight 
configs on validation only, ≤5 min each ≈ **~2 hr**, run only if Core budget 
allows.

**Total estimated Core compute: ~13 GPU-hr training + ~3 hr CPU evaluation.**

---

## 3. Expected Data-Download Time

- General pool: 500+ stars via bulk `lightkurve`/MAST download, Sectors 2–5.
- Reference library: ~5 classes (ExoFOP transits, SIMBAD UV Ceti flares, VSX 
  eclipsing binaries, pulsators, rotational variables) × 15–20 exemplars 
  ≈ 75–100 additional curves.
- Estimate: ~10 sec/curve average (MAST throughput, incl. retries) × ~600–700 
  curves ≈ **2–4 hours**, batched with resume capability to survive session 
  disconnects (Notebook 1, Cell 3).

---

## 4. Storage Requirements

| Item | Est. size |
|---|---|
| Raw curves (general pool + library, multi-sector stitched) | ~1.8 GB |
| Injected variants (multiple instances/star × 3 splits) | ~5–8 GB |
| Model checkpoints (7 configs × up to 10 seeds) | ~1 GB |
| Reproducibility artifacts (configs, logs, provenance manifests) | <0.1 GB |
| **Total** | **~10–15 GB** |

**Risk flag:** this is close to Google Drive's 15 GB free-tier ceiling. 
Decision needed before Notebook 1 runs at scale: either use a paid Drive tier, 
or plan to delete redundant injected-variant intermediates once Step 6's 
injection-recovery numbers are finalized and archived.

---

## 5. Total Estimated Experiment Count

- **60 seeded training runs** across the 7 Core model/baseline configurations 
  (Models A–D, Baselines 1–3).
- **~5 leave-one-class-out evaluation reruns** (one per reference class, 
  reusing Model D — headline classes at ≥10 rotations, exploratory classes at ≥5).
- **~20–30 hyperparameter sweep configs** (Recommended, validation-only).
- **4 cross-cutting statistical procedures** (bootstrap CI, permutation test, 
  faithfulness check, TESS-systematics check) — computed once each on frozen 
  test-split predictions, not seed-multiplied.

---

## 6. Target Timeline

Consistent with the under-1-month Core-experiment target:

| Week | Notebooks | Milestone |
|---|---|---|
| 1 | 0, 1, 2 | Gate passes; data acquired, vetted, split (CP1 passes) |
| 2 | 3, 4 | Models A–D + 3 baselines trained (CP2 passes) |
| 3 | 5, 6 | Reference-library distributions, ablation, Track A stats (CP3 passes) |
| 4 | 7 | Track B, robustness, error analysis, S2/S8 case study (CP4 passes) |
| Optional | 8 | SHAP + cross-mission generalization, if time permits |

This nests inside the project's broader ~13-week publication roadmap — Weeks 
1–4 above cover implementation; the remainder is writing, revision, and review.

---

## 7. Core / Revision Tag — Every Part C Row

*(Carried forward from Part C's own Set column, v3.1 — this memo is the 
formal CP0 sign-off confirming that tagging.)*

| Experiment | Tag |
|---|---|
| Model A — VAE only | **Core** |
| Model B — VAE + hand-crafted fingerprint | **Core** |
| Model C — VAE + CNN fingerprint | **Core** |
| Model D — VAE + CNN + contrastive (proposed) | **Core** |
| External Baseline 1 — Villar-style | **Core** |
| External Baseline 2 — Webb-style | **Core** |
| External Baseline 3 — CLARA-style | **Core** |
| Novelty-formula selection | **Validation-only** (internal, not FDR-bearing) |
| Leave-one-class-out validation | **Core** |
| Injection-recovery curve | **Core** |
| Hyperparameter sensitivity sweep | **Recommended** |
| TESS-systematics check | **Core** |
| Faithfulness check | **Core** |
| Bootstrap CI on novelty scores | **Core** |
| Permutation test on anomaly rate | **Core** |
| Error analysis (Step 10.5) | **Core** |
| Cross-mission generalization | **Revision / Optional** |
| SHAP comparison on S8 | **Revision / Optional** |

---

## CP0 Checklist

- [x] Hardware decision made (Colab free tier, T4, with Pro fallback trigger)
- [x] Time estimates cover both ≥5-seed and ≥10-seed headline runs
- [x] Storage requirements estimated, risk flagged (near 15 GB Drive limit)
- [x] Total experiment count estimated
- [x] Target timeline set (4-week Core implementation window)
- [x] Every Part C row carries a Core/Recommended/Revision tag

**CP0 status: PASS**, pending your review/sign-off of the assumptions above.
