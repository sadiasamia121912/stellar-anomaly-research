# Step 5 Resolution: S2/S8 Identity, and the Plan Going Forward

**Date:** 2026-09-06
**Status:** Identity question closed. Feeds Step 12 (discovery-vs-tool framing), formally decided
after Track B (Step 10) per the roadmap's own sequencing -- this note records the interim
conclusion and the concrete plan, not a final Step 12 decision.

## What was found

`results/s2_s8_verification/s2_s8_verification_report.json` (SIMBAD cross-match + alias scan,
`src/candidate_verification.py`):

- **S2 (TIC 261136679) = π Mensae.** SIMBAD type `PM*` (planet-hosting star), alias **TOI-144**.
  A naked-eye star with a confirmed transiting planet (π Men c -- one of TESS's first-ever
  discoveries, Sector 1) plus an additional known RV planet (π Men b).
- **S8 (TIC 149603524) = WASP-62.** Alias **TOI-102**. A confirmed transiting hot-Jupiter host
  (WASP-62b), independently discovered by the ground-based WASP survey and later observed by TESS.

Both stars the original 8-star pilot flagged as anomalous are, with very high confidence, showing
the known transit signal of an already-published planet -- not a genuinely uncatalogued
phenomenon. This is not a negative result for the project; it is exactly the kind of outcome
Stage 8 of `star_PROJECT_ANALYSIS_UPDATED.md` already anticipated and pre-approved as a
legitimate, publishable outcome (the "tool paper" alternative).

## What this changes

1. **S2 and S8 stop being discovery candidates.** They remain extremely useful as **validation
   case studies**: does the full pipeline (once trained) correctly flag a real, confirmed
   transiting-planet signal as anomalous, and does the Contrastive Explanation Engine produce an
   explanation that is astrophysically sensible for a *known* case (e.g., attributing the anomaly
   to the features a ~1-2% transit would actually be expected to perturb)? That is a real,
   citable validation result, independent of any discovery claim.
2. **Step 12's framing leans toward "tool paper"** as the working default: "the system correctly
   detects and automatically characterizes known anomalies, enabling prioritized follow-up" --
   not a "new class" claim. This is provisional, not locked; Track B's leave-one-class-out results
   (Step 10) are the actual pre-registered evidence for any novelty language (M4), and the formal
   Step 12 decision waits for those, per the roadmap.
3. **The discovery angle is not dead -- it just doesn't rest on S2/S8 anymore.** The project scaled
   from an 8-star pilot to a 550-star general pool specifically so the pipeline could surface
   candidates beyond the two the pilot happened to flag. That surfacing hasn't happened yet because
   Models A-D haven't been trained (Part 2, GPU-blocked).

## The concrete plan

1. `src/s2_s8_verification.py`'s core logic was generalized into
   `src/candidate_verification.py` (`verify_candidates(candidates: dict[label, tic_id])`), so
   checking a new star is a one-line call, not a rewrite.
2. Once Model D is trained (Notebook 7) and produces a novelty score for every star in the
   **full 550-star general pool** (not just S2/S8), take the top-N ranked by novelty score
   (N=10 is a reasonable starting point -- adjust based on how the score distribution looks) and
   run `candidate_verification.py` on each.
3. Any candidate that comes back **without** a known-planet-host alias (and, ideally, without any
   specific SIMBAD variable-star classification either) is a genuine candidate for the
   discovery-adjacent framing ("statistically robust, previously uncatalogued photometric
   behavior warranting spectroscopic follow-up" -- the ceiling language already fixed in
   `docs/pre_registration_memo.md` Section 1.3). Cross-sector persistence (Step 5's original
   check, now reusable via the same script) still applies to whichever candidates pass this filter.
4. This does not change any pre-registered rule: R9 still applies (no candidate's identity is
   ever used to tune a hyperparameter or threshold), and R1b's S2/S8-specific test-set forcing is
   untouched regardless of this finding.
