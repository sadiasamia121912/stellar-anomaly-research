"""
Parallel Track — S2/S8 Verification (Step 5 / notebook_roadmap.md's "Parallel Track").

Thin, backward-compatible entry point. The actual implementation was
generalized into candidate_verification.py once S2/S8's own results (both are
already-known planet hosts) made it clear the same check would need to run
again on whatever new candidates the full pipeline's novelty scores surface
later (see candidate_verification.py's module docstring) -- kept this file so
anything referencing "run s2_s8_verification.py" per the roadmap still works
unchanged.

Usage: python src/s2_s8_verification.py
"""

from pathlib import Path

from candidate_verification import DEFAULT_CANDIDATES, verify_candidates

ORIGINAL_OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "s2_s8_verification"

if __name__ == "__main__":
    verify_candidates(DEFAULT_CANDIDATES, out_name="s2_s8_verification_report.json", out_dir=ORIGINAL_OUT_DIR)
