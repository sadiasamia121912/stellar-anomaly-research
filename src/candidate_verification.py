"""
General-purpose anomaly-candidate verification: SIMBAD cross-match + known-
planet-host alias check + cross-sector data-availability/variability stats,
for ANY TIC ID -- not just S2/S8.

Originally built as `s2_s8_verification.py` for Step 5's specific pair. Both
turned out to be already-known planet hosts (S2 = pi Mensae/TOI-144, S8 =
WASP-62/TOI-102) -- see results/s2_s8_verification/s2_s8_verification_report.json.
That closes the door on S2/S8 specifically as discovery candidates, but not on
the project's discovery angle in general: once Notebook 3-7's full pipeline
runs on the real 550-star general pool, it will produce its own novelty-score
ranking across ALL of them, not just the original 8-star pilot's two flagged
stars. Whatever stars rank highest there are the real discovery candidates,
and they need exactly this same check before anyone gets excited about them.
Generalized here so that check is a one-line rerun, not a rewrite, when that
day comes (see results/next_candidates/ for the plan/placeholder).

Usage:
  python src/candidate_verification.py                     # runs the default S2/S8 pair
  python src/candidate_verification.py --tic 123456789 my_label 987654321 other_label
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

SECTORS = [1, 2, 3, 4, 5]
DEFAULT_CANDIDATES = {"S2": "261136679", "S8": "149603524"}
OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "candidate_verification"


def simbad_lookup(tic_id: str) -> dict:
    """Cross-match a TIC ID against SIMBAD via its TESS Input Catalog identifier."""
    from astroquery.simbad import Simbad

    simbad = Simbad()
    # Default query_object only returns astrometry -- explicitly request object
    # type, all known catalog identifiers/aliases, and spectral type, or this
    # check silently reports "no classification" for stars that in fact have one.
    simbad.add_votable_fields("otype", "ids", "sp")

    result = {"tic_id": tic_id, "simbad_queried_at": datetime.now(timezone.utc).isoformat()}
    try:
        table = simbad.query_object(f"TIC {tic_id}")
        if table is None or len(table) == 0:
            result["simbad_status"] = "NOT_FOUND"
            result["simbad_main_id"] = None
            result["simbad_otype"] = None
        else:
            row = table[0]
            colnames = table.colnames

            def _get(*candidates):
                for cand in candidates:
                    col = next((c for c in colnames if c.upper() == cand.upper()), None)
                    if col is not None:
                        val = row[col]
                        return None if val is None or str(val).strip() in ("", "--") else str(val)
                return None

            result["simbad_status"] = "FOUND"
            result["simbad_main_id"] = _get("main_id")
            result["simbad_otype"] = _get("otype")
            result["simbad_spectral_type"] = _get("sp_type", "sp")
            ids_raw = _get("ids")
            result["simbad_all_ids"] = ids_raw.split("|") if ids_raw else None
            result["simbad_raw_columns"] = colnames
    except Exception as exc:  # network/service errors are reportable, not fatal
        result["simbad_status"] = "QUERY_ERROR"
        result["simbad_error"] = str(exc)
    return result


def cross_sector_availability(tic_id: str, sectors: list[int] = SECTORS) -> dict:
    """
    Search (not necessarily download -- downloads are large and this is a
    lightweight availability + basic-stats check) each sector's light curve
    for this TIC ID, and where available, download and compute simple
    per-sector flux statistics as a first-pass persistence signal.
    """
    import lightkurve as lk

    per_sector = {}
    for sector in sectors:
        entry = {"sector": sector}
        try:
            search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", sector=sector, author="SPOC")
            entry["n_products_found"] = len(search)
            if len(search) == 0:
                entry["status"] = "NOT_OBSERVED_OR_NOT_SPOC"
                per_sector[str(sector)] = entry
                continue
            lc = search[0].download()
            lc = lc.remove_nans()
            try:
                lc = lc.remove_outliers(sigma=5)
            except Exception:
                pass
            flux = lc.flux.value
            time = lc.time.value
            if len(flux) < 10:
                entry["status"] = "DOWNLOADED_TOO_SHORT"
                per_sector[str(sector)] = entry
                continue
            entry["status"] = "OK"
            entry["n_points"] = int(len(flux))
            entry["time_span_days"] = float(time.max() - time.min())
            entry["flux_mean"] = float(np.mean(flux))
            entry["flux_std"] = float(np.std(flux))
            entry["flux_rel_std"] = float(np.std(flux) / (np.mean(flux) + 1e-12))
            entry["flux_min"] = float(np.min(flux))
            entry["flux_max"] = float(np.max(flux))
            entry["flux_ptp_rel"] = float((np.max(flux) - np.min(flux)) / (np.mean(flux) + 1e-12))
        except Exception as exc:
            entry["status"] = "ERROR"
            entry["error"] = str(exc)
        per_sector[str(sector)] = entry
    return per_sector


def interpret_known_status(simbad: dict) -> dict:
    """
    Flag whether the SIMBAD alias list contains a known-planet-host naming
    pattern (TOI/WASP/KOI/Kepler/K2/HAT-P/etc.). SIMBAD's generic `otype`
    field (e.g. "*", plain "star") can lag behind a star's actual planet-host
    status -- the alias list is the more reliable signal and must be checked
    explicitly, not inferred from otype alone (this is exactly the gap that
    otype="*" for S8 would otherwise have hidden).
    """
    all_ids = simbad.get("simbad_all_ids") or []
    known_host_prefixes = ("TOI-", "WASP-", "KOI-", "KEPLER-", "K2-", "HAT-P-", "HATS-", "QATAR-", "XO-")
    matches = [i for i in all_ids if i.upper().startswith(known_host_prefixes)]
    return {
        "known_planet_host_aliases": matches,
        "appears_to_be_known_planet_host": bool(matches),
        "interpretation": (
            f"Alias list includes {matches} -- this star already has a confirmed/"
            f"candidate planet designation independent of this project. Its anomaly "
            f"flag most likely corresponds to a known transiting-planet signal, not "
            f"a novel phenomenon. Not a discovery candidate; usable only as a "
            f"validation/tool-paper case study."
            if matches else
            "No known-planet-host alias pattern found in SIMBAD's ID list. "
            "Does not by itself confirm novelty -- absence of a designation is "
            "not evidence of absence -- but this star is a genuine candidate "
            "for further novelty investigation, pending the trained pipeline's "
            "own novelty score and a literature check beyond just this alias scan."
        ),
    }


def build_report(label: str, tic_id: str, sectors: list[int] = SECTORS) -> dict:
    print(f"\n=== {label} (TIC {tic_id}) ===")
    print("Querying SIMBAD...")
    simbad = simbad_lookup(tic_id)
    print(f"  SIMBAD status: {simbad['simbad_status']}"
          + (f" ({simbad.get('simbad_main_id')}, otype={simbad.get('simbad_otype')})"
             if simbad['simbad_status'] == "FOUND" else ""))
    known_status = interpret_known_status(simbad)
    if known_status["appears_to_be_known_planet_host"]:
        print(f"  Known planet-host alias(es) found: {known_status['known_planet_host_aliases']}")

    print(f"Checking cross-sector availability across sectors {sectors}...")
    sector_data = cross_sector_availability(tic_id, sectors)
    for sec, entry in sector_data.items():
        if entry.get("status") == "OK":
            print(f"  Sector {sec}: OK, n={entry['n_points']}, "
                  f"rel_std={entry['flux_rel_std']:.4%}, span={entry['time_span_days']:.1f}d")
        else:
            print(f"  Sector {sec}: {entry.get('status')}")

    n_ok = sum(1 for e in sector_data.values() if e.get("status") == "OK")
    return {
        "label": label,
        "tic_id": tic_id,
        "simbad": simbad,
        "known_status": known_status,
        "per_sector": sector_data,
        "n_sectors_with_data": n_ok,
        "preliminary_persistence_note": (
            f"{n_ok}/{len(sectors)} sectors yielded usable data. This is a "
            "lightweight, model-free availability/variability-statistics check "
            "only -- it does NOT run the trained VAE/Model D pipeline. The formal, "
            "pipeline-based cross-sector anomaly-persistence check belongs in "
            "Notebook 7, Cell 8, once Model D is trained, and will use this same "
            "TIC ID and sector list."
        ),
        "r9_compliance_note": (
            "This result has not been, and will not be, used to select any "
            "hyperparameter, threshold, or feature."
        ),
    }


def verify_candidates(candidates: dict[str, str], sectors: list[int] = SECTORS,
                       out_name: str | None = None, out_dir: Path | None = None) -> dict:
    out_dir = out_dir or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sectors_checked": sectors,
        **{label: build_report(label, tic_id, sectors) for label, tic_id in candidates.items()},
    }
    out_name = out_name or ("_".join(candidates.keys()).lower() + "_verification_report.json")
    out_path = out_dir / out_name
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved verification report to {out_path}")
    return report


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--tic":
        rest = args[1:]
        if len(rest) % 2 != 0:
            raise SystemExit("Usage: --tic <TIC_ID> <label> [<TIC_ID> <label> ...]")
        candidates = {rest[i + 1]: rest[i] for i in range(0, len(rest), 2)}
    else:
        candidates = DEFAULT_CANDIDATES
    verify_candidates(candidates)


if __name__ == "__main__":
    main()
