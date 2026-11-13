"""Cross-file integration check: confirm the whole chain is mutually consistent.

Run AFTER the pipeline (`python -m stgreen.pipeline`). It loads every produced
table and asserts that keys, schemas and joins line up across modules -- i.e. the
output of each stage is a valid input to the next. Prints a PASS/FAIL report and
exits non-zero if any check fails.

    python scripts/integration_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stgreen import config as cfg
from stgreen.config import CANONICAL_COLUMNS  # noqa: E402  (after sys.path tweak)
from stgreen.metrics.structure import STRUCTURE_PANEL_TEMPLATE
from stgreen.metrics.ecoservices import ECOSERVICES_PANEL_TEMPLATE
from stgreen.metrics.gvi import GVI_PANEL_TEMPLATE
from stgreen.data.renewal import RENEWAL_PANEL_TEMPLATE
from stgreen.change.change_metrics import ANALYSIS_PANEL_TEMPLATE, CHANGE_TABLE_TEMPLATE

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(cond), detail))


def _keys(df: pd.DataFrame) -> set:
    return set(zip(df["unit_id"], df["source_year"]))


def main() -> int:
    P = cfg.PROCESSED_DIR

    # --- 1. Canonical panel -------------------------------------------------
    panel = pd.read_parquet(cfg.CANONICAL_PANEL_PARQUET)
    check("canonical: non-empty", len(panel) > 0, f"{len(panel):,} rows")
    check("canonical: schema matches CANONICAL_COLUMNS", list(panel.columns) == list(CANONICAL_COLUMNS))
    check("canonical: (source_year,orig_id) unique", not panel.duplicated(["source_year", "orig_id"]).any())
    check("canonical: epochs are the census years", set(panel["source_year"]) == set(cfg.CENSUS_YEARS))

    # --- 2. Per-unit structure / eco / gvi ---------------------------------
    for u in ("nta", "borough"):
        s = pd.read_parquet(P / STRUCTURE_PANEL_TEMPLATE.format(unit=u))
        e = pd.read_parquet(P / ECOSERVICES_PANEL_TEMPLATE.format(unit=u))
        g = pd.read_parquet(P / GVI_PANEL_TEMPLATE.format(unit=u))
        sk = _keys(s)
        check(f"[{u}] structure keys unique", not s.duplicated(["unit_id", "source_year"]).any())
        check(f"[{u}] eco keys subset of structure", _keys(e).issubset(sk),
              f"{len(_keys(e) - sk)} orphan eco keys")
        check(f"[{u}] gvi schema aligned", {"unit_type", "unit_id", "source_year", "gvi_mean"}.issubset(g.columns))
        check(f"[{u}] gvi units subset of structure", set(g["unit_id"]).issubset(set(s["unit_id"])))
        check(f"[{u}] gvi in [0,100]", g["gvi_mean"].dropna().between(0, 100).all())

    # --- 3. Renewal treatment (nta) ----------------------------------------
    s_nta = pd.read_parquet(P / STRUCTURE_PANEL_TEMPLATE.format(unit="nta"))
    ren = pd.read_parquet(P / RENEWAL_PANEL_TEMPLATE.format(unit="nta"))
    check("renewal: units subset of structure", set(ren["unit_id"]).issubset(set(s_nta["unit_id"])))
    check("renewal: treated<=>year present",
          bool(ren.loc[ren["treated"].eq(True), "first_treatment_year"].notna().all()
               and ren.loc[ren["treated"].eq(False), "first_treatment_year"].isna().all()))

    # --- 4. Analysis panel = structure x eco x renewal ---------------------
    ap = pd.read_parquet(P / ANALYSIS_PANEL_TEMPLATE.format(unit="nta"))
    check("analysis: row count equals structure(nta)", len(ap) == len(s_nta),
          f"analysis {len(ap)} vs structure {len(s_nta)}")
    check("analysis: keys unique", not ap.duplicated(["unit_id", "source_year"]).any())
    check("analysis: carries eco + treatment columns",
          {"carbon_storage_Mg", "treated", "treatment_year", "post_treatment"}.issubset(ap.columns))
    check("analysis: post_treatment implies treated",
          not (ap["post_treatment"].eq(True) & ap["treated"].eq(False)).any())
    # join integrity: structural value survives the merge unchanged
    merged = ap.merge(s_nta[["unit_id", "source_year", "n_alive"]], on=["unit_id", "source_year"],
                      suffixes=("", "_s"))
    check("analysis: n_alive consistent with structure", (merged["n_alive"] == merged["n_alive_s"]).all())

    # --- 5. Change table ----------------------------------------------------
    ct = pd.read_parquet(P / CHANGE_TABLE_TEMPLATE.format(unit="nta"))
    check("change: periods valid", set(ct["period"]).issubset({"1995_2005", "2005_2015", "1995_2015"}))
    check("change: units subset of analysis", set(ct["unit_id"]).issubset(set(ap["unit_id"])))
    check("change: treatment_in_period implies treated",
          not (ct["treatment_in_period"].eq(True) & ct["treated"].eq(False)).any())
    if "turnover" in ct.columns:
        check("change: turnover in [0,1]", ct["turnover"].dropna().between(0, 1).all())

    # --- 6. DID results -----------------------------------------------------
    did = pd.read_csv(cfg.TABLES_DIR / "did_suite_nta.csv")
    check("did: suite non-empty", len(did) > 0, f"{len(did)} outcomes")
    check("did: n_obs positive and consistent", (did["n_obs"] > 0).all() and did["n_obs"].nunique() == 1)

    # --- 7. Figures ---------------------------------------------------------
    figs = [f"fig{i}" for i in range(1, 8)]
    missing = [f for f in figs if not list(cfg.FIGURES_DIR.glob(f"{f}_*.pdf"))]
    check("figures: all 7 PDFs present", not missing, f"missing: {missing}" if missing else "7/7")
    missing_png = [f for f in figs if not list(cfg.FIGURES_DIR.glob(f"{f}_*.png"))]
    check("figures: all 7 PNGs present", not missing_png, f"missing: {missing_png}" if missing_png else "7/7")

    # --- Report -------------------------------------------------------------
    width = max(len(n) for n, _, _ in CHECKS)
    n_fail = 0
    print("\nCross-file integration check")
    print("=" * (width + 16))
    for name, ok, detail in CHECKS:
        n_fail += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    print("=" * (width + 16))
    print(f"{len(CHECKS) - n_fail}/{len(CHECKS)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
