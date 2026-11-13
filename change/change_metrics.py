"""Integrate the pillars and quantify structural change across census epochs.

This batch is where the study's title verb -- *tracking* the transformation --
becomes a table. It performs two jobs:

1. :func:`build_analysis_panel` merges the structural indicators (Batch 2), the
   carbon ecosystem services (Batch 3) and the urban-renewal treatment table
   (Batch 4) into one long **unit x epoch** panel, and adds the treatment-timing
   flags (``treated``, ``post_treatment``, ``years_since_treatment``) that the
   difference-in-differences analysis needs.

2. :func:`build_change_table` differences that panel between consecutive epochs
   (and across the full span) to produce per-unit change: absolute deltas and
   growth rates for abundance, basal area, diameter, large-tree share, diversity
   and carbon; compositional turnover (Bray--Curtis); mature-canopy gain/loss;
   and a flag marking the period in which each unit's renewal took effect.

Both tables are keyed by ``(unit_type, unit_id)`` and are derived purely from the
processed Parquet panels, so the whole step is deterministic and reproducible.

Build (defaults to unit=nta)::

    python -m stgreen.change.change_metrics --unit nta
"""

from __future__ import annotations

import argparse
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from stgreen import config as C
from stgreen.metrics.structure import STRUCTURE_PANEL_TEMPLATE
from stgreen.metrics.ecoservices import ECOSERVICES_PANEL_TEMPLATE
from stgreen.data.renewal import RENEWAL_PANEL_TEMPLATE

logger = logging.getLogger(__name__)

ANALYSIS_PANEL_TEMPLATE = "analysis_panel_{unit}.parquet"
CHANGE_TABLE_TEMPLATE = "change_table_{unit}.parquet"

# Indicators tracked for change. MAGNITUDE_COLS are strictly-positive extensive
# quantities for which a growth rate is meaningful; the rest get absolute deltas.
OUTCOME_COLS: tuple[str, ...] = (
    "n_alive", "basal_area_m2", "mean_dbh_cm", "large_dbh_share",
    "richness", "shannon", "evenness", "mortality_rate", "cond_good_share",
    "carbon_storage_Mg", "co2e_storage_Mg", "carbon_per_tree_kg",
)
MAGNITUDE_COLS: frozenset[str] = frozenset(
    {"n_alive", "basal_area_m2", "carbon_storage_Mg", "co2e_storage_Mg"}
)

# Columns the ecosystem-services panel contributes to the merged panel.
_ECO_COLS = (
    "n_carbon_trees", "agb_Mg", "carbon_storage_Mg", "co2e_storage_Mg",
    "carbon_per_tree_kg", "extrapolated_share",
)


# --------------------------------------------------------------------------- #
# 1. Merged long analysis panel
# --------------------------------------------------------------------------- #
def build_analysis_panel(unit: str, processed_dir: Path = C.PROCESSED_DIR) -> pd.DataFrame:
    """Merge structure + ecoservices + treatment into one long unit x epoch panel."""
    structure = pd.read_parquet(processed_dir / STRUCTURE_PANEL_TEMPLATE.format(unit=unit))
    eco = pd.read_parquet(processed_dir / ECOSERVICES_PANEL_TEMPLATE.format(unit=unit))
    renewal = pd.read_parquet(processed_dir / RENEWAL_PANEL_TEMPLATE.format(unit=unit))

    keys = ["unit_type", "unit_id", "source_year"]
    eco_slim = eco[[*keys, *[c for c in _ECO_COLS if c in eco.columns]]]
    panel = structure.merge(eco_slim, on=keys, how="left")

    ren_cols = ["unit_type", "unit_id", "treated", "treated_share", "first_treatment_year"]
    renewal_slim = renewal[ren_cols].rename(columns={"first_treatment_year": "treatment_year"})
    panel = panel.merge(renewal_slim, on=["unit_type", "unit_id"], how="left")

    # Units absent from the treatment table (no coordinate-valid trees) are, by
    # construction, untreated and have no treatment year.
    panel["treated"] = panel["treated"].astype("boolean").fillna(False)
    panel["treated_share"] = panel["treated_share"].astype("float64").fillna(0.0)

    # Treatment-timing flags for (generalized) DID / event study.
    treated = panel["treated"].eq(True)
    panel["post_treatment"] = (
        treated & (panel["source_year"] >= panel["treatment_year"])
    ).fillna(False).astype("boolean")
    panel["years_since_treatment"] = np.where(
        treated, panel["source_year"] - panel["treatment_year"], np.nan
    )

    panel = panel.sort_values(["unit_id", "source_year"]).reset_index(drop=True)
    _validate_analysis(panel)
    return panel


def _validate_analysis(panel: pd.DataFrame) -> None:
    assert panel["source_year"].isin(C.CENSUS_YEARS).all(), "unexpected epoch"
    # post_treatment may only be True for treated units.
    bad = panel["post_treatment"].eq(True) & panel["treated"].eq(False)
    assert not bad.any(), "post_treatment set on an untreated unit"


# --------------------------------------------------------------------------- #
# 2. Compositional turnover (Bray--Curtis) from the canonical panel
# --------------------------------------------------------------------------- #
def compositional_turnover(
    tree_panel: pd.DataFrame,
    unit: str,
    periods: list[tuple[int, int]],
    level: str = "genus",
) -> pd.DataFrame:
    """Bray--Curtis dissimilarity of living-tree composition between epochs.

    ``level`` is ``"genus"`` (default, robust to identification noise) or
    ``"species_key"``. Returns long rows: unit columns, ``period``, ``turnover``
    in [0, 1] (0 = identical composition, 1 = no shared individuals).
    """
    unit_cols = list(C.SPATIAL_UNITS[unit])
    live = tree_panel[tree_panel["alive"].eq(True)].dropna(subset=[level, *unit_cols])
    counts = (
        live.groupby([*unit_cols, "source_year", level], observed=True)
        .size().rename("n").reset_index()
    )

    frames = []
    for y1, y2 in periods:
        a = counts[counts["source_year"] == y1]
        b = counts[counts["source_year"] == y2]
        m = a.merge(b, on=[*unit_cols, level], how="outer", suffixes=("_1", "_2"))
        m[["n_1", "n_2"]] = m[["n_1", "n_2"]].fillna(0.0)
        m["min_ab"] = np.minimum(m["n_1"], m["n_2"])
        agg = m.groupby(unit_cols, observed=True).agg(
            sum_min=("min_ab", "sum"), sum_1=("n_1", "sum"), sum_2=("n_2", "sum")
        )
        tot = agg["sum_1"] + agg["sum_2"]
        agg["turnover"] = np.where(tot > 0, 1.0 - 2.0 * agg["sum_min"] / tot, np.nan).round(4)
        agg = agg[["turnover"]].reset_index()
        agg["period"] = f"{y1}_{y2}"
        frames.append(agg)

    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# 3. Between-epoch change table
# --------------------------------------------------------------------------- #
def build_change_table(
    analysis_panel: pd.DataFrame,
    unit: str,
    tree_panel: pd.DataFrame | None = None,
    full_span: bool = True,
) -> pd.DataFrame:
    """Difference the analysis panel between epochs into a per-unit change table.

    Produces consecutive-epoch periods (e.g. 1995_2005, 2005_2015) and, when
    ``full_span`` is set, the first-to-last period (1995_2015). For each outcome
    it emits ``delta_<col>`` and, for extensive magnitudes, ``growth_<col>``.
    Optionally merges compositional turnover when ``tree_panel`` is supplied.
    """
    unit_cols = list(C.SPATIAL_UNITS[unit])
    years = sorted(y for y in analysis_panel["source_year"].unique())
    periods = list(zip(years[:-1], years[1:]))
    if full_span and len(years) > 2:
        periods.append((years[0], years[-1]))

    wide = analysis_panel.set_index(["unit_type", "unit_id", "source_year"])
    rows = []
    for y1, y2 in periods:
        try:
            v1 = wide.xs(y1, level="source_year")
            v2 = wide.xs(y2, level="source_year")
        except KeyError:  # pragma: no cover
            continue
        common = v1.index.intersection(v2.index)
        v1, v2 = v1.loc[common], v2.loc[common]

        rec = pd.DataFrame(index=common)
        for col in OUTCOME_COLS:
            if col not in wide.columns:
                continue
            delta = v2[col] - v1[col]
            rec[f"delta_{col}"] = delta.round(4)
            if col in MAGNITUDE_COLS:
                rec[f"growth_{col}"] = np.where(v1[col] > 0, delta / v1[col], np.nan).round(4)

        # Carry treatment context and mark the period that brackets treatment.
        rec["treated"] = v2["treated"]
        rec["treated_share"] = v2["treated_share"]
        rec["treatment_year"] = v2["treatment_year"]
        rec["treatment_in_period"] = (
            v2["treated"].eq(True)
            & (v2["treatment_year"] > y1) & (v2["treatment_year"] <= y2)
        ).astype("boolean")
        rec["period"] = f"{y1}_{y2}"
        rec["year_start"] = y1
        rec["year_end"] = y2
        rows.append(rec.reset_index())

    change = pd.concat(rows, ignore_index=True)

    if tree_panel is not None:
        turnover = compositional_turnover(tree_panel, unit, periods, level="genus")
        turnover["unit_type"] = unit
        if len(unit_cols) == 1:
            turnover["unit_id"] = turnover[unit_cols[0]].astype("string")
        else:
            turnover["unit_id"] = turnover[unit_cols].astype("string").agg("|".join, axis=1)
        change = change.merge(
            turnover[["unit_type", "unit_id", "period", "turnover"]],
            on=["unit_type", "unit_id", "period"], how="left",
        )

    change = change.sort_values(["unit_id", "year_start"]).reset_index(drop=True)
    _validate_change(change)
    return change


def _validate_change(df: pd.DataFrame) -> None:
    if "turnover" in df.columns:
        t = df["turnover"].dropna()
        assert ((t >= -1e-9) & (t <= 1 + 1e-9)).all(), "turnover outside [0,1]"
    # treatment_in_period implies the unit is treated.
    bad = df["treatment_in_period"].eq(True) & df["treated"].eq(False)
    assert not bad.any(), "treatment_in_period on an untreated unit"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--unit", default=C.DEFAULT_SPATIAL_UNIT, choices=list(C.SPATIAL_UNITS))
    parser.add_argument("--panel", type=Path, default=C.CANONICAL_PANEL_PARQUET,
                        help="Canonical tree panel (for compositional turnover).")
    parser.add_argument("--no-turnover", action="store_true", help="Skip Bray-Curtis turnover.")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    analysis = build_analysis_panel(args.unit)
    tree_panel = None
    if not args.no_turnover:
        if not args.panel.exists():
            raise SystemExit(f"Panel not found: {args.panel} (needed for turnover; use --no-turnover).")
        tree_panel = pd.read_parquet(
            args.panel, columns=["alive", "genus", "species_key", "source_year", *C.SPATIAL_UNITS[args.unit]]
        )
    change = build_change_table(analysis, args.unit, tree_panel=tree_panel)

    logger.info("analysis panel: %d rows; change table: %d rows", len(analysis), len(change))
    if not args.no_save:
        analysis.to_parquet(C.PROCESSED_DIR / ANALYSIS_PANEL_TEMPLATE.format(unit=args.unit), index=False)
        change.to_parquet(C.PROCESSED_DIR / CHANGE_TABLE_TEMPLATE.format(unit=args.unit), index=False)
        logger.info("Wrote analysis + change tables for unit=%s", args.unit)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
