"""Carbon ecosystem services of the street-tree population from SI diameter.

This module estimates the carbon stored by living street trees and aggregates it
to the (spatial-unit x census-year) table that sits alongside the structural
indicators from :mod:`stgreen.metrics.structure`. It is the *ecosystem-services*
pillar of the study and is derived entirely from trunk diameter, so it is fully
reproducible from the inventories alone, with no external imagery or API.

Method (all constants are declared and cited in :mod:`stgreen.config`)
----------------------------------------------------------------------
1. Total aboveground biomass per tree follows Jenkins et al. (2003),
   ``AGB_kg = exp(beta0 + beta1 * ln(dbh_cm))``, with species-group coefficients
   assigned from genus (and species, for maples that span groups).
2. Open-grown urban trees carry less biomass than forest trees of equal DBH, so
   AGB is multiplied by 0.8 (Nowak & Crane, 2002; Nowak, 1994).
3. Whole-tree biomass adds coarse roots via a root-to-shoot ratio of 0.26
   (Cairns et al., 1997).
4. Stored carbon is 0.5 x dry biomass; CO2-equivalent is carbon x 44/12.

Carbon is computed only for **living** trees with a **valid** DBH; dead stems and
stumps are excluded (decaying wood is not counted as stored live-tree carbon).
Trees outside a group's fitted DBH range are flagged (``carbon_extrapolated``)
but retained. A growth-based gross-sequestration estimate is available but
optional and explicitly approximate; the preferred sequestration figure is the
empirical cross-epoch change in stored carbon built in a later batch.

Build the per-unit carbon table from the processed panel::

    python -m stgreen.metrics.ecoservices --unit nta
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from stgreen import config as C

logger = logging.getLogger(__name__)

ECOSERVICES_PANEL_TEMPLATE = "ecoservices_panel_{unit}.parquet"


# --------------------------------------------------------------------------- #
# Species-group assignment
# --------------------------------------------------------------------------- #
def assign_jenkins_group(species_key: object, genus: object) -> str:
    """Return the Jenkins (2003) species group for a tree.

    Resolution order: exact species override -> genus map -> default group
    (``mixed_hardwood``). ``species_key`` and ``genus`` are the lower-case
    fields produced by the census harmoniser.
    """
    if isinstance(species_key, str):
        grp = C.SPECIES_TO_JENKINS.get(species_key)
        if grp is not None:
            return grp
    if isinstance(genus, str):
        grp = C.GENUS_TO_JENKINS.get(genus.lower())
        if grp is not None:
            return grp
    return C.JENKINS_DEFAULT_GROUP


# --------------------------------------------------------------------------- #
# Per-tree carbon
# --------------------------------------------------------------------------- #
def aboveground_biomass_kg(dbh_cm: pd.Series, group: pd.Series) -> pd.Series:
    """Vectorised Jenkins (2003) total aboveground biomass (kg, oven-dry)."""
    beta0 = group.map(lambda g: C.JENKINS_COEFFS[g][0]).astype("float64")
    beta1 = group.map(lambda g: C.JENKINS_COEFFS[g][1]).astype("float64")
    return np.exp(beta0 + beta1 * np.log(dbh_cm.astype("float64")))


def compute_tree_ecoservices(panel: pd.DataFrame) -> pd.DataFrame:
    """Add per-tree carbon columns for living, valid-DBH trees.

    Returns a copy of ``panel`` restricted to living, measured trees with the
    added columns: ``jenkins_group``, ``agb_kg`` (urban-adjusted aboveground
    biomass), ``biomass_total_kg`` (incl. roots), ``carbon_kg``, ``co2e_kg`` and
    ``carbon_extrapolated``.
    """
    live = panel[panel["alive"].eq(True) & panel["dbh_valid"].eq(True) & panel["dbh_cm"].notna()].copy()

    live["jenkins_group"] = [
        assign_jenkins_group(sk, g)
        for sk, g in zip(live["species_key"], live["genus"], strict=True)
    ]

    agb_raw = aboveground_biomass_kg(live["dbh_cm"], live["jenkins_group"])
    live["agb_kg"] = (agb_raw * C.URBAN_BIOMASS_ADJ).round(3)
    live["biomass_total_kg"] = (live["agb_kg"] * (1.0 + C.ROOT_SHOOT_RATIO)).round(3)
    live["carbon_kg"] = (live["biomass_total_kg"] * C.CARBON_FRACTION).round(3)
    live["co2e_kg"] = (live["carbon_kg"] * C.CO2_PER_C).round(3)

    group_max = live["jenkins_group"].map(C.JENKINS_MAX_DBH_CM)
    live["carbon_extrapolated"] = (
        (live["dbh_cm"] < C.JENKINS_MIN_DBH_CM) | (live["dbh_cm"] > group_max)
    ).astype("boolean")
    return live


def estimate_gross_sequestration_kg(tree_df: pd.DataFrame) -> pd.Series:
    """Approximate annual gross carbon sequestration per tree (kg C / yr).

    Grows DBH by a condition-scaled annual increment, recomputes stored carbon
    and differences it. This is a coarse proxy; prefer the empirical cross-epoch
    carbon change. Returns a Series aligned to ``tree_df``.
    """
    factor = tree_df["condition"].map(C.CONDITION_GROWTH_FACTOR).fillna(
        C.CONDITION_GROWTH_FACTOR["Fair"]
    )
    dbh2 = tree_df["dbh_cm"] + C.DBH_GROWTH_CM_PER_YR * factor
    agb2 = aboveground_biomass_kg(dbh2, tree_df["jenkins_group"]) * C.URBAN_BIOMASS_ADJ
    carbon2 = agb2 * (1.0 + C.ROOT_SHOOT_RATIO) * C.CARBON_FRACTION
    return (carbon2 - tree_df["carbon_kg"]).clip(lower=0.0).round(4)


# --------------------------------------------------------------------------- #
# Aggregation to (spatial-unit x year)
# --------------------------------------------------------------------------- #
def aggregate_ecoservices(
    tree_df: pd.DataFrame,
    unit: str = C.DEFAULT_SPATIAL_UNIT,
    sequestration: bool = False,
) -> pd.DataFrame:
    """Aggregate per-tree carbon to one row per (unit, year).

    Megagrams (Mg = tonnes) are used for unit totals; per-tree means stay in kg.
    """
    if unit not in C.SPATIAL_UNITS:
        raise ValueError(f"Unknown unit {unit!r}; expected one of {tuple(C.SPATIAL_UNITS)}")

    unit_cols = list(C.SPATIAL_UNITS[unit])
    gcols = [*unit_cols, "source_year"]
    df = tree_df.dropna(subset=unit_cols).copy()

    if sequestration:
        df["seq_kg_yr"] = estimate_gross_sequestration_kg(df)

    grp = df.groupby(gcols, observed=True)
    out = pd.DataFrame(
        {
            "n_carbon_trees": grp.size(),
            "agb_Mg": grp["agb_kg"].sum() / 1000.0,
            "carbon_storage_Mg": grp["carbon_kg"].sum() / 1000.0,
            "co2e_storage_Mg": grp["co2e_kg"].sum() / 1000.0,
            "carbon_per_tree_kg": grp["carbon_kg"].mean(),
            "extrapolated_share": grp["carbon_extrapolated"].mean(),
        }
    )
    if sequestration:
        out["seq_MgC_per_yr"] = grp["seq_kg_yr"].sum() / 1000.0

    out = out.round(
        {"agb_Mg": 3, "carbon_storage_Mg": 3, "co2e_storage_Mg": 3,
         "carbon_per_tree_kg": 2, "extrapolated_share": 4,
         **({"seq_MgC_per_yr": 4} if sequestration else {})}
    ).reset_index()

    out.insert(0, "unit_type", unit)
    if len(unit_cols) == 1:
        out.insert(1, "unit_id", out[unit_cols[0]].astype("string"))
    else:
        out.insert(
            1, "unit_id",
            out[unit_cols].astype("string").agg("|".join, axis=1).astype("string"),
        )
    out = out.sort_values(["unit_id", "source_year"]).reset_index(drop=True)
    _validate(out)
    return out


def _validate(df: pd.DataFrame) -> None:
    assert (df["carbon_storage_Mg"] >= 0).all(), "negative carbon storage"
    # CO2e must equal carbon * 44/12 within rounding.
    assert np.allclose(
        df["co2e_storage_Mg"], df["carbon_storage_Mg"] * C.CO2_PER_C, rtol=1e-3, atol=1e-2
    ), "CO2e inconsistent with stored carbon"
    assert ((df["extrapolated_share"] >= 0) & (df["extrapolated_share"] <= 1)).all()


def save_ecoservices_panel(df: pd.DataFrame, unit: str, out_dir: Path = C.PROCESSED_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ECOSERVICES_PANEL_TEMPLATE.format(unit=unit)
    df.to_parquet(path, index=False)
    logger.info("Wrote %d rows to %s", len(df), path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--panel", type=Path, default=C.CANONICAL_PANEL_PARQUET)
    parser.add_argument("--unit", default=C.DEFAULT_SPATIAL_UNIT, choices=list(C.SPATIAL_UNITS))
    parser.add_argument("--sequestration", action="store_true",
                        help="Also estimate approximate growth-based gross sequestration.")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if not args.panel.exists():
        raise SystemExit(f"Panel not found: {args.panel}. Run `python -m stgreen.data.census` first.")

    panel = pd.read_parquet(args.panel)
    trees = compute_tree_ecoservices(panel)
    eco = aggregate_ecoservices(trees, unit=args.unit, sequestration=args.sequestration)

    total = eco.groupby("source_year")["carbon_storage_Mg"].sum()
    logger.info("Total living street-tree carbon storage by epoch (Mg C):\n%s", total.round(0).to_string())
    if not args.no_save:
        save_ecoservices_panel(eco, args.unit)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
