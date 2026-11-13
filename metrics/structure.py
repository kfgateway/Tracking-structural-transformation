"""Structural indicators of the street-tree population per spatial unit and epoch.

This module turns the canonical tree panel (see :mod:`stgreen.data.census`) into
the **(spatial-unit x census-year) structural indicator table** that is the
analytical backbone of the study. Because the 1995/2005/2015 censuses carry no
persistent per-tree identifiers, structural *change* cannot be measured tree by
tree; it is measured as the change in these aggregate indicators within a shared
spatial unit (NTA by default; census tract or borough optionally).

Indicator families
------------------
* **Abundance & mortality** — stems present, living stems, dead/stump stems, and
  the mortality share among determinate-status trees.
* **Size / maturity structure** — basal area (from SI diameter), mean and median
  DBH, the share of large (>= :data:`~stgreen.config.LARGE_DBH_THRESHOLD_CM`)
  trees, and the full DBH size-class distribution.
* **Taxonomic diversity** — species and genus richness, Shannon ``H'``,
  Gini--Simpson, inverse Simpson, Pielou evenness, and Berger--Parker dominance.
* **Condition** — shares on a coarse, cross-year-comparable 3-level scale
  (see the caveat in :data:`~stgreen.config.CONDITION_COARSE_MAP`).

All indicators are computed on the *living* population except abundance and
mortality (which need the dead/stump counts). Groups with no qualifying trees
yield ``NaN`` for ratio/diversity metrics and ``0`` for counts, never a crash.

Build the table from the processed panel::

    python -m stgreen.metrics.structure --unit nta
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from stgreen import config as C

logger = logging.getLogger(__name__)

# Output location for the default spatial unit; the CLI templates the unit in.
STRUCTURE_PANEL_TEMPLATE = "structure_panel_{unit}.parquet"


# --------------------------------------------------------------------------- #
# Small, independently testable helpers
# --------------------------------------------------------------------------- #
def basal_area_m2(dbh_cm: pd.Series) -> pd.Series:
    """Cross-sectional trunk area in square metres from DBH in centimetres.

    Area = pi * r^2 with r = DBH/2 (cm); dividing by 10,000 converts cm^2 to m^2,
    which simplifies to ``pi * dbh_cm**2 / 40000``.
    """
    return np.pi * dbh_cm.astype("float64") ** 2 / 40_000.0


def shannon_index(p: np.ndarray) -> float:
    """Shannon diversity ``H' = -sum(p_i ln p_i)`` for proportions ``p``."""
    p = p[p > 0]
    if p.size == 0:
        return float("nan")
    return float(-(p * np.log(p)).sum())


def simpson_D(p: np.ndarray) -> float:
    """Simpson's ``D = sum(p_i^2)`` (dominance form) for proportions ``p``."""
    if p.size == 0:
        return float("nan")
    return float((p * p).sum())


# --------------------------------------------------------------------------- #
# Diversity and size-structure sub-tables
# --------------------------------------------------------------------------- #
def _diversity_table(live: pd.DataFrame, gcols: list[str]) -> pd.DataFrame:
    """Per-group taxonomic diversity from the living, species-identified trees."""
    sp = live.dropna(subset=["species_key"])
    if sp.empty:
        return pd.DataFrame(
            columns=[
                "richness", "genus_richness", "shannon", "simpson_gini",
                "inv_simpson", "evenness", "dominance_bp",
            ]
        )

    counts = (
        sp.groupby(gcols + ["species_key"], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    totals = counts.groupby(gcols, observed=True)["n"].transform("sum")
    counts["p"] = counts["n"] / totals

    grp = counts.groupby(gcols, observed=True)
    richness = grp["species_key"].nunique().rename("richness")
    shannon = grp["p"].apply(lambda s: shannon_index(s.to_numpy())).rename("shannon")
    simp = grp["p"].apply(lambda s: simpson_D(s.to_numpy())).rename("_D")
    dominance = grp["p"].max().rename("dominance_bp")
    genus_richness = (
        sp.dropna(subset=["genus"]).groupby(gcols, observed=True)["genus"]
        .nunique().rename("genus_richness")
    )

    out = pd.concat([richness, genus_richness, shannon, simp, dominance], axis=1)
    out["simpson_gini"] = 1.0 - out["_D"]
    out["inv_simpson"] = np.where(out["_D"] > 0, 1.0 / out["_D"], np.nan)
    # Pielou evenness is undefined for a single species (ln 1 = 0).
    out["evenness"] = np.where(
        out["richness"] > 1, out["shannon"] / np.log(out["richness"]), np.nan
    )
    return out.drop(columns="_D")


def _size_structure_table(live_dbh: pd.DataFrame, gcols: list[str]) -> pd.DataFrame:
    """Per-group basal area, DBH moments, large-tree share and size-class shares."""
    if live_dbh.empty:
        return pd.DataFrame()

    work = live_dbh.copy()
    work["_ba"] = basal_area_m2(work["dbh_cm"])
    work["_large"] = (work["dbh_cm"] >= C.LARGE_DBH_THRESHOLD_CM).astype(float)
    work["_sc"] = pd.cut(
        work["dbh_cm"],
        bins=list(C.DBH_SIZE_CLASS_EDGES_CM),
        labels=list(C.DBH_SIZE_CLASS_LABELS),
        right=False,
        include_lowest=True,
    )

    grp = work.groupby(gcols, observed=True)
    base = pd.DataFrame(
        {
            "basal_area_m2": grp["_ba"].sum(),
            "mean_dbh_cm": grp["dbh_cm"].mean().round(2),
            "median_dbh_cm": grp["dbh_cm"].median().round(2),
            "large_dbh_share": grp["_large"].mean().round(4),
        }
    )
    # Size-class shares (fraction of measured living trees in each class).
    sc = (
        work.groupby(gcols, observed=True)["_sc"]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
        .reindex(columns=list(C.DBH_SIZE_CLASS_LABELS), fill_value=0.0)
        .round(4)
    )
    return base.join(sc)


def _condition_table(live: pd.DataFrame, gcols: list[str]) -> pd.DataFrame:
    """Per-group coarse condition shares among living trees with known condition."""
    work = live.assign(_cc=live["condition"].map(C.CONDITION_COARSE_MAP))
    work = work.dropna(subset=["_cc"])
    if work.empty:
        return pd.DataFrame()
    shares = (
        work.groupby(gcols, observed=True)["_cc"]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
        .reindex(columns=["Good", "Fair", "Poor"], fill_value=0.0)
        .round(4)
    )
    return shares.rename(
        columns={"Good": "cond_good_share", "Fair": "cond_fair_share", "Poor": "cond_poor_share"}
    )


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def compute_structure_panel(
    panel: pd.DataFrame,
    unit: str = C.DEFAULT_SPATIAL_UNIT,
) -> pd.DataFrame:
    """Compute the (spatial-unit x census-year) structural indicator table.

    Parameters
    ----------
    panel:
        Canonical tree panel from :func:`stgreen.data.census.build_panel`.
    unit:
        Spatial unit, one of :data:`stgreen.config.SPATIAL_UNITS` keys
        (``"nta"``, ``"census_tract"`` or ``"borough"``).

    Returns
    -------
    pandas.DataFrame
        One row per (unit, year) with abundance, mortality, size/maturity,
        diversity and condition indicators. Carries ``unit_type`` and a string
        ``unit_id`` so tables for different units share one schema.
    """
    if unit not in C.SPATIAL_UNITS:
        raise ValueError(f"Unknown unit {unit!r}; expected one of {tuple(C.SPATIAL_UNITS)}")

    unit_cols = list(C.SPATIAL_UNITS[unit])
    gcols = [*unit_cols, "source_year"]

    # Only real tree records (excludes 1995 empty "Planting Space" sites) with a
    # fully-resolved spatial key participate in the aggregation.
    present = panel[panel["tree_present"].eq(True)].copy()
    before = len(present)
    present = present.dropna(subset=unit_cols)
    dropped = before - len(present)
    if dropped:
        logger.warning("Dropped %d records with missing %s key", dropped, unit)

    present["_alive"] = present["alive"].eq(True)
    present["_dead"] = present["alive"].eq(False)  # dead/stump (NA status excluded)

    gp = present.groupby(gcols, observed=True)
    counts = pd.DataFrame(
        {
            "n_present": gp.size(),
            "n_alive": gp["_alive"].sum(),
            "n_dead_stump": gp["_dead"].sum(),
        }
    )
    # Mortality share among determinate-status trees (alive + dead); NaN if none.
    denom = counts["n_alive"] + counts["n_dead_stump"]
    counts["mortality_rate"] = np.where(denom > 0, counts["n_dead_stump"] / denom, np.nan).round(4)

    live = present[present["_alive"]]
    live_dbh = live[live["dbh_valid"].eq(True) & live["dbh_cm"].notna()]

    n_dbh = live_dbh.groupby(gcols, observed=True).size().rename("n_dbh_measured")
    size_tbl = _size_structure_table(live_dbh, gcols)
    div_tbl = _diversity_table(live, gcols)
    cond_tbl = _condition_table(live, gcols)

    out = counts.join([n_dbh, size_tbl, div_tbl, cond_tbl], how="left")
    out["n_dbh_measured"] = out["n_dbh_measured"].fillna(0).astype("int64")
    out["basal_area_m2"] = out["basal_area_m2"].fillna(0.0).round(3)
    # basal area per living tree (m^2), a density-free structural intensity.
    out["basal_area_per_tree_m2"] = np.where(
        out["n_alive"] > 0, (out["basal_area_m2"] / out["n_alive"]).round(4), np.nan
    )

    out = out.reset_index()
    out.insert(0, "unit_type", unit)
    if len(unit_cols) == 1:
        out.insert(1, "unit_id", out[unit_cols[0]].astype("string"))
    else:
        out.insert(
            1, "unit_id",
            out[unit_cols].astype("string").agg("|".join, axis=1).astype("string"),
        )

    out = out.sort_values(["unit_id", "source_year"]).reset_index(drop=True)
    _validate_structure(out)
    return out


def _validate_structure(df: pd.DataFrame) -> None:
    """Guard structural invariants of the indicator table."""
    assert (df["n_alive"] + df["n_dead_stump"] <= df["n_present"]).all(), (
        "alive + dead exceeds present (status accounting error)"
    )
    for col in ("mortality_rate", "large_dbh_share", "evenness"):
        s = df[col].dropna()
        assert ((s >= -1e-9) & (s <= 1 + 1e-9)).all(), f"{col} outside [0, 1]"
    # Size-class shares must sum to ~1 wherever any tree was measured. Shares are
    # stored rounded to 4 dp, so the tolerance accommodates accumulated rounding
    # across the six classes (<= 6 * 5e-5), not floating-point noise alone.
    sc = df[list(C.DBH_SIZE_CLASS_LABELS)].sum(axis=1)
    measured = df["n_dbh_measured"] > 0
    assert np.allclose(sc[measured], 1.0, atol=1e-3), "size-class shares do not sum to 1"


def save_structure_panel(df: pd.DataFrame, unit: str, out_dir: Path = C.PROCESSED_DIR) -> Path:
    """Write the structural indicator table to Parquet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / STRUCTURE_PANEL_TEMPLATE.format(unit=unit)
    df.to_parquet(path, index=False)
    logger.info("Wrote %d rows to %s", len(df), path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--panel", type=Path, default=C.CANONICAL_PANEL_PARQUET,
                        help="Input canonical tree panel (Parquet).")
    parser.add_argument("--unit", default=C.DEFAULT_SPATIAL_UNIT,
                        choices=list(C.SPATIAL_UNITS), help="Spatial unit.")
    parser.add_argument("--no-save", action="store_true", help="Build but do not write output.")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging to warnings.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.panel.exists():
        raise SystemExit(
            f"Panel not found: {args.panel}. Run `python -m stgreen.data.census` first."
        )
    panel = pd.read_parquet(args.panel)
    structure = compute_structure_panel(panel, unit=args.unit)

    logger.info(
        "Built %s structural panel: %d units x years, %d columns",
        args.unit, len(structure), structure.shape[1],
    )
    if not args.no_save:
        save_structure_panel(structure, args.unit)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
