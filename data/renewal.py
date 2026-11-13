"""Assign urban-renewal treatment status to spatial units.

The study's causal question is how street-greenery structure changed *because of*
urban renewal. This module turns an urban-renewal / rezoning polygon layer into a
per-spatial-unit treatment table: whether a unit was renewed, in which year, and
how much of its street-tree stock lies inside a renewal footprint. That table is
the treatment definition consumed by the difference-in-differences analysis.

Expected real data (NYC, open, no API, non-Chinese)
---------------------------------------------------
Any polygon layer with an effective date works. The two canonical NYC sources:

* **NYC DCP Zoning Map Amendments ("nyzma")** — rezoning footprints with an
  effective date; the usual operationalisation of "urban renewal / redevelopment".
* **Urban Renewal Areas** — formally designated URAs with a designation date.

Download from NYC Open Data / the DCP "BYTES of the BIG APPLE" site as GeoJSON or
Shapefile and pass the path to :func:`load_renewal_polygons`, naming the date and
id columns. Geometry may be in any CRS; it is reprojected to WGS84 internally.

Until the real layer is in hand, :func:`make_synthetic_renewal` yields a small
fixture so the pipeline runs and is testable end to end.

Build the treatment table (synthetic fixture by default)::

    python -m stgreen.data.renewal --unit nta --synthetic
    python -m stgreen.data.renewal --unit nta --renewal path/to/nyzma.geojson \\
        --date-col EffectiveDate --id-col ProjectID
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    from shapely.geometry import box
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "renewal.py requires geopandas and shapely (`pip install geopandas shapely`)."
    ) from exc

from stgreen import config as C

logger = logging.getLogger(__name__)

RENEWAL_PANEL_TEMPLATE = "renewal_treatment_{unit}.parquet"


# --------------------------------------------------------------------------- #
# Loading / normalising the renewal polygon layer
# --------------------------------------------------------------------------- #
def _coerce_year(values: pd.Series) -> pd.Series:
    """Parse a date/year column to an integer effective year, within bounds."""
    # Accept full dates ("2008-06-30"), year strings ("2008"), or integers.
    dt = pd.to_datetime(values, errors="coerce")
    year = dt.dt.year
    # Fall back to a plain 4-digit year where date parsing failed.
    plain = pd.to_numeric(values.astype("string").str.extract(r"(\d{4})")[0], errors="coerce")
    year = year.fillna(plain)
    year = year.where(
        (year >= C.RENEWAL_MIN_YEAR) & (year <= C.RENEWAL_MAX_YEAR)
    )
    return year.astype("Int64")


def load_renewal_polygons(
    path: Path,
    date_col: str,
    id_col: str,
) -> "gpd.GeoDataFrame":
    """Load a renewal/rezoning polygon layer and normalise it.

    Returns a GeoDataFrame in WGS84 with columns ``renewal_id``,
    ``treatment_year`` (Int64) and ``geometry``. Rows with an unparseable or
    out-of-range year, or with empty/invalid geometry, are dropped (and counted).
    """
    gdf = gpd.read_file(path)
    if date_col not in gdf.columns or id_col not in gdf.columns:
        raise KeyError(
            f"Expected columns {date_col!r} and {id_col!r}; found {list(gdf.columns)}"
        )
    gdf = gdf.rename(columns={id_col: "renewal_id"})
    gdf["treatment_year"] = _coerce_year(gdf[date_col])
    gdf = gdf[["renewal_id", "treatment_year", "geometry"]]

    if gdf.crs is None:
        raise ValueError("Renewal layer has no CRS; set one before loading.")
    gdf = gdf.to_crs(C.CRS_WGS84)

    n0 = len(gdf)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty & gdf.geometry.is_valid]
    gdf = gdf.dropna(subset=["treatment_year"])
    dropped = n0 - len(gdf)
    if dropped:
        logger.warning("Dropped %d renewal polygons (bad geometry or year)", dropped)
    logger.info("Loaded %d renewal polygons (%s)", len(gdf), path)
    return gdf.reset_index(drop=True)


def make_synthetic_renewal(seed: int = 0) -> "gpd.GeoDataFrame":
    """Small, deterministic fixture covering parts of NYC, for tests/demos.

    Three rectangular footprints over Brooklyn/Queens with effective years, so
    the treatment pipeline can be exercised without the real layer.
    """
    _ = seed  # reserved for future randomised fixtures; kept deterministic here
    polys = [
        ("RZ_bk_central", 2008, box(-73.99, 40.66, -73.92, 40.70)),  # central Brooklyn
        ("RZ_qn_west", 2012, box(-73.96, 40.73, -73.90, 40.77)),     # western Queens (LIC-ish)
        ("RZ_bx_south", 2010, box(-73.93, 40.80, -73.88, 40.84)),    # south Bronx
    ]
    gdf = gpd.GeoDataFrame(
        {"renewal_id": [p[0] for p in polys], "treatment_year": pd.array([p[1] for p in polys], dtype="Int64")},
        geometry=[p[2] for p in polys],
        crs=C.CRS_WGS84,
    )
    return gdf


# --------------------------------------------------------------------------- #
# Treatment assignment
# --------------------------------------------------------------------------- #
def assign_tree_treatment(panel: pd.DataFrame, polygons: "gpd.GeoDataFrame") -> pd.DataFrame:
    """Flag each coordinate-valid tree that falls within a renewal footprint.

    Returns the input trees (coordinate-valid only) with two added columns:
    ``treated_tree`` (bool) and ``tree_treatment_year`` (Int64, earliest
    overlapping renewal year). A point inside overlapping footprints takes the
    earliest year.
    """
    pts = panel[panel["coord_valid"].eq(True)].reset_index(drop=True).copy()
    gpts = gpd.GeoDataFrame(
        pts[["lon", "lat"]],
        geometry=gpd.points_from_xy(pts["lon"], pts["lat"]),
        crs=C.CRS_WGS84,
    )
    poly = polygons[["renewal_id", "treatment_year", "geometry"]]
    joined = gpd.sjoin(gpts, poly, how="left", predicate="within")

    # A point can match several polygons -> several rows; collapse to one.
    g = joined.groupby(level=0)
    pts["treated_tree"] = g["renewal_id"].apply(lambda s: s.notna().any())
    pts["tree_treatment_year"] = g["treatment_year"].min().astype("Int64")
    return pts


def build_treatment_table(
    panel: pd.DataFrame,
    polygons: "gpd.GeoDataFrame",
    unit: str = C.DEFAULT_SPATIAL_UNIT,
    treated_threshold: float = C.RENEWAL_TREATED_SHARE_THRESHOLD,
) -> pd.DataFrame:
    """Build the per-unit treatment table (one row per spatial unit).

    Treatment is a time-invariant property of place: a unit's ``treated`` flag,
    ``treated_share`` (share of its street trees inside a renewal footprint,
    pooled across censuses) and ``first_treatment_year``. The DID batch applies
    the pre/post split using the census epochs.
    """
    if unit not in C.SPATIAL_UNITS:
        raise ValueError(f"Unknown unit {unit!r}; expected one of {tuple(C.SPATIAL_UNITS)}")

    unit_cols = list(C.SPATIAL_UNITS[unit])
    trees = assign_tree_treatment(panel, polygons)
    trees = trees.dropna(subset=unit_cols + ["lon"]).copy()
    # unit columns must be carried over from the panel for grouping.
    trees[unit_cols] = panel.loc[panel["coord_valid"].eq(True), unit_cols].reset_index(drop=True).loc[trees.index, unit_cols]

    grp = trees.groupby(unit_cols, observed=True)
    out = pd.DataFrame(
        {
            "n_trees_total": grp.size(),
            "n_trees_treated": grp["treated_tree"].sum(),
            "first_treatment_year": grp["tree_treatment_year"].min(),
        }
    )
    out["treated_share"] = (out["n_trees_treated"] / out["n_trees_total"]).round(4)
    out["treated"] = (out["treated_share"] >= treated_threshold).astype("boolean")
    # A unit's treatment year is meaningful only where it is treated.
    out["first_treatment_year"] = out["first_treatment_year"].astype("Int64").where(out["treated"])

    out = out.reset_index()
    out.insert(0, "unit_type", unit)
    if len(unit_cols) == 1:
        out.insert(1, "unit_id", out[unit_cols[0]].astype("string"))
    else:
        out.insert(
            1, "unit_id",
            out[unit_cols].astype("string").agg("|".join, axis=1).astype("string"),
        )
    out = out.sort_values(["unit_id"]).reset_index(drop=True)
    _validate(out)
    return out


def _validate(df: pd.DataFrame) -> None:
    assert (df["n_trees_treated"] <= df["n_trees_total"]).all(), "treated exceeds total"
    assert ((df["treated_share"] >= 0) & (df["treated_share"] <= 1)).all(), "share outside [0,1]"
    yr = df["first_treatment_year"].dropna()
    assert ((yr >= C.RENEWAL_MIN_YEAR) & (yr <= C.RENEWAL_MAX_YEAR)).all(), "treatment year out of range"
    # Treatment year must be present exactly where treated, and absent otherwise.
    assert df.loc[df["treated"].eq(True), "first_treatment_year"].notna().all(), "treated unit lacks year"
    assert df.loc[df["treated"].eq(False), "first_treatment_year"].isna().all(), "untreated unit has year"


def save_treatment_table(df: pd.DataFrame, unit: str, out_dir: Path = C.PROCESSED_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / RENEWAL_PANEL_TEMPLATE.format(unit=unit)
    df.to_parquet(path, index=False)
    logger.info("Wrote %d rows to %s", len(df), path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--panel", type=Path, default=C.CANONICAL_PANEL_PARQUET)
    parser.add_argument("--unit", default=C.DEFAULT_SPATIAL_UNIT, choices=list(C.SPATIAL_UNITS))
    parser.add_argument("--renewal", type=Path, default=None, help="Path to the renewal polygon layer.")
    parser.add_argument("--date-col", default=None, help="Effective-date column in the renewal layer.")
    parser.add_argument("--id-col", default=None, help="Identifier column in the renewal layer.")
    parser.add_argument("--synthetic", action="store_true", help="Use the built-in synthetic fixture.")
    parser.add_argument("--threshold", type=float, default=C.RENEWAL_TREATED_SHARE_THRESHOLD)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if not args.panel.exists():
        raise SystemExit(f"Panel not found: {args.panel}. Run `python -m stgreen.data.census` first.")

    if args.synthetic or args.renewal is None:
        logger.warning("Using SYNTHETIC renewal fixture (not real NYC data).")
        polygons = make_synthetic_renewal()
    else:
        if not (args.date_col and args.id_col):
            raise SystemExit("--renewal requires --date-col and --id-col.")
        polygons = load_renewal_polygons(args.renewal, date_col=args.date_col, id_col=args.id_col)

    panel = pd.read_parquet(args.panel)
    table = build_treatment_table(panel, polygons, unit=args.unit, treated_threshold=args.threshold)

    n_treated = int(table["treated"].sum())
    logger.info("Treated units: %d / %d (threshold=%.2f)", n_treated, len(table), args.threshold)
    if not args.no_save:
        save_treatment_table(table, args.unit)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
