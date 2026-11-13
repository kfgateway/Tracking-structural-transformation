"""Green View Index (GVI) per spatial unit and epoch from street-view imagery.

GVI is the share of the street-level visual field occupied by vegetation
(Yang et al., 2009; Li et al., 2015). Given image records with a per-image
vegetation fraction, this module computes the GVI at each sample point (mean over
its images) and aggregates to the (spatial-unit x epoch) table that joins the
"function" layer to the structural and ecosystem-service panels.

Two entry points:

* :func:`score_records` -- fill a ``green_fraction`` column by scoring each image
  file with a :class:`~stgreen.svi.green.Segmenter` (default: Excess-Green). For
  offline/synthetic records, the pre-computed ``green_truth`` is used instead.
* :func:`aggregate_gvi` -- point-level then unit-level GVI (0-100 scale), with the
  epoch renamed ``source_year`` so the result merges with the analysis panel when
  SVI epochs match the census years.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from stgreen import config as cfg
from stgreen.svi.green import ExGSegmenter, Segmenter, image_green_fraction

logger = logging.getLogger(__name__)

GVI_PANEL_TEMPLATE = "gvi_panel_{unit}.parquet"


def score_records(records: pd.DataFrame, image_dir: Path | None = None,
                  segmenter: Segmenter | None = None) -> pd.DataFrame:
    """Add a ``green_fraction`` column to image records.

    If a record carries ``green_truth`` (offline fixture) that value is used;
    otherwise the image file ``<image_dir>/<image_id>.jpg`` is scored with
    ``segmenter`` (default Excess-Green).
    """
    df = records.copy()
    if "green_truth" in df.columns:
        df["green_fraction"] = df["green_truth"].astype("float64")
        return df
    if image_dir is None:
        raise ValueError("image_dir is required to score real image files.")
    seg = segmenter or ExGSegmenter()
    df["green_fraction"] = [
        image_green_fraction(Path(image_dir) / f"{iid}.jpg", seg) for iid in df["image_id"]
    ]
    return df


def aggregate_gvi(scored: pd.DataFrame, unit: str) -> pd.DataFrame:
    """Aggregate scored image records to GVI per (unit, epoch).

    Point GVI = mean image vegetation fraction x 100; unit GVI = mean point GVI.
    """
    if "green_fraction" not in scored.columns:
        raise ValueError("records must be scored first (call score_records).")
    point = (
        scored.groupby(["unit_id", "epoch", "point_id"], observed=True)["green_fraction"]
        .mean().mul(100.0).rename("point_gvi").reset_index()
    )
    agg = (
        point.groupby(["unit_id", "epoch"], observed=True)
        .agg(gvi_mean=("point_gvi", "mean"),
             gvi_sd=("point_gvi", "std"),
             n_points=("point_id", "nunique"))
        .reset_index()
    )
    n_img = (
        scored.groupby(["unit_id", "epoch"], observed=True).size().rename("n_images").reset_index()
    )
    agg = agg.merge(n_img, on=["unit_id", "epoch"])
    agg["unit_type"] = unit
    agg = agg.rename(columns={"epoch": "source_year"})
    agg[["gvi_mean", "gvi_sd"]] = agg[["gvi_mean", "gvi_sd"]].round(3)
    _validate(agg)
    cols = ["unit_type", "unit_id", "source_year", "gvi_mean", "gvi_sd", "n_points", "n_images"]
    return agg[cols].sort_values(["unit_id", "source_year"]).reset_index(drop=True)


def _validate(df: pd.DataFrame) -> None:
    g = df["gvi_mean"].dropna()
    assert ((g >= 0) & (g <= 100)).all(), "GVI outside [0, 100]"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--unit", default=cfg.DEFAULT_SPATIAL_UNIT, choices=list(cfg.SPATIAL_UNITS))
    parser.add_argument("--panel", type=Path, default=cfg.CANONICAL_PANEL_PARQUET)
    parser.add_argument("--n-per-unit", type=int, default=25,
                        help="Sample points per unit (synthetic offline demo).")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    # Offline demonstration path: sample points, synthesise records, score, aggregate.
    from stgreen.svi.harvest import sample_points, make_synthetic_records
    panel = pd.read_parquet(args.panel, columns=["coord_valid", "lon", "lat", *cfg.SPATIAL_UNITS[args.unit]])
    points = sample_points(panel, args.unit, n_per_unit=args.n_per_unit)
    logger.warning("Using SYNTHETIC street-view records (no imagery harvested).")
    records = make_synthetic_records(points)
    gvi = aggregate_gvi(score_records(records), args.unit)
    gvi.to_parquet(cfg.PROCESSED_DIR / GVI_PANEL_TEMPLATE.format(unit=args.unit), index=False)
    logger.info("Wrote GVI panel: %d rows (unit=%s)", len(gvi), args.unit)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
