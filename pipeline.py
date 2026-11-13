"""End-to-end pipeline: raw censuses -> processed panels -> results -> figures.

One entry point runs every stage in dependency order so a reviewer reproduces all
numbers and figures from scratch::

    python -m stgreen.pipeline                      # full run (synthetic renewal)
    python -m stgreen.pipeline --renewal nyzma.geojson --date-col EffectiveDate --id-col ProjectID
    python -m stgreen.pipeline --no-figures --no-did

Every stage is idempotent and writes to ``data/processed`` (tables), ``outputs/
tables`` (results) and ``outputs/figures`` (figures). The pipeline calls library
functions directly (no subprocesses), so a failure surfaces a normal traceback.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from stgreen import config as cfg
from stgreen.data import census, renewal
from stgreen.metrics import structure, ecoservices
from stgreen.change import change_metrics as change
from stgreen.analysis import did as did_mod

logger = logging.getLogger(__name__)

ANALYSIS_UNITS = ("nta", "borough")  # borough tables feed the figures; nta drives analysis


def run_pipeline(
    units: tuple[str, ...] = ANALYSIS_UNITS,
    renewal_path: Path | None = None,
    date_col: str | None = None,
    id_col: str | None = None,
    run_figures: bool = True,
    run_did: bool = True,
    run_gvi: bool = True,
    gvi_points_per_unit: int = 50,
) -> None:
    """Execute the whole chain. ``renewal_path=None`` uses the synthetic fixture.

    ``run_gvi`` adds the optional street-view Green View Index layer using an
    OFFLINE SYNTHETIC sample (no imagery); set it False to skip, or supply real
    imagery via :mod:`stgreen.svi.harvest` outside the pipeline.
    """
    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cfg.TABLES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("[1/8] Harmonising censuses -> canonical panel")
    panel = census.build_panel()
    census.save_panel(panel)

    logger.info("[2/8] Structural indicators")
    for u in units:
        structure.save_structure_panel(structure.compute_structure_panel(panel, unit=u), u)

    logger.info("[3/8] Carbon ecosystem services")
    trees = ecoservices.compute_tree_ecoservices(panel)
    for u in units:
        ecoservices.save_ecoservices_panel(
            ecoservices.aggregate_ecoservices(trees, unit=u, sequestration=True), u
        )

    logger.info("[4/8] Urban-renewal treatment (%s)", "real" if renewal_path else "SYNTHETIC")
    if renewal_path is not None:
        if not (date_col and id_col):
            raise SystemExit("--renewal requires --date-col and --id-col.")
        polygons = renewal.load_renewal_polygons(renewal_path, date_col=date_col, id_col=id_col)
    else:
        polygons = renewal.make_synthetic_renewal()
    renewal.save_treatment_table(renewal.build_treatment_table(panel, polygons, unit="nta"), "nta")

    logger.info("[5/8] Change tables (analysis + deltas)")
    analysis = change.build_analysis_panel("nta")
    tree_cols = pd.read_parquet(
        cfg.CANONICAL_PANEL_PARQUET,
        columns=["alive", "genus", "species_key", "source_year", *cfg.SPATIAL_UNITS["nta"]],
    )
    change_tbl = change.build_change_table(analysis, "nta", tree_panel=tree_cols)
    analysis.to_parquet(cfg.PROCESSED_DIR / change.ANALYSIS_PANEL_TEMPLATE.format(unit="nta"), index=False)
    change_tbl.to_parquet(cfg.PROCESSED_DIR / change.CHANGE_TABLE_TEMPLATE.format(unit="nta"), index=False)

    if run_did:
        logger.info("[6/8] DID estimates -> outputs/tables")
        suite = did_mod.run_did_suite(analysis, change_tbl)
        suite.to_csv(cfg.TABLES_DIR / "did_suite_nta.csv", index=False)
        robust = pd.DataFrame([
            did_mod.estimate_twfe(analysis, "carbon_storage_Mg"),
            did_mod.parallel_trends_test(change_tbl, "carbon_storage_Mg"),
        ])
        robust.to_csv(cfg.TABLES_DIR / "did_robustness_nta.csv", index=False)
    else:
        logger.info("[6/8] DID skipped")

    if run_gvi:
        logger.info("[7/8] Green View Index (SYNTHETIC offline sample)")
        from stgreen.svi.harvest import sample_points, make_synthetic_records
        from stgreen.metrics import gvi as gvi_mod
        for u in units:
            pts = sample_points(panel, unit=u, n_per_unit=gvi_points_per_unit)
            recs = make_synthetic_records(pts)
            g = gvi_mod.aggregate_gvi(gvi_mod.score_records(recs), unit=u)
            g.to_parquet(cfg.PROCESSED_DIR / gvi_mod.GVI_PANEL_TEMPLATE.format(unit=u), index=False)
    else:
        logger.info("[7/8] GVI skipped")

    if run_figures:
        logger.info("[8/8] Rendering figures -> outputs/figures")
        from stgreen.viz import figures  # imported here so matplotlib stays optional
        figures.generate_all()
    else:
        logger.info("[8/8] Figures skipped")

    logger.info("Pipeline complete.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--renewal", type=Path, default=None, help="Real renewal polygon layer.")
    parser.add_argument("--date-col", default=None)
    parser.add_argument("--id-col", default=None)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-did", action="store_true")
    parser.add_argument("--no-gvi", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    run_pipeline(
        renewal_path=args.renewal, date_col=args.date_col, id_col=args.id_col,
        run_figures=not args.no_figures, run_did=not args.no_did, run_gvi=not args.no_gvi,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
