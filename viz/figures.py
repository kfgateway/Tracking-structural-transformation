"""Publication figures for the manuscript, rendered from the processed tables.

Every figure is generated deterministically from the Parquet outputs of the
pipeline (and, where needed, the canonical panel), so the figure set regenerates
exactly with ``python -m stgreen.viz.figures``. Output is written to
``outputs/figures/`` as both vector PDF (for submission) and 300-dpi PNG (preview),
within the journal's ten-figure budget.

Design choices: a colour-blind-safe palette (Okabe & Ito, 2008), minimal chart
"ink", consistent borough colours across every panel, and no author-identifying
content (double-anonymised review).

Figures
-------
1. study_area          -- density of street-tree locations (2015), study context
2. structural_trajectories -- six structural indicators by borough across epochs
3. size_class_shift    -- city-wide DBH size-class distribution by epoch
4. carbon_trajectories -- carbon storage and carbon-per-tree by borough
5. did_forest          -- standardized DID effects with 95% CIs across outcomes
6. treated_vs_control  -- distributions of 2005->2015 change by treatment status
7. moran_scatter       -- spatial autocorrelation of carbon change (Moran plot)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless, deterministic rasterisation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stgreen import config as cfg
from stgreen.metrics.structure import STRUCTURE_PANEL_TEMPLATE
from stgreen.metrics.ecoservices import ECOSERVICES_PANEL_TEMPLATE
from stgreen.change.change_metrics import ANALYSIS_PANEL_TEMPLATE, CHANGE_TABLE_TEMPLATE
from stgreen.analysis.did import build_did_frame, run_did_suite

logger = logging.getLogger(__name__)

# Okabe-Ito colour-blind-safe qualitative palette.
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442"]
BOROUGHS = ["Manhattan", "Bronx", "Brooklyn", "Queens", "Staten Island"]
BOROUGH_COLOR = dict(zip(BOROUGHS, OKABE_ITO))

_RC = {
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 1.8,
    "lines.markersize": 5,
}


def _save(fig: plt.Figure, name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)
    logger.info("Wrote %s.pdf / .png", name)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_study_area(panel: pd.DataFrame, out_dir: Path) -> None:
    d = panel[(panel["source_year"] == 2015) & (panel["coord_valid"].eq(True))]
    fig, ax = plt.subplots(figsize=(5.2, 5.4))
    hb = ax.hexbin(d["lon"], d["lat"], gridsize=60, cmap="viridis", mincnt=1, bins="log")
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Street-tree density (2015 census)")
    cb = fig.colorbar(hb, ax=ax, shrink=0.8); cb.set_label("Trees per cell (log)")
    _save(fig, "fig1_study_area", out_dir)


def fig_structural_trajectories(structure_borough: pd.DataFrame, out_dir: Path) -> None:
    specs = [
        ("n_alive", "Living stems"), ("basal_area_m2", "Basal area (m$^2$)"),
        ("mean_dbh_cm", "Mean DBH (cm)"), ("large_dbh_share", "Large-tree share"),
        ("richness", "Species richness"), ("shannon", "Shannon $H'$"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.6))
    for ax, (col, label) in zip(axes.ravel(), specs):
        for b in BOROUGHS:
            s = structure_borough[structure_borough["unit_id"] == b].sort_values("source_year")
            ax.plot(s["source_year"], s[col], marker="o", color=BOROUGH_COLOR[b], label=b)
        ax.set_title(label); ax.set_xticks([1995, 2005, 2015])
    axes[0, 0].legend(ncol=1, loc="upper left")
    fig.suptitle("Structural trajectories of street greenery by borough, 1995–2015", y=1.0)
    fig.tight_layout()
    _save(fig, "fig2_structural_trajectories", out_dir)


def fig_size_class_shift(structure_borough: pd.DataFrame, out_dir: Path) -> None:
    labels = list(cfg.DBH_SIZE_CLASS_LABELS)
    pretty = ["0–15", "15–30", "30–46", "46–61", "61–76", "76+ cm"]
    # City-wide shares = size-class counts (share x measured) summed, re-normalised.
    rows = []
    for yr, g in structure_borough.groupby("source_year"):
        counts = (g[labels].to_numpy() * g["n_dbh_measured"].to_numpy()[:, None]).sum(axis=0)
        rows.append(pd.Series(counts / counts.sum(), index=labels, name=yr))
    shares = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    bottom = np.zeros(len(shares))
    for lab, pl, color in zip(labels, pretty, OKABE_ITO):
        ax.bar(shares.index.astype(str), shares[lab], bottom=bottom, label=pl, color=color)
        bottom += shares[lab].to_numpy()
    ax.set_ylabel("Share of measured living trees"); ax.set_xlabel("Census year")
    ax.set_title("City-wide DBH size-class distribution by epoch")
    ax.legend(title="DBH class", ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    ax.grid(axis="x", visible=False)
    _save(fig, "fig3_size_class_shift", out_dir)


def fig_carbon_trajectories(eco_borough: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9))
    for b in BOROUGHS:
        s = eco_borough[eco_borough["unit_id"] == b].sort_values("source_year")
        axes[0].plot(s["source_year"], s["carbon_storage_Mg"], marker="o", color=BOROUGH_COLOR[b], label=b)
        axes[1].plot(s["source_year"], s["carbon_per_tree_kg"], marker="o", color=BOROUGH_COLOR[b], label=b)
    axes[0].set_title("Stored carbon (Mg C)"); axes[1].set_title("Carbon per living tree (kg C)")
    for ax in axes:
        ax.set_xticks([1995, 2005, 2015]); ax.set_xlabel("Census year")
    axes[0].legend(loc="upper left")
    fig.suptitle("Carbon ecosystem services by borough, 1995–2015", y=1.02)
    fig.tight_layout()
    _save(fig, "fig4_carbon_trajectories", out_dir)


def fig_did_forest(analysis_panel: pd.DataFrame, change_table: pd.DataFrame, out_dir: Path) -> None:
    suite = run_did_suite(analysis_panel, change_table)
    # Standardize each ATT by the SD of that outcome's change, for comparability.
    period = "2005_2015"
    ct = change_table[change_table["period"] == period]
    sd = {o: ct[f"delta_{o}"].std(ddof=1) for o in suite["outcome"]}
    suite = suite.assign(
        z_att=[r.att / sd[r.outcome] if sd[r.outcome] else np.nan for r in suite.itertuples()],
        z_lo=[r.ci_low / sd[r.outcome] if sd[r.outcome] else np.nan for r in suite.itertuples()],
        z_hi=[r.ci_high / sd[r.outcome] if sd[r.outcome] else np.nan for r in suite.itertuples()],
    ).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    y = np.arange(len(suite))
    ax.errorbar(suite["z_att"], y,
                xerr=[suite["z_att"] - suite["z_lo"], suite["z_hi"] - suite["z_att"]],
                fmt="o", color=OKABE_ITO[0], ecolor="gray", capsize=3)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_yticks(y); ax.set_yticklabels(suite["outcome"])
    ax.set_xlabel("Standardized DID effect (ATT / SD of change), 95% CI")
    ax.set_title("Estimated urban-renewal effect across outcomes")
    _save(fig, "fig5_did_forest", out_dir)


def fig_treated_vs_control(change_table: pd.DataFrame, out_dir: Path) -> None:
    sub = change_table[change_table["period"] == "2005_2015"].copy()
    sub["grp"] = np.where(sub["treated"].astype("boolean").fillna(False), "Treated", "Control")
    specs = [("growth_carbon_storage_Mg", "Carbon growth"),
             ("growth_basal_area_m2", "Basal-area growth"),
             ("delta_shannon", "Δ Shannon"), ("turnover", "Comp. turnover")]
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 3.4))
    for ax, (col, label) in zip(axes, specs):
        data = [sub.loc[sub["grp"] == g, col].dropna() for g in ("Control", "Treated")]
        bp = ax.boxplot(data, tick_labels=["Control", "Treated"], patch_artist=True, widths=0.6)
        for patch, c in zip(bp["boxes"], [OKABE_ITO[5], OKABE_ITO[1]]):
            patch.set_facecolor(c); patch.set_alpha(0.7)
        for med in bp["medians"]:
            med.set_color("black")
        ax.set_title(label)
    fig.suptitle("Change 2005–2015 by renewal status", y=1.03)
    fig.tight_layout()
    _save(fig, "fig6_treated_vs_control", out_dir)


def fig_moran_scatter(analysis_panel: pd.DataFrame, change_table: pd.DataFrame,
                      panel: pd.DataFrame, out_dir: Path, outcome: str = "carbon_storage_Mg") -> None:
    # Local import keeps the PySAL dependency optional for the other figures.
    from stgreen.analysis.spatial import prepare, morans_i
    frame = build_did_frame(analysis_panel, change_table)
    f, w = prepare(frame, panel, cfg.DEFAULT_SPATIAL_UNIT, outcome)
    y = f[f"delta_{outcome}"].to_numpy(dtype="float64")
    z = (y - y.mean()) / y.std()
    wz = np.asarray(w.sparse @ z).flatten()
    stats = morans_i(y, w)
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.scatter(z, wz, s=14, color=OKABE_ITO[0], alpha=0.6, edgecolor="none")
    b = np.polyfit(z, wz, 1)[0]
    xs = np.linspace(z.min(), z.max(), 50)
    ax.plot(xs, b * xs, color=OKABE_ITO[3], lw=1.6)
    ax.axhline(0, color="gray", lw=0.6); ax.axvline(0, color="gray", lw=0.6)
    ax.set_xlabel(f"Standardized Δ {outcome}"); ax.set_ylabel("Spatial lag")
    ax.set_title(f"Moran scatter: I = {stats['morans_I']:.3f} (p = {stats['p_sim']:.3f})")
    _save(fig, "fig7_moran_scatter", out_dir)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def generate_all(out_dir: Path = cfg.FIGURES_DIR) -> None:
    pdir = cfg.PROCESSED_DIR
    structure_b = pd.read_parquet(pdir / STRUCTURE_PANEL_TEMPLATE.format(unit="borough"))
    eco_b = pd.read_parquet(pdir / ECOSERVICES_PANEL_TEMPLATE.format(unit="borough"))
    analysis = pd.read_parquet(pdir / ANALYSIS_PANEL_TEMPLATE.format(unit="nta"))
    change = pd.read_parquet(pdir / CHANGE_TABLE_TEMPLATE.format(unit="nta"))
    canon = pd.read_parquet(
        cfg.CANONICAL_PANEL_PARQUET,
        columns=["source_year", "lon", "lat", "coord_valid", "nta_code"],
    )
    with plt.rc_context(_RC):
        fig_study_area(canon, out_dir)
        fig_structural_trajectories(structure_b, out_dir)
        fig_size_class_shift(structure_b, out_dir)
        fig_carbon_trajectories(eco_b, out_dir)
        fig_did_forest(analysis, change, out_dir)
        fig_treated_vs_control(change, out_dir)
        fig_moran_scatter(analysis, change, canon, out_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=cfg.FIGURES_DIR)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    generate_all(args.out)
    logger.info("All figures written to %s", args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
