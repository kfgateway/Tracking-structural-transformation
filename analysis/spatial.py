"""Spatial-dependence diagnostics and robustness for the renewal effect.

Neighbourhoods are not independent: treated and control units cluster in space,
so (a) ordinary standard errors understate uncertainty, (b) unmodelled spatial
autocorrelation can bias the estimate, and (c) renewal in one unit may spill over
into its neighbours. A spatial-ecology-literate journal expects these addressed.

This module provides:

* :func:`unit_centroids` -- unit locations from the mean of their coordinate-valid
  street-tree points (so it needs no extra boundary file), or from supplied
  polygon geometry.
* :func:`build_weights` -- a row-standardised k-nearest-neighbour spatial weights
  matrix over the units.
* :func:`morans_i` / :func:`moran_residuals` -- Moran's I on an outcome and on the
  DID residuals (residual autocorrelation is the real warning sign).
* :func:`spatial_did` -- the first-difference DID re-estimated as a spatial-lag or
  spatial-error model (maximum likelihood), so the renewal coefficient is robust
  to spatial dependence.
* :func:`spillover_did` -- adds the spatially-lagged treatment (``W*treated``) to
  separate a unit's own renewal effect from its neighbours'.

All estimators align the data array and the weights matrix to the same ordered
set of units, dropping units without a centroid or with missing model inputs.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from libpysal.weights import KNN
from esda.moran import Moran
from spreg import OLS, ML_Lag, ML_Error

from stgreen import config as cfg
from stgreen.analysis import matching as M
from stgreen.change.change_metrics import ANALYSIS_PANEL_TEMPLATE, CHANGE_TABLE_TEMPLATE
from stgreen.analysis.did import build_did_frame

logger = logging.getLogger(__name__)

DEFAULT_K = 6


# --------------------------------------------------------------------------- #
# Geometry and weights
# --------------------------------------------------------------------------- #
def unit_centroids(panel: pd.DataFrame, unit: str) -> pd.DataFrame:
    """Unit centroids (lon/lat) from the mean of coordinate-valid tree points."""
    unit_cols = list(cfg.SPATIAL_UNITS[unit])
    pts = panel[panel["coord_valid"].eq(True)].dropna(subset=unit_cols)
    grp = pts.groupby(unit_cols, observed=True)[["lon", "lat"]].mean().reset_index()
    if len(unit_cols) == 1:
        grp["unit_id"] = grp[unit_cols[0]].astype("string")
    else:
        grp["unit_id"] = grp[unit_cols].astype("string").agg("|".join, axis=1)
    return grp[["unit_id", "lon", "lat"]]


def _align(frame: pd.DataFrame, centroids: pd.DataFrame, needed: list[str]):
    """Intersect frame and centroids, drop missing model inputs, fix one order.

    Returns (ordered_frame, coords_array) sharing row order, so a weights matrix
    built from ``coords`` lines up with the data.
    """
    f = frame.dropna(subset=needed).merge(centroids, on="unit_id", how="inner")
    f = f.dropna(subset=["lon", "lat"]).reset_index(drop=True)
    coords = f[["lon", "lat"]].to_numpy()
    return f, coords


def build_weights(coords: np.ndarray, k: int = DEFAULT_K):
    """Row-standardised k-nearest-neighbour spatial weights from point coords."""
    w = KNN.from_array(coords, k=k)
    w.transform = "r"
    return w


# --------------------------------------------------------------------------- #
# Autocorrelation
# --------------------------------------------------------------------------- #
def morans_i(values: np.ndarray, w, permutations: int = 999) -> dict:
    """Moran's I of a value vector under ``w`` (permutation inference)."""
    mi = Moran(np.asarray(values, dtype="float64"), w, permutations=permutations)
    return {"morans_I": round(float(mi.I), 4), "expected_I": round(float(mi.EI), 4),
            "p_sim": round(float(mi.p_sim), 4), "z_sim": round(float(mi.z_sim), 3)}


def moran_residuals(frame: pd.DataFrame, outcome: str, w,
                    covariates: tuple[str, ...] = M.DEFAULT_COVARIATES) -> dict:
    """Moran's I on the residuals of the (aspatial) first-difference DID."""
    dep = f"delta_{outcome}"
    xcols = ["treated"] + [f"base_{c}" for c in covariates if f"base_{c}" in frame.columns]
    y = frame[[dep]].to_numpy(dtype="float64")
    X = frame[xcols].to_numpy(dtype="float64")
    ols = OLS(y, X, name_y=dep, name_x=xcols)
    out = morans_i(ols.u.flatten(), w)
    out["on"] = "did_residuals"
    return out


# --------------------------------------------------------------------------- #
# Spatial DID
# --------------------------------------------------------------------------- #
def _coef(model, term: str, names: list[str]) -> dict:
    betas = np.asarray(model.betas).flatten()
    se = np.asarray(model.std_err).flatten()
    # ML models (lag/error) expose z_stat; OLS exposes t_stat. Both are lists of
    # (stat, p) aligned with [CONSTANT, *names].
    zp = getattr(model, "z_stat", None) or model.t_stat
    full = ["CONSTANT", *names]
    i = full.index(term)
    return {"att": round(float(betas[i]), 5), "se": round(float(se[i]), 5),
            "z": round(float(zp[i][0]), 3), "p_value": round(float(zp[i][1]), 5)}


def spatial_did(frame: pd.DataFrame, outcome: str, w, model: str = "lag",
                covariates: tuple[str, ...] = M.DEFAULT_COVARIATES) -> dict:
    """First-difference DID as a spatial-lag or spatial-error ML model.

    The ``treated`` coefficient is the renewal ATT net of spatial dependence;
    ``rho`` (lag) or ``lambda`` (error) measures that dependence.
    """
    dep = f"delta_{outcome}"
    xcols = ["treated"] + [f"base_{c}" for c in covariates if f"base_{c}" in frame.columns]
    y = frame[[dep]].to_numpy(dtype="float64")
    X = frame[xcols].to_numpy(dtype="float64")

    if model == "lag":
        res = ML_Lag(y, X, w=w, name_y=dep, name_x=xcols)
        out = _coef(res, "treated", xcols)
        out["estimator"], out["rho"] = "spatial_lag", round(float(res.rho), 5)
    elif model == "error":
        res = ML_Error(y, X, w=w, name_y=dep, name_x=xcols)
        out = _coef(res, "treated", xcols)
        out["estimator"], out["lambda"] = "spatial_error", round(float(res.lam), 5)
    else:  # pragma: no cover
        raise ValueError("model must be 'lag' or 'error'")
    out["outcome"], out["n_obs"] = outcome, int(frame.shape[0])
    return out


def spillover_did(frame: pd.DataFrame, outcome: str, w,
                  covariates: tuple[str, ...] = M.DEFAULT_COVARIATES) -> dict:
    """Separate direct renewal effect from neighbours' (spatially-lagged treatment).

    Adds ``W_treated`` (the row-standardised spatial lag of the treatment) to the
    aspatial first-difference DID. Its coefficient is the spillover.
    """
    dep = f"delta_{outcome}"
    base_x = ["treated"] + [f"base_{c}" for c in covariates if f"base_{c}" in frame.columns]
    w_treated = np.asarray(w.sparse @ frame["treated"].to_numpy(dtype="float64")).flatten()
    f = frame.copy()
    f["W_treated"] = w_treated
    xcols = base_x + ["W_treated"]
    y = f[[dep]].to_numpy(dtype="float64")
    X = f[xcols].to_numpy(dtype="float64")
    ols = OLS(y, X, w=w, spat_diag=True, name_y=dep, name_x=xcols)
    direct = _coef(ols, "treated", xcols)
    spill = _coef(ols, "W_treated", xcols)
    return {"outcome": outcome, "estimator": "spillover_ols",
            "direct_att": direct["att"], "direct_p": direct["p_value"],
            "spillover": spill["att"], "spillover_p": spill["p_value"],
            "n_obs": int(f.shape[0])}


def prepare(frame: pd.DataFrame, panel: pd.DataFrame, unit: str, outcome: str,
            covariates: tuple[str, ...] = M.DEFAULT_COVARIATES, k: int = DEFAULT_K):
    """Align a DID frame with centroids and build the weights matrix."""
    needed = [f"delta_{outcome}", "treated"] + [
        f"base_{c}" for c in covariates if f"base_{c}" in frame.columns
    ]
    cents = unit_centroids(panel, unit)
    f, coords = _align(frame, cents, needed)
    w = build_weights(coords, k=k)
    return f, w


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--unit", default=cfg.DEFAULT_SPATIAL_UNIT, choices=list(cfg.SPATIAL_UNITS))
    parser.add_argument("--outcome", default="carbon_storage_Mg")
    parser.add_argument("--pre", type=int, default=2005)
    parser.add_argument("--post", type=int, default=2015)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    ap = pd.read_parquet(cfg.PROCESSED_DIR / ANALYSIS_PANEL_TEMPLATE.format(unit=args.unit))
    ct = pd.read_parquet(cfg.PROCESSED_DIR / CHANGE_TABLE_TEMPLATE.format(unit=args.unit))
    panel = pd.read_parquet(
        cfg.CANONICAL_PANEL_PARQUET,
        columns=["coord_valid", "lon", "lat", *cfg.SPATIAL_UNITS[args.unit]],
    )
    frame = build_did_frame(ap, ct, args.pre, args.post)
    f, w = prepare(frame, panel, args.unit, args.outcome, k=args.k)
    logger.info("Spatial sample: %d units, KNN k=%d", f.shape[0], args.k)

    o = args.outcome
    logger.info("Moran's I on delta_%s: %s", o, morans_i(f[f"delta_{o}"].to_numpy(), w))
    logger.info("Moran's I on DID residuals: %s", moran_residuals(f, o, w))
    logger.info("Spatial-lag DID:   %s", spatial_did(f, o, w, model="lag"))
    logger.info("Spatial-error DID: %s", spatial_did(f, o, w, model="error"))
    logger.info("Spillover DID:     %s", spillover_did(f, o, w))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
