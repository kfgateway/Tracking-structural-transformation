"""Difference-in-differences estimation of the urban-renewal effect.

The causal claim in the title -- that street-greenery structure changed *because
of* urban renewal -- is tested here. With three census epochs and a treatment
that takes effect between them, the design is a clean quasi-experiment:

* **Treated**: units whose renewal took effect between the pre and post census.
* **Control**: units never renewed over the study window.
* **Outcome**: the between-epoch change in a structural / carbon indicator.

Three complementary estimators are provided:

1. :func:`estimate_did` -- the first-difference DID (regress the post--pre change
   on the treated indicator, optionally covariate-adjusted), the headline ATT,
   with heteroskedasticity-robust SEs.
2. :func:`estimate_twfe` -- a two-way fixed-effects robustness check on the long
   two-period panel, with unit-clustered SEs.
3. :func:`parallel_trends_test` -- the identifying assumption's falsification
   test, using the *pre-treatment* 1995->2005 change (should show no treated
   effect). The three epochs make this check possible, which is a core strength.

:func:`estimate_matched_did` runs the first-difference DID on a propensity-score
matched sample (see :mod:`stgreen.analysis.matching`).

All estimators return a tidy dict/DataFrame; nothing is printed except by the CLI.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from stgreen import config as cfg
from stgreen.analysis import matching as M
from stgreen.change.change_metrics import (
    ANALYSIS_PANEL_TEMPLATE, CHANGE_TABLE_TEMPLATE, OUTCOME_COLS,
)

logger = logging.getLogger(__name__)

DEFAULT_PRE, DEFAULT_POST = 2005, 2015


# --------------------------------------------------------------------------- #
# Data assembly
# --------------------------------------------------------------------------- #
def build_did_frame(
    analysis_panel: pd.DataFrame,
    change_table: pd.DataFrame,
    pre: int = DEFAULT_PRE,
    post: int = DEFAULT_POST,
    covariates: tuple[str, ...] = M.DEFAULT_COVARIATES,
) -> pd.DataFrame:
    """One row per unit: post--pre change outcomes + baseline covariates + treated.

    Baseline covariates are the *pre-period* levels, prefixed ``base_`` so they
    never collide with the ``delta_`` / ``growth_`` outcome columns.
    """
    period = f"{pre}_{post}"
    ct = change_table[change_table["period"] == period].copy()
    if ct.empty:
        raise ValueError(f"No change rows for period {period!r}.")
    ct["treated"] = ct["treated"].astype("boolean").fillna(False).astype(int)

    base_cols = [c for c in covariates if c in analysis_panel.columns]
    base = (
        analysis_panel[analysis_panel["source_year"] == pre][["unit_id", *base_cols]]
        .rename(columns={c: f"base_{c}" for c in base_cols})
    )
    frame = ct.merge(base, on="unit_id", how="left")
    return frame


# --------------------------------------------------------------------------- #
# Estimators
# --------------------------------------------------------------------------- #
def _tidy(res, term: str, extra: dict) -> dict:
    ci = res.conf_int().loc[term]
    return {
        **extra,
        "att": round(float(res.params[term]), 5),
        "se": round(float(res.bse[term]), 5),
        "t": round(float(res.tvalues[term]), 3),
        "p_value": round(float(res.pvalues[term]), 5),
        "ci_low": round(float(ci[0]), 5),
        "ci_high": round(float(ci[1]), 5),
        "n_obs": int(res.nobs),
    }


def estimate_did(
    did_frame: pd.DataFrame,
    outcome: str,
    adjusted: bool = True,
    covariates: tuple[str, ...] = M.DEFAULT_COVARIATES,
) -> dict:
    """First-difference DID: regress the post--pre change on ``treated``.

    With one observation per unit, the treated coefficient is the ATT under
    (conditional) parallel trends. SEs are heteroskedasticity-robust (HC1).
    """
    dep = f"delta_{outcome}"
    if dep not in did_frame.columns:
        raise ValueError(f"Outcome {dep!r} not in frame.")
    rhs = ["treated"]
    if adjusted:
        rhs += [f"base_{c}" for c in covariates if f"base_{c}" in did_frame.columns]
    data = did_frame.dropna(subset=[dep, *rhs]).copy()
    res = smf.ols(f"{dep} ~ " + " + ".join(rhs), data=data).fit(cov_type="HC1")
    return _tidy(res, "treated", {
        "outcome": outcome, "estimator": "first_diff" + ("_adj" if adjusted else ""),
        "n_treated": int(data["treated"].sum()),
    })


def estimate_twfe(
    analysis_panel: pd.DataFrame,
    outcome: str,
    pre: int = DEFAULT_PRE,
    post: int = DEFAULT_POST,
) -> dict:
    """Two-way fixed-effects DID on the long two-period panel (robustness check).

    ``outcome ~ C(unit_id) + C(source_year) + did`` with ``did = treated * post``;
    SEs clustered by unit. The ``did`` coefficient is the DID estimate.
    """
    data = analysis_panel[analysis_panel["source_year"].isin([pre, post])].copy()
    data = data.dropna(subset=[outcome])
    data["post"] = (data["source_year"] == post).astype(int)
    data["treated_i"] = data["treated"].astype("boolean").fillna(False).astype(int)
    data["did"] = data["treated_i"] * data["post"]
    res = smf.ols(
        f"{outcome} ~ C(unit_id) + C(source_year) + did", data=data
    ).fit(cov_type="cluster", cov_kwds={"groups": data["unit_id"]})
    return _tidy(res, "did", {"outcome": outcome, "estimator": "twfe",
                              "n_treated": int(data.loc[data["post"].eq(1), "treated_i"].sum())})


def parallel_trends_test(
    change_table: pd.DataFrame,
    outcome: str,
    pre_period: str = "1995_2005",
) -> dict:
    """Falsification test: treated effect on the PRE-treatment change.

    Regresses the pre-period change on ``treated``; a coefficient near zero
    (large p-value) supports the parallel-trends assumption.
    """
    dep = f"delta_{outcome}"
    data = change_table[change_table["period"] == pre_period].copy()
    data["treated"] = data["treated"].astype("boolean").fillna(False).astype(int)
    data = data.dropna(subset=[dep])
    res = smf.ols(f"{dep} ~ treated", data=data).fit(cov_type="HC1")
    out = _tidy(res, "treated", {"outcome": outcome, "estimator": "pre_trend", "period": pre_period})
    out["parallel_trends_ok"] = bool(out["p_value"] > 0.10)  # fail to reject -> supportive
    return out


def estimate_matched_did(
    analysis_panel: pd.DataFrame,
    did_frame: pd.DataFrame,
    outcome: str,
    pre: int = DEFAULT_PRE,
    covariates: tuple[str, ...] = M.DEFAULT_COVARIATES,
) -> tuple[dict, pd.DataFrame]:
    """First-difference DID on a propensity-score matched sample."""
    baseline = analysis_panel[analysis_panel["source_year"] == pre].copy()
    matched, balance = M.match_summary(baseline, "treated", covariates)
    if matched.empty:
        logger.warning("Matching produced no pairs; returning unmatched estimate.")
        return estimate_did(did_frame, outcome, adjusted=False), balance
    sub = did_frame[did_frame["unit_id"].isin(matched["unit_id"])]
    res = estimate_did(sub, outcome, adjusted=False)
    res["estimator"] = "first_diff_matched"
    return res, balance


def run_did_suite(
    analysis_panel: pd.DataFrame,
    change_table: pd.DataFrame,
    outcomes: tuple[str, ...] = OUTCOME_COLS,
    pre: int = DEFAULT_PRE,
    post: int = DEFAULT_POST,
) -> pd.DataFrame:
    """Run the headline first-difference DID across all outcomes -> tidy table."""
    frame = build_did_frame(analysis_panel, change_table, pre, post)
    rows = []
    for o in outcomes:
        if f"delta_{o}" in frame.columns:
            rows.append(estimate_did(frame, o, adjusted=True))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--unit", default=cfg.DEFAULT_SPATIAL_UNIT, choices=list(cfg.SPATIAL_UNITS))
    parser.add_argument("--pre", type=int, default=DEFAULT_PRE)
    parser.add_argument("--post", type=int, default=DEFAULT_POST)
    parser.add_argument("--outcome", default="carbon_storage_Mg",
                        help="Primary outcome for the detailed/robustness printout.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    ap = pd.read_parquet(cfg.PROCESSED_DIR / ANALYSIS_PANEL_TEMPLATE.format(unit=args.unit))
    ct = pd.read_parquet(cfg.PROCESSED_DIR / CHANGE_TABLE_TEMPLATE.format(unit=args.unit))
    frame = build_did_frame(ap, ct, args.pre, args.post)

    suite = run_did_suite(ap, ct, pre=args.pre, post=args.post)
    with pd.option_context("display.width", 160, "display.max_columns", None):
        logger.info("First-difference DID (adjusted) across outcomes:\n%s", suite.to_string(index=False))

    o = args.outcome
    logger.info("Robustness for %s:", o)
    logger.info("  TWFE:        %s", estimate_twfe(ap, o, args.pre, args.post))
    logger.info("  Parallel-trends (pre): %s", parallel_trends_test(ct, o))
    matched, balance = estimate_matched_did(ap, frame, o, args.pre)
    logger.info("  Matched DID: %s", matched)
    with pd.option_context("display.width", 120):
        logger.info("  Covariate balance (SMD before/after):\n%s", balance.to_string())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
