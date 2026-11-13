"""Propensity-score matching of treated and control spatial units.

Urban-renewal "treatment" is not randomly assigned: renewed neighbourhoods may
differ systematically in their baseline tree stock. To make the treated and
control groups comparable before estimating the renewal effect, this module
estimates each unit's propensity to be treated from *baseline* (pre-period)
characteristics and matches treated units to similar controls.

The output (a matched subset and balance diagnostics) feeds the matched
difference-in-differences in :mod:`stgreen.analysis.did`.

Matching is on the logit of the propensity score, 1:1, greedy, without
replacement, within a caliper (default 0.2 SD of the logit score; Austin, 2011).
Balance is reported as the standardized mean difference (SMD) per covariate; an
|SMD| below 0.1 after matching is the usual adequacy threshold.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

DEFAULT_COVARIATES: tuple[str, ...] = (
    "n_alive", "basal_area_m2", "mean_dbh_cm", "large_dbh_share",
    "richness", "shannon", "carbon_storage_Mg",
)


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def estimate_propensity(
    df: pd.DataFrame,
    treated_col: str = "treated",
    covariates: tuple[str, ...] = DEFAULT_COVARIATES,
) -> pd.Series:
    """Estimate P(treated=1 | baseline covariates) by standardized logistic reg.

    Returns a propensity Series aligned to ``df``'s index. Rows with missing
    covariates get ``NaN`` and are excluded from matching downstream.
    """
    cov = [c for c in covariates if c in df.columns]
    if not cov:
        raise ValueError("None of the requested covariates are present.")
    X = df[cov].astype("float64")
    y = df[treated_col].astype("boolean").fillna(False).astype(int)

    ok = X.notna().all(axis=1)
    ps = pd.Series(np.nan, index=df.index, dtype="float64")
    if ok.sum() == 0 or y[ok].nunique() < 2:
        logger.warning("Cannot estimate propensity (no variation / no complete rows).")
        return ps

    Xs = StandardScaler().fit_transform(X[ok])
    model = LogisticRegression(max_iter=1000, C=1e6)  # near-unpenalised
    model.fit(Xs, y[ok])
    ps.loc[ok] = model.predict_proba(Xs)[:, 1]
    return ps


def nearest_neighbor_match(
    df: pd.DataFrame,
    propensity: pd.Series,
    treated_col: str = "treated",
    caliper_sd: float = 0.2,
) -> pd.DataFrame:
    """Greedy 1:1 matching on the logit propensity, without replacement.

    Returns a copy of the matched rows (treated + their matched controls) with
    added columns ``match_id`` (pairs a treated unit with its control) and
    ``ps`` (propensity). Unmatched units are omitted.
    """
    work = df.copy()
    work["ps"] = propensity
    work = work[work["ps"].notna()]
    treated = work[work[treated_col].eq(True)].copy()
    control = work[work[treated_col].eq(False)].copy()
    if treated.empty or control.empty:
        logger.warning("No treated or no control units with valid propensity.")
        return work.iloc[0:0]

    lp = _logit(work["ps"].to_numpy())
    caliper = caliper_sd * np.nanstd(lp)
    t_lp = _logit(treated["ps"].to_numpy())
    c_lp = _logit(control["ps"].to_numpy())
    c_idx = list(control.index)
    c_used: set = set()

    pairs = []
    # Match treated units in ascending propensity order for stability.
    order = np.argsort(treated["ps"].to_numpy())
    for mi, ti in enumerate(order):
        t_index = treated.index[ti]
        best_j, best_d = None, np.inf
        for j, ci in enumerate(c_idx):
            if ci in c_used:
                continue
            d = abs(t_lp[ti] - c_lp[j])
            if d < best_d:
                best_d, best_j = d, (j, ci)
        if best_j is not None and best_d <= caliper:
            j, ci = best_j
            c_used.add(ci)
            pairs.append((t_index, ci, mi))

    if not pairs:
        logger.warning("No matches within caliper %.3f.", caliper)
        return work.iloc[0:0]

    rows = []
    for t_index, ci, mid in pairs:
        tr = work.loc[t_index].copy(); tr["match_id"] = mid
        co = work.loc[ci].copy(); co["match_id"] = mid
        rows.extend([tr, co])
    matched = pd.DataFrame(rows)
    logger.info("Matched %d treated units to controls (caliper=%.3f).", len(pairs), caliper)
    return matched


def standardized_mean_differences(
    df: pd.DataFrame,
    treated_col: str = "treated",
    covariates: tuple[str, ...] = DEFAULT_COVARIATES,
) -> pd.DataFrame:
    """Per-covariate standardized mean difference between treated and control."""
    cov = [c for c in covariates if c in df.columns]
    t = df[df[treated_col].eq(True)]
    c = df[df[treated_col].eq(False)]
    out = {}
    for col in cov:
        mt, mc = t[col].mean(), c[col].mean()
        vt, vc = t[col].var(ddof=1), c[col].var(ddof=1)
        pooled = np.sqrt((vt + vc) / 2.0)
        out[col] = np.nan if pooled == 0 else (mt - mc) / pooled
    return pd.DataFrame({"smd": pd.Series(out)}).round(4)


def match_summary(
    baseline: pd.DataFrame,
    treated_col: str = "treated",
    covariates: tuple[str, ...] = DEFAULT_COVARIATES,
    caliper_sd: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper: match on baseline and return (matched_df, balance).

    ``balance`` has SMDs before and after matching for every covariate.
    """
    ps = estimate_propensity(baseline, treated_col, covariates)
    matched = nearest_neighbor_match(baseline, ps, treated_col, caliper_sd)
    before = standardized_mean_differences(baseline, treated_col, covariates).rename(columns={"smd": "smd_before"})
    after = (
        standardized_mean_differences(matched, treated_col, covariates).rename(columns={"smd": "smd_after"})
        if not matched.empty else pd.DataFrame(index=before.index)
    )
    balance = before.join(after)
    return matched, balance
