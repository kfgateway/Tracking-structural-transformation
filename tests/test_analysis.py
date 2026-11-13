"""Tests for renewal assignment, change metrics and the DID estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stgreen.data import renewal
from stgreen.change import change_metrics as change
from stgreen.analysis import did


def _points(lonlats):
    n = len(lonlats)
    return pd.DataFrame({
        "source_year": [2015] * n,
        "orig_id": [str(i) for i in range(n)],
        "coord_valid": [True] * n,
        "lon": [p[0] for p in lonlats],
        "lat": [p[1] for p in lonlats],
        "nta_code": [f"U{i}" for i in range(n)],
    })


def test_renewal_point_in_polygon():
    poly = renewal.make_synthetic_renewal()  # box ~(-73.99,40.66,-73.92,40.70) = 2008
    panel = _points([(-73.95, 40.68), (-74.25, 40.50)])  # inside, far outside
    trees = renewal.assign_tree_treatment(panel, poly)
    inside = trees[trees["lon"] == -73.95].iloc[0]
    outside = trees[trees["lon"] == -74.25].iloc[0]
    assert bool(inside["treated_tree"]) and int(inside["tree_treatment_year"]) == 2008
    assert not bool(outside["treated_tree"]) and pd.isna(outside["tree_treatment_year"])


def test_treatment_table_invariant():
    poly = renewal.make_synthetic_renewal()
    panel = _points([(-73.95, 40.68), (-74.25, 40.50)])
    tbl = renewal.build_treatment_table(panel, poly, unit="nta")
    # treated unit has a year; untreated does not.
    assert tbl.loc[tbl["treated"].eq(True), "first_treatment_year"].notna().all()
    assert tbl.loc[tbl["treated"].eq(False), "first_treatment_year"].isna().all()


def test_turnover_bounds():
    # Unit A: identical composition -> 0; Unit B: disjoint -> 1.
    panel = pd.DataFrame({
        "alive": [True] * 8,
        "genus": ["Quercus", "Quercus", "Quercus", "Quercus",   # A: oak both years
                  "Quercus", "Quercus", "Acer", "Acer"],          # B: oak -> maple
        "species_key": ["q"] * 8,
        "source_year": [1995, 1995, 2015, 2015, 1995, 1995, 2015, 2015],
        "nta_code": ["A", "A", "A", "A", "B", "B", "B", "B"],
    })
    t = change.compositional_turnover(panel, "nta", [(1995, 2015)], level="genus")
    t = t.set_index("nta_code")["turnover"]
    assert t.loc["A"] == pytest.approx(0.0)
    assert t.loc["B"] == pytest.approx(1.0)


def test_did_recovers_known_effect():
    # delta_y = 3 * treated exactly -> ATT must be 3.
    rng = np.random.default_rng(0)
    n = 200
    treated = rng.integers(0, 2, n)
    frame = pd.DataFrame({"unit_id": [str(i) for i in range(n)],
                          "treated": treated, "delta_y": 3.0 * treated})
    res = did.estimate_did(frame, "y", adjusted=False)
    assert res["att"] == pytest.approx(3.0, abs=1e-6)
    assert res["n_treated"] == int(treated.sum())
