"""Tests for structural indicators and carbon ecosystem services."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from stgreen import config as cfg
from stgreen.metrics import structure, ecoservices


def test_diversity_helpers():
    # Two equally-abundant species: H' = ln 2, Simpson D = 0.5.
    p = np.array([0.5, 0.5])
    assert structure.shannon_index(p) == pytest.approx(math.log(2))
    assert structure.simpson_D(p) == pytest.approx(0.5)
    assert math.isnan(structure.shannon_index(np.array([])))


def test_basal_area():
    # 100 cm DBH -> r=0.5 m -> area = pi*0.25 = 0.7854 m^2.
    ba = structure.basal_area_m2(pd.Series([100.0]))
    assert ba.iloc[0] == pytest.approx(math.pi * 0.25, rel=1e-6)


def _toy_panel():
    # One NTA, one epoch: 2 oaks + 2 maples, all alive, valid DBH 30 cm.
    n = 4
    return pd.DataFrame({
        "source_year": [2015] * n,
        "orig_id": [str(i) for i in range(n)],
        "tree_present": [True] * n,
        "alive": [True] * n,
        "dbh_valid": [True] * n,
        "dbh_cm": [30.0] * n,
        "condition": ["Good"] * n,
        "species_key": ["quercus palustris", "quercus palustris", "acer rubrum", "acer rubrum"],
        "genus": ["Quercus", "Quercus", "Acer", "Acer"],
        "nta_code": ["QN17"] * n,
    })


def test_structure_panel_diversity():
    out = structure.compute_structure_panel(_toy_panel(), unit="nta")
    row = out.iloc[0]
    assert row["richness"] == 2
    assert row["shannon"] == pytest.approx(math.log(2), rel=1e-3)
    assert row["evenness"] == pytest.approx(1.0, rel=1e-3)   # perfectly even
    assert row["n_alive"] == 4


def test_jenkins_group_assignment():
    assert ecoservices.assign_jenkins_group("acer rubrum", "acer") == "soft_maple_birch"
    assert ecoservices.assign_jenkins_group("quercus palustris", "quercus") == "hard_maple_oak_hickory_beech"
    assert ecoservices.assign_jenkins_group("acer platanoides", "acer") == "mixed_hardwood"
    assert ecoservices.assign_jenkins_group(None, None) == cfg.JENKINS_DEFAULT_GROUP


def test_carbon_hand_check():
    # Mixed hardwood @ 25 cm: independent recomputation of the documented pipeline.
    b0, b1 = cfg.JENKINS_COEFFS["mixed_hardwood"]
    agb = math.exp(b0 + b1 * math.log(25)) * cfg.URBAN_BIOMASS_ADJ
    carbon = agb * (1 + cfg.ROOT_SHOOT_RATIO) * cfg.CARBON_FRACTION
    panel = pd.DataFrame({
        "alive": [True], "dbh_valid": [True], "dbh_cm": [25.0],
        "species_key": ["tilia cordata"], "genus": ["Tilia"], "condition": ["Good"],
    })
    t = ecoservices.compute_tree_ecoservices(panel)
    assert t["carbon_kg"].iloc[0] == pytest.approx(carbon, rel=1e-3)
    assert t["co2e_kg"].iloc[0] == pytest.approx(carbon * cfg.CO2_PER_C, rel=1e-3)
