"""Tests for the census harmoniser: the data-quality traps that matter most."""

from __future__ import annotations

import pandas as pd
import pytest

from stgreen.data import census


def _write(tmp_path, name, df):
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    df.to_csv(raw / name, index=False)


@pytest.fixture
def mini_raw(tmp_path):
    # 1995: Planting Space (no tree), Shaft (stump), a clean tree, a bad coord.
    _write(tmp_path, "1995_Street_Tree_Census.csv", pd.DataFrame({
        "RecordId": [1, 2, 3, 4],
        "Spc_Latin": ["PLATANUS ACERIFOLIA", "ACER PLATANOIDES", "QUERCUS PALUSTRIS", "TILIA CORDATA"],
        "Spc_Common": ["LONDON PLANETREE", "NORWAY MAPLE", "PIN OAK", "LITTLELEAF LINDEN"],
        "Diameter": [10, 8, 12, 6],
        "Condition": ["Good", "Shaft", "Planting Space", "Good"],
        "Longitude": [-73.98, -73.95, -73.90, 1.0],   # last is placeholder junk
        "Latitude": [40.73, 40.70, 40.68, 1.0],
        "Borough": ["Manhattan", "Brooklyn", "Queens", "Bronx"],
        "NTA_2010": ["MN21", "BK88", "QN17", "BX01"],
        "CensusTract_2010": ["48", "100", "200", "300"],
        "BBL": ["1", "2", "3", "4"],
    }))
    # 2005: Staten Island coded "5"; sentinel DBH -1; UNKNOWN species; a Dead tree.
    _write(tmp_path, "2005_Street_Tree_Census.csv", pd.DataFrame({
        "OBJECTID": [10, 11, 12],
        "spc_latin": ["PYRUS CALLERYANA", "UNKNOWN", "ACER PLATANOIDES"],
        "spc_common": ["CALLERY PEAR", "UNKNOWN", "NORWAY MAPLE"],
        "tree_dbh": [6, -1, 11],
        "status": ["Good", "Dead", "Excellent"],
        "longitude": [-74.10, -73.80, -73.95],
        "latitude": [40.60, 40.75, 40.70],
        "boroname": ["5", "Queens", "Brooklyn"],
        "nta": ["SI01", "QN17", "BK88"],
        "nta_name": ["SI nbhd", "QN nbhd", "BK nbhd"],
        "census tract": ["5", "6", "7"],
        "bbl": ["50", "60", "70"],
    }))
    # 2015: Alive w/ health, a Stump, a Dead; hybrid species name.
    _write(tmp_path, "2015_Street_Tree_Census_-_Tree_Data.csv", pd.DataFrame({
        "tree_id": [20, 21, 22],
        "spc_latin": ["Acer rubrum", "Platanus x acerifolia", None],
        "spc_common": ["red maple", "london planetree", None],
        "tree_dbh": [3, 20, 0],
        "status": ["Alive", "Alive", "Stump"],
        "health": ["Good", "Fair", None],
        "longitude": [-73.88, -73.92, -73.96],
        "latitude": [40.72, 40.74, 40.76],
        "borough": ["Queens", "Brooklyn", "Manhattan"],
        "nta": ["QN17", "BK88", "MN21"],
        "nta_name": ["QN nbhd", "BK nbhd", "MN nbhd"],
        "census tract": ["8", "9", "10"],
        "bbl": ["80", "90", "100"],
    }))
    return tmp_path


def test_species_key_cross_year_match():
    # Uppercase, hybrid and cultivar variants must collapse to one key.
    for raw in ["PLATANUS ACERIFOLIA", "Platanus x acerifolia", "Platanus  acerifolia"]:
        norm = census.normalize_species_name(raw)
        key, genus = census.species_key_and_genus(norm)
        assert key == "platanus acerifolia"
        assert genus == "Platanus"
    # Cultivar words are dropped from the key; genus-only for "SPECIES".
    assert census.species_key_and_genus(
        census.normalize_species_name("ACER PLATANOIDES CRIMSON KING"))[0] == "acer platanoides"
    assert census.species_key_and_genus(
        census.normalize_species_name("CRATAEGUS SPECIES"))[0] == "crataegus"


def test_binomial_casing():
    assert census.normalize_species_name("PLATANUS ACERIFOLIA") == "Platanus acerifolia"
    assert census.normalize_species_name("Acer rubrum") == "Acer rubrum"
    assert census.normalize_species_name("UNKNOWN") is None


def test_1995_condition_and_coords(mini_raw):
    df = census.load_census(1995, data_dir=mini_raw).set_index("orig_id")
    assert df.loc["3", "condition"] == "NoTree" and df.loc["3", "tree_present"] == False  # noqa: E712
    assert df.loc["2", "condition"] == "Stump"            # Shaft -> Stump
    assert df.loc["1", "dbh_cm"] == pytest.approx(25.4)   # 10 in -> cm
    assert df.loc["4", "coord_valid"] == False            # placeholder lon/lat  # noqa: E712
    assert df.loc["1", "coord_valid"] == True             # noqa: E712


def test_2005_borough_and_bad_dbh(mini_raw):
    df = census.load_census(2005, data_dir=mini_raw).set_index("orig_id")
    assert df.loc["10", "borough"] == "Staten Island"     # "5" -> name
    assert df.loc["11", "dbh_valid"] == False and pd.isna(df.loc["11", "dbh_cm"])  # noqa: E712
    assert df.loc["11", "condition"] == "Dead" and df.loc["11", "alive"] == False  # noqa: E712
    assert pd.isna(df.loc["11", "species_key"])           # UNKNOWN -> null


def test_2015_status_health(mini_raw):
    df = census.load_census(2015, data_dir=mini_raw).set_index("orig_id")
    assert df.loc["20", "alive"] == True and df.loc["20", "condition"] == "Good"   # noqa: E712
    assert df.loc["22", "alive"] == False and df.loc["22", "condition"] == "Stump"  # noqa: E712
    assert df.loc["21", "species_key"] == "platanus acerifolia"


def test_panel_key_unique(mini_raw):
    panel = census.build_panel(data_dir=mini_raw)
    assert not panel.duplicated(["source_year", "orig_id"]).any()
    assert list(panel.columns) == list(census.C.CANONICAL_COLUMNS)
