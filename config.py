"""Central configuration for the street-greenery / urban-renewal study.

This module is the single source of truth for every path, constant, controlled
vocabulary and harmonisation rule used across the code base. Nothing here
performs I/O or heavy computation: importing it must be cheap and side-effect
free so that it can be used by scripts, notebooks and tests alike.

Design notes
------------
* Trunk diameter (DBH) is stored natively in **inches** in the NYC street-tree
  censuses. Because *Urban Forestry & Urban Greening* requires SI units, the
  canonical schema carries ``dbh_cm`` (SI) while retaining ``dbh_in`` for
  provenance (see :data:`INCH_TO_CM`).
* The three censuses (1995, 2005, 2015) use *different* column names, case
  conventions and condition vocabularies. All year-specific knowledge is
  declared here as data (the ``*_COLUMNS`` / ``*_CONDITION`` maps) so that the
  loader in :mod:`stgreen.data.census` stays generic and auditable.
* Paths resolve relative to the repository root by default, but can be
  overridden with the ``STGREEN_DATA_DIR`` environment variable so the code
  runs unchanged on a reviewer's machine or in CI.

The module deliberately contains no author-identifying information, in line with
the journal's double-anonymised review policy.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# ``config.py`` lives at <root>/src/stgreen/config.py, so the project root is
# three levels up. This keeps path resolution independent of the caller's CWD.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

# ``STGREEN_DATA_DIR`` lets a user point the pipeline at data stored outside the
# repository (e.g. a mounted volume) without editing code.
DATA_DIR: Final[Path] = Path(
    os.environ.get("STGREEN_DATA_DIR", PROJECT_ROOT / "data")
).resolve()

RAW_DIR: Final[Path] = DATA_DIR / "raw"
INTERIM_DIR: Final[Path] = DATA_DIR / "interim"
PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"

# Rendered manuscript figures (vector PDF + 300-dpi PNG) and result tables.
FIGURES_DIR: Final[Path] = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR: Final[Path] = PROJECT_ROOT / "outputs" / "tables"

# Canonical output of the census harmoniser (one tidy row per tree record).
CANONICAL_PANEL_PARQUET: Final[Path] = PROCESSED_DIR / "nyc_tree_census_panel.parquet"

# --------------------------------------------------------------------------- #
# Census epochs and their raw source files
# --------------------------------------------------------------------------- #
# Keys are the canonical epoch years used throughout the study. Values are the
# raw file names as published by NYC Open Data (and as uploaded to data/raw/).
CENSUS_YEARS: Final[tuple[int, ...]] = (1995, 2005, 2015)

RAW_FILENAMES: Final[dict[int, str]] = {
    1995: "1995_Street_Tree_Census.csv",
    2005: "2005_Street_Tree_Census.csv",
    2015: "2015_Street_Tree_Census_-_Tree_Data.csv",
}

# --------------------------------------------------------------------------- #
# Units and coordinate reference systems
# --------------------------------------------------------------------------- #
INCH_TO_CM: Final[float] = 2.54  # exact by definition

# Longitude/latitude are stored in WGS84. The censuses also provide projected
# state-plane coordinates (x_sp/y_sp) which we do not use in the canonical panel
# but record here for downstream spatial joins to NYC LION / renewal layers.
CRS_WGS84: Final[str] = "EPSG:4326"
CRS_NY_STATE_PLANE_FT: Final[str] = "EPSG:2263"  # NAD83 / New York Long Island (ftUS)

# Bounding box of New York City (WGS84), padded slightly beyond the official
# extent. Coordinates outside this box are treated as invalid placeholders
# (the 1995 file contains lon/lat == 1.0; the 2005 file contains 0.0).
NYC_BBOX: Final[dict[str, float]] = {
    "lon_min": -74.30,
    "lon_max": -73.68,
    "lat_min": 40.47,
    "lat_max": 40.93,
}

# Plausible DBH range for a street tree, in inches. Values of 0 are legitimate
# for stumps/empty sites but are *not* valid for a living trunk; negative values
# (2005 uses -1 as a sentinel) and absurd values (2005: 2100 in; 2015: 450 in)
# are measurement errors. We never silently drop records — we flag them.
DBH_MIN_VALID_IN: Final[float] = 0.0   # exclusive lower bound for a live trunk
DBH_MAX_VALID_IN: Final[float] = 120.0  # ~305 cm; generous ceiling for any street tree

# DBH size classes in centimetres (SI), defined on 6-inch increments to mirror
# the native inch-based measurements and standard urban-forestry diameter bins.
# Left-closed, right-open bins; the final class is open-ended.
DBH_SIZE_CLASS_EDGES_CM: Final[tuple[float, ...]] = (0.0, 15.2, 30.5, 45.7, 61.0, 76.2, float("inf"))
DBH_SIZE_CLASS_LABELS: Final[tuple[str, ...]] = (
    "dbh_0_15", "dbh_15_30", "dbh_30_46", "dbh_46_61", "dbh_61_76", "dbh_76plus",
)
# A tree with DBH >= this threshold (cm) is treated as large/mature canopy, a
# proxy for the structural value that renewal most often removes or adds slowly.
LARGE_DBH_THRESHOLD_CM: Final[float] = 61.0  # ~24 inches

# --------------------------------------------------------------------------- #
# Canonical schema
# --------------------------------------------------------------------------- #
# The ordered list of columns produced by the census harmoniser. Downstream code
# should reference these names, never the raw NYC column names.
CANONICAL_COLUMNS: Final[tuple[str, ...]] = (
    "source_year",     # int  — canonical census epoch (1995/2005/2015)
    "orig_id",         # str  — original record id within that census
    "spc_latin",       # str  — normalised Latin name (Title case), nullable
    "spc_latin_raw",   # str  — original Latin name as published, nullable
    "spc_common",      # str  — common name (lower case), nullable
    "genus",           # str  — genus (first token of species_key), nullable
    "species_key",     # str  — stable cross-year key: "genus epithet", nullable
    "dbh_in",          # float — trunk diameter at breast height, inches (raw)
    "dbh_cm",          # float — DBH in centimetres (SI); NaN if not valid
    "dbh_valid",       # bool  — DBH within (DBH_MIN_VALID_IN, DBH_MAX_VALID_IN]
    "alive",           # bool  — harmonised life status; nullable (pandas BooleanDtype)
    "condition",       # str  — harmonised condition category (see CONDITION_ORDER)
    "tree_present",    # bool  — False for 1995 "Planting Space" (empty site)
    "status_raw",      # str  — original status/condition token, nullable
    "health_raw",      # str  — original health token (2015 only), nullable
    "lon",             # float — longitude (WGS84)
    "lat",             # float — latitude (WGS84)
    "coord_valid",     # bool  — lon/lat inside NYC_BBOX
    "borough",         # str  — normalised borough name
    "nta_code",        # str  — Neighborhood Tabulation Area code (2010), nullable
    "nta_name",        # str  — NTA name, nullable
    "census_tract",    # str  — 2010 census tract, nullable
    "bbl",             # str  — Borough-Block-Lot tax key, nullable
)

# Harmonised condition categories, ordered from best to "not a living tree".
# ``NoTree`` is used only for 1995 "Planting Space" (an empty planting site).
CONDITION_ORDER: Final[tuple[str, ...]] = (
    "Excellent",
    "Good",
    "Fair",
    "Poor",
    "Critical",
    "Dead",
    "Stump",
    "Unknown",
    "NoTree",
)

# Conditions that represent a *living* tree (used to derive ``alive``).
LIVING_CONDITIONS: Final[frozenset[str]] = frozenset(
    {"Excellent", "Good", "Fair", "Poor", "Critical"}
)

# Coarse, cross-year-comparable condition scale for living trees. The three
# censuses use different fine vocabularies (1995: Excellent/Good/Fair/Poor/
# Critical; 2005: Excellent/Good/Poor; 2015: Good/Fair/Poor), so fine condition
# is NOT directly comparable across epochs. This 3-level collapse is the most
# defensible common denominator; note that 2005 structurally lacks "Fair".
CONDITION_COARSE_MAP: Final[dict[str, str]] = {
    "Excellent": "Good",
    "Good": "Good",
    "Fair": "Fair",
    "Poor": "Poor",
    "Critical": "Poor",
}
# Dead/Stump/Unknown/NoTree deliberately absent -> not counted in condition shares.

# --------------------------------------------------------------------------- #
# Per-census column mappings  (raw name -> canonical name)
# --------------------------------------------------------------------------- #
# Only the columns we need are listed; everything else in the raw files is
# ignored. ``usecols`` in the loader is derived from these keys.
COLUMNS_1995: Final[dict[str, str]] = {
    "RecordId": "orig_id",
    "Spc_Latin": "spc_latin_raw",
    "Spc_Common": "spc_common",
    "Diameter": "dbh_in",
    "Condition": "status_raw",
    "Longitude": "lon",
    "Latitude": "lat",
    "Borough": "borough",
    "NTA_2010": "nta_code",
    "SegmentID": "nta_name",  # placeholder; overwritten below — see note
    "CensusTract_2010": "census_tract",
    "BBL": "bbl",
}
# NOTE: the 1995 file has no NTA *name* column, only the code (NTA_2010). To keep
# the mapping honest we do not invent one; the placeholder above is corrected by
# removing it here so nta_name stays null for 1995.
del COLUMNS_1995["SegmentID"]

COLUMNS_2005: Final[dict[str, str]] = {
    "OBJECTID": "orig_id",
    "spc_latin": "spc_latin_raw",
    "spc_common": "spc_common",
    "tree_dbh": "dbh_in",
    "status": "status_raw",
    "longitude": "lon",
    "latitude": "lat",
    "boroname": "borough",
    "nta": "nta_code",
    "nta_name": "nta_name",
    "census tract": "census_tract",
    "bbl": "bbl",
}

COLUMNS_2015: Final[dict[str, str]] = {
    "tree_id": "orig_id",
    "spc_latin": "spc_latin_raw",
    "spc_common": "spc_common",
    "tree_dbh": "dbh_in",
    "status": "status_raw",   # Alive / Dead / Stump
    "health": "health_raw",   # Good / Fair / Poor (Alive only)
    "longitude": "lon",
    "latitude": "lat",
    "borough": "borough",
    "nta": "nta_code",
    "nta_name": "nta_name",
    "census tract": "census_tract",
    "bbl": "bbl",
}

COLUMN_MAPS: Final[dict[int, dict[str, str]]] = {
    1995: COLUMNS_1995,
    2005: COLUMNS_2005,
    2015: COLUMNS_2015,
}

# --------------------------------------------------------------------------- #
# Per-census condition / status harmonisation
# --------------------------------------------------------------------------- #
# Maps the raw condition/status token (upper-cased, stripped) to a canonical
# CONDITION_ORDER category. Tokens absent from a map resolve to "Unknown".

# 1995 — "Condition" field.
CONDITION_MAP_1995: Final[dict[str, str]] = {
    "EXCELLENT": "Excellent",
    "GOOD": "Good",
    "FAIR": "Fair",
    "POOR": "Poor",
    "CRITICAL": "Critical",
    "DEAD": "Dead",
    "STUMP": "Stump",
    "SHAFT": "Stump",          # standing dead trunk -> treated as stump-like
    "PLANTING SPACE": "NoTree",  # empty planting site, not a tree
    "UNKNOWN": "Unknown",
}

# 2005 — "status" field is a condition scale (no Fair/Stump present in the data).
CONDITION_MAP_2005: Final[dict[str, str]] = {
    "EXCELLENT": "Excellent",
    "GOOD": "Good",
    "POOR": "Poor",
    "DEAD": "Dead",
}

# 2015 — combine "status" (life state) with "health" (condition of living trees).
# Handled procedurally in the loader, but the health vocabulary is declared here.
HEALTH_MAP_2015: Final[dict[str, str]] = {
    "GOOD": "Good",
    "FAIR": "Fair",
    "POOR": "Poor",
}

CONDITION_MAPS: Final[dict[int, dict[str, str]]] = {
    1995: CONDITION_MAP_1995,
    2005: CONDITION_MAP_2005,
    # 2015 is derived in code from status + health; see census.py
}

# --------------------------------------------------------------------------- #
# Borough normalisation
# --------------------------------------------------------------------------- #
# The 2005 file encodes Staten Island as the borocode "5" in the boroname field.
# Numeric borocodes are mapped to names; already-correct names pass through.
BOROUGH_MAP: Final[dict[str, str]] = {
    "1": "Manhattan",
    "2": "Bronx",
    "3": "Brooklyn",
    "4": "Queens",
    "5": "Staten Island",
    "MANHATTAN": "Manhattan",
    "BRONX": "Bronx",
    "BROOKLYN": "Brooklyn",
    "QUEENS": "Queens",
    "STATEN ISLAND": "Staten Island",
}

# Species tokens that denote "not identified" and should map to a null species.
SPECIES_NULL_TOKENS: Final[frozenset[str]] = frozenset(
    {"", "UNKNOWN", "UNKNOWN SPECIES", "TREE", "NA", "N/A", "NONE"}
)

# --------------------------------------------------------------------------- #
# Ecosystem services: carbon from allometry
# --------------------------------------------------------------------------- #
# Total aboveground biomass follows Jenkins et al. (2003), Forest Science 49(1):
#   AGB_kg = exp(beta0 + beta1 * ln(dbh_cm)),  for trees >= 2.5 cm DBH.
# Coefficients per species group, with the maximum DBH (cm) of the source trees
# (beyond which the equation extrapolates).
JENKINS_COEFFS: Final[dict[str, tuple[float, float]]] = {
    "aspen_alder_cottonwood_willow": (-2.2094, 2.3867),
    "soft_maple_birch": (-1.9123, 2.3651),
    "mixed_hardwood": (-2.4800, 2.4835),
    "hard_maple_oak_hickory_beech": (-2.0127, 2.4342),
    "cedar_larch": (-2.0336, 2.2592),
    "douglas_fir": (-2.2304, 2.4435),
    "true_fir_hemlock": (-2.5384, 2.4814),
    "pine": (-2.5356, 2.4349),
    "spruce": (-2.0773, 2.3323),
    "woodland": (-0.7152, 1.7029),
}
JENKINS_MAX_DBH_CM: Final[dict[str, float]] = {
    "aspen_alder_cottonwood_willow": 70.0,
    "soft_maple_birch": 66.0,
    "mixed_hardwood": 56.0,
    "hard_maple_oak_hickory_beech": 73.0,
    "cedar_larch": 250.0,
    "douglas_fir": 210.0,
    "true_fir_hemlock": 230.0,
    "pine": 180.0,
    "spruce": 250.0,
    "woodland": 78.0,
}
JENKINS_MIN_DBH_CM: Final[float] = 2.5          # lower validity bound of the equations
JENKINS_DEFAULT_GROUP: Final[str] = "mixed_hardwood"  # fallback for unresolved hardwoods

# Genus -> Jenkins group. NYC street trees are overwhelmingly hardwoods; the
# mixed-hardwood group is the catch-all. Ambiguous maples are split by species
# in SPECIES_TO_JENKINS below.
GENUS_TO_JENKINS: Final[dict[str, str]] = {
    "populus": "aspen_alder_cottonwood_willow",
    "salix": "aspen_alder_cottonwood_willow",
    "alnus": "aspen_alder_cottonwood_willow",
    "betula": "soft_maple_birch",
    "acer": "soft_maple_birch",            # default; overridden per species below
    "quercus": "hard_maple_oak_hickory_beech",
    "carya": "hard_maple_oak_hickory_beech",
    "fagus": "hard_maple_oak_hickory_beech",
    "ostrya": "hard_maple_oak_hickory_beech",
    "carpinus": "hard_maple_oak_hickory_beech",
    "pinus": "pine",
    "picea": "spruce",
    "abies": "true_fir_hemlock",
    "tsuga": "true_fir_hemlock",
    "pseudotsuga": "douglas_fir",
    "thuja": "cedar_larch",
    "juniperus": "cedar_larch",
    "larix": "cedar_larch",
    "chamaecyparis": "cedar_larch",
    "metasequoia": "cedar_larch",
    "taxodium": "cedar_larch",
    "calocedrus": "cedar_larch",
    # Common NYC hardwoods -> mixed hardwood (explicit for readability/audit).
    "tilia": "mixed_hardwood", "fraxinus": "mixed_hardwood", "platanus": "mixed_hardwood",
    "gleditsia": "mixed_hardwood", "ulmus": "mixed_hardwood", "pyrus": "mixed_hardwood",
    "prunus": "mixed_hardwood", "styphnolobium": "mixed_hardwood", "sophora": "mixed_hardwood",
    "ginkgo": "mixed_hardwood", "zelkova": "mixed_hardwood", "celtis": "mixed_hardwood",
    "robinia": "mixed_hardwood", "liquidambar": "mixed_hardwood", "malus": "mixed_hardwood",
    "cornus": "mixed_hardwood", "crataegus": "mixed_hardwood", "aesculus": "mixed_hardwood",
    "catalpa": "mixed_hardwood", "koelreuteria": "mixed_hardwood", "magnolia": "mixed_hardwood",
    "liriodendron": "mixed_hardwood", "nyssa": "mixed_hardwood", "amelanchier": "mixed_hardwood",
}
# Species-level overrides where a genus spans Jenkins groups (hard vs soft maple).
SPECIES_TO_JENKINS: Final[dict[str, str]] = {
    "acer saccharum": "hard_maple_oak_hickory_beech",
    "acer nigrum": "hard_maple_oak_hickory_beech",
    "acer rubrum": "soft_maple_birch",
    "acer saccharinum": "soft_maple_birch",
    "acer negundo": "soft_maple_birch",
    "acer platanoides": "mixed_hardwood",
    "acer pseudoplatanus": "mixed_hardwood",
    "acer campestre": "mixed_hardwood",
}

# Nowak & Crane (2002), Environmental Pollution 116, urban-tree carbon method:
URBAN_BIOMASS_ADJ: Final[float] = 0.8   # open-grown trees carry less biomass than forest trees
ROOT_SHOOT_RATIO: Final[float] = 0.26   # whole-tree biomass = AGB * (1 + 0.26); Cairns et al. 1997
CARBON_FRACTION: Final[float] = 0.5     # dry biomass -> stored carbon
CO2_PER_C: Final[float] = 44.0 / 12.0   # stored carbon -> CO2 equivalent

# Optional growth-based gross sequestration (approximate). Average annual DBH
# increment for established street trees, scaled by condition. The preferred,
# empirical estimate is the cross-epoch change in stored carbon (see Batch 5).
DBH_GROWTH_CM_PER_YR: Final[float] = 0.61  # ~0.24 in/yr; adjust per local growth data
CONDITION_GROWTH_FACTOR: Final[dict[str, float]] = {"Good": 1.0, "Fair": 0.76, "Poor": 0.50}

# --------------------------------------------------------------------------- #
# Spatial units for structural change analysis
# --------------------------------------------------------------------------- #
# Change between censuses is measured on a shared spatial unit, since the three
# inventories carry no persistent per-tree identifiers across epochs. Each key
# maps to the canonical grouping column(s). NTA codes are unique city-wide;
# census-tract numbers repeat per borough, hence they are paired with borough.
SPATIAL_UNITS: Final[dict[str, tuple[str, ...]]] = {
    "nta": ("nta_code",),
    "census_tract": ("borough", "census_tract"),
    "borough": ("borough",),
}
DEFAULT_SPATIAL_UNIT: Final[str] = "nta"

# --------------------------------------------------------------------------- #
# Street-view imagery (optional Green View Index "function" layer)
# --------------------------------------------------------------------------- #
# Mapillary is the recommended source: free, open-licence, non-Chinese, with
# per-image timestamps. A token is supplied at run time (never stored here).
MAPILLARY_GRAPH_URL: Final[str] = "https://graph.mapillary.com"

# Default green-pixel extraction: Excess Green index ExG = 2g - r - b on channels
# normalised to [0, 1]; a pixel is vegetation when ExG exceeds this threshold.
# This color-index approach follows the GVI tradition (Yang 2009; Li 2015) and
# needs no model weights/GPU. A deep-learning segmenter is an optional upgrade.
SVI_EXG_THRESHOLD: Final[float] = 0.10

# Epochs that street-view imagery is binned to, and the maximum year gap for an
# image to count toward an epoch. NOTE: crowdsourced SVI realistically covers
# only recent years, so for a real study set these to the years your imagery
# actually spans (e.g., (2015, 2020, 2024)); the census epochs are the default
# only so the layer aligns with the inventory panel out of the box.
SVI_EPOCHS: Final[tuple[int, ...]] = (1995, 2005, 2015)
SVI_EPOCH_MAX_GAP_YEARS: Final[int] = 3

# --------------------------------------------------------------------------- #
# Urban-renewal treatment definition
# --------------------------------------------------------------------------- #
# A spatial unit counts as "treated" when at least this share of its street
# trees fall within an urban-renewal / rezoning footprint. The treatment year is
# the earliest effective year of an overlapping renewal action.
RENEWAL_TREATED_SHARE_THRESHOLD: Final[float] = 0.10
# Plausibility window for a renewal effective year (guards against bad dates).
RENEWAL_MIN_YEAR: Final[int] = 1960
RENEWAL_MAX_YEAR: Final[int] = 2025

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DIR",
    "INTERIM_DIR",
    "PROCESSED_DIR",
    "FIGURES_DIR",
    "TABLES_DIR",
    "CANONICAL_PANEL_PARQUET",
    "CENSUS_YEARS",
    "RAW_FILENAMES",
    "INCH_TO_CM",
    "CRS_WGS84",
    "CRS_NY_STATE_PLANE_FT",
    "NYC_BBOX",
    "DBH_MIN_VALID_IN",
    "DBH_MAX_VALID_IN",
    "DBH_SIZE_CLASS_EDGES_CM",
    "DBH_SIZE_CLASS_LABELS",
    "LARGE_DBH_THRESHOLD_CM",
    "CANONICAL_COLUMNS",
    "CONDITION_ORDER",
    "LIVING_CONDITIONS",
    "CONDITION_COARSE_MAP",
    "COLUMN_MAPS",
    "CONDITION_MAPS",
    "HEALTH_MAP_2015",
    "BOROUGH_MAP",
    "SPECIES_NULL_TOKENS",
    "SPATIAL_UNITS",
    "DEFAULT_SPATIAL_UNIT",
    "JENKINS_COEFFS",
    "JENKINS_MAX_DBH_CM",
    "JENKINS_MIN_DBH_CM",
    "JENKINS_DEFAULT_GROUP",
    "GENUS_TO_JENKINS",
    "SPECIES_TO_JENKINS",
    "URBAN_BIOMASS_ADJ",
    "ROOT_SHOOT_RATIO",
    "CARBON_FRACTION",
    "CO2_PER_C",
    "DBH_GROWTH_CM_PER_YR",
    "CONDITION_GROWTH_FACTOR",
    "RENEWAL_TREATED_SHARE_THRESHOLD",
    "RENEWAL_MIN_YEAR",
    "RENEWAL_MAX_YEAR",
    "MAPILLARY_GRAPH_URL",
    "SVI_EXG_THRESHOLD",
    "SVI_EPOCHS",
    "SVI_EPOCH_MAX_GAP_YEARS",
]
