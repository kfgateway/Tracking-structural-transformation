"""Harmonise the NYC street-tree censuses (1995, 2005, 2015) into one panel.

The three decadal censuses are the structural backbone of the study: each is a
near-complete inventory of street trees with species, trunk diameter (DBH),
condition and location. They are published with *different* schemas, case
conventions and condition vocabularies, and each carries its own data-quality
quirks (placeholder coordinates, sentinel DBH values, a mis-coded borough).

This module reads the raw files and emits a single tidy panel with one row per
tree record and the canonical schema declared in :data:`stgreen.config.CANONICAL_COLUMNS`.
The unique key of a record is the pair ``(source_year, orig_id)``.

Nothing is silently discarded. Problematic values are preserved in the ``*_raw``
columns and marked with boolean validity flags (``dbh_valid``, ``coord_valid``,
``tree_present``) so that every downstream filtering decision is explicit and
auditable — a requirement for the reproducible-data deposit the journal mandates.

Run as a script to (re)build the processed panel::

    python -m stgreen.data.census            # builds data/processed/...parquet
    python -m stgreen.data.census --no-save   # dry run, prints QA summary only
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from stgreen import config as C

logger = logging.getLogger(__name__)

# Collapses any run of whitespace to a single space.
_WS_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------- #
# Text / species normalisation
# --------------------------------------------------------------------------- #
def _clean_text(value: object) -> str | None:
    """Trim and collapse internal whitespace; return ``None`` for empties."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = _WS_RE.sub(" ", str(value)).strip()
    return s or None


def normalize_species_name(raw: object) -> str | None:
    """Return a Title-cased, whitespace-collapsed Latin name, or ``None``.

    The 1995/2005 files store names in UPPER CASE (sometimes with trailing
    cultivar words); 2015 uses Title case with ``var.``/hybrid markers. We
    Title-case consistently, lower-case the botanical markers ``x`` and ``var.``,
    and map explicit "unknown" tokens to ``None``.
    """
    s = _clean_text(raw)
    if s is None or s.upper() in C.SPECIES_NULL_TOKENS:
        return None
    out: list[str] = []
    for idx, tok in enumerate(s.split(" ")):
        low = tok.lower()
        if low in {"x", "×"}:
            out.append("x")               # hybrid marker stays lower-case
        elif low in {"var.", "var", "ssp.", "ssp", "subsp.", "subsp", "f.", "cv."}:
            out.append(low if low.endswith(".") else low + ".")
        elif idx == 0:
            out.append(tok.capitalize())  # genus is capitalised
        else:
            out.append(low)               # epithet/cultivar lower-case (binomial convention)
    return " ".join(out)


def species_key_and_genus(normalized: str | None) -> tuple[str | None, str | None]:
    """Derive a stable cross-year species key and genus from a normalised name.

    The key is ``"genus epithet"`` in lower case, which matches the same taxon
    across censuses regardless of case or cultivar noise:

    * ``"PLATANUS ACERIFOLIA"`` and ``"Platanus x acerifolia"`` -> ``"platanus acerifolia"``
    * ``"ACER PLATANOIDES CRIMSON KING"``                       -> ``"acer platanoides"``
    * ``"Gleditsia triacanthos var. inermis"``                  -> ``"gleditsia triacanthos"``
    * ``"CRATAEGUS SPECIES"``                                   -> ``"crataegus"`` (genus only)
    """
    if normalized is None:
        return None, None
    toks = [t for t in normalized.lower().replace(".", " ").split(" ") if t]
    if not toks:
        return None, None
    genus = toks[0]
    key_parts = [genus]
    i = 1
    while i < len(toks) and toks[i] in {"x", "×"}:  # skip hybrid marker(s)
        i += 1
    if i < len(toks):
        epithet = toks[i]
        if epithet not in {"species", "spp", "sp"}:
            key_parts.append(epithet)
    return " ".join(key_parts), genus.capitalize()


# --------------------------------------------------------------------------- #
# Condition / life-status harmonisation
# --------------------------------------------------------------------------- #
def _condition_to_alive(condition: pd.Series) -> pd.Series:
    """Map a canonical condition Series to a nullable boolean ``alive`` Series."""
    alive = pd.Series(pd.NA, index=condition.index, dtype="boolean")
    alive[condition.isin(C.LIVING_CONDITIONS)] = True
    alive[condition.isin(["Dead", "Stump", "NoTree"])] = False
    # "Unknown" stays <NA>.
    return alive


def _harmonize_condition(year: int, df: pd.DataFrame) -> pd.DataFrame:
    """Add ``condition``, ``alive`` and ``tree_present`` columns for one census.

    1995 / 2005 use a single condition field; 2015 combines a life-status field
    (Alive/Dead/Stump) with a health field (Good/Fair/Poor for living trees).
    """
    status_up = df["status_raw"].str.strip().str.upper()

    if year in (1995, 2005):
        cmap = C.CONDITION_MAPS[year]
        condition = status_up.map(cmap).fillna("Unknown")
        condition = condition.astype("string")
        alive = _condition_to_alive(condition)
        tree_present = condition != "NoTree"

    elif year == 2015:
        health_up = df["health_raw"].str.strip().str.upper()
        health_cond = health_up.map(C.HEALTH_MAP_2015)  # NaN if missing/other
        condition = pd.Series("Unknown", index=df.index, dtype="object")
        condition[status_up == "ALIVE"] = health_cond[status_up == "ALIVE"].fillna("Unknown")
        condition[status_up == "DEAD"] = "Dead"
        condition[status_up == "STUMP"] = "Stump"
        condition = condition.astype("string")
        alive = pd.Series(pd.NA, index=df.index, dtype="boolean")
        alive[status_up == "ALIVE"] = True
        alive[status_up.isin(["DEAD", "STUMP"])] = False
        # 2015 records are all real tree records (no empty planting sites).
        tree_present = pd.Series(True, index=df.index, dtype="boolean")

    else:  # pragma: no cover - guarded by caller
        raise ValueError(f"Unsupported census year: {year}")

    df["condition"] = condition
    df["alive"] = alive
    df["tree_present"] = tree_present.astype("boolean")
    return df


# --------------------------------------------------------------------------- #
# Per-census loader
# --------------------------------------------------------------------------- #
def _raw_path(year: int, data_dir: Path | None) -> Path:
    base = (data_dir / "raw") if data_dir is not None else C.RAW_DIR
    return base / C.RAW_FILENAMES[year]


def load_census(year: int, data_dir: Path | None = None) -> pd.DataFrame:
    """Load and harmonise a single census year into the canonical schema.

    Parameters
    ----------
    year:
        One of :data:`stgreen.config.CENSUS_YEARS`.
    data_dir:
        Optional override of the data directory (expects a ``raw/`` subfolder).
        Defaults to :data:`stgreen.config.RAW_DIR`.

    Returns
    -------
    pandas.DataFrame
        One row per tree record, columns exactly :data:`CANONICAL_COLUMNS`.
    """
    if year not in C.COLUMN_MAPS:
        raise ValueError(f"Unknown census year {year!r}; expected one of {C.CENSUS_YEARS}")

    colmap = C.COLUMN_MAPS[year]
    path = _raw_path(year, data_dir)
    logger.info("Loading %s census from %s", year, path)

    # Read only the columns we need, all as strings, so parsing is explicit.
    raw = pd.read_csv(path, usecols=list(colmap), dtype=str, low_memory=False)
    df = raw.rename(columns=colmap)

    # Ensure optional columns exist even when a given year does not provide them
    # (e.g. 1995 has no nta_name; 1995/2005 have no health_raw).
    for col in ("nta_name", "health_raw"):
        if col not in df.columns:
            df[col] = pd.NA

    df["source_year"] = year

    # --- species ---------------------------------------------------------- #
    # Normalise over the unique raw strings, then map back (fast on ~1.8M rows).
    uniq = pd.Series(df["spc_latin_raw"].dropna().unique())
    norm_lookup = {u: normalize_species_name(u) for u in uniq}
    df["spc_latin"] = df["spc_latin_raw"].map(norm_lookup).astype("string")
    key_genus = {n: species_key_and_genus(n) for n in set(norm_lookup.values())}
    df["species_key"] = df["spc_latin"].map(
        lambda n: key_genus.get(n, (None, None))[0]
    ).astype("string")
    df["genus"] = df["spc_latin"].map(
        lambda n: key_genus.get(n, (None, None))[1]
    ).astype("string")
    df["spc_latin_raw"] = df["spc_latin_raw"].map(_clean_text).astype("string")
    df["spc_common"] = (
        df["spc_common"].map(_clean_text).str.lower().astype("string")
    )

    # --- diameter (inches -> cm, SI) ------------------------------------- #
    dbh_in = pd.to_numeric(df["dbh_in"], errors="coerce")
    dbh_valid = (dbh_in > C.DBH_MIN_VALID_IN) & (dbh_in <= C.DBH_MAX_VALID_IN)
    df["dbh_in"] = dbh_in.astype("float64")
    df["dbh_cm"] = np.where(dbh_valid, (dbh_in * C.INCH_TO_CM).round(1), np.nan)
    df["dbh_valid"] = dbh_valid.astype("boolean")

    # --- condition / alive / tree_present -------------------------------- #
    df = _harmonize_condition(year, df)

    # --- coordinates ------------------------------------------------------ #
    lon = pd.to_numeric(df["lon"], errors="coerce")
    lat = pd.to_numeric(df["lat"], errors="coerce")
    coord_valid = (
        lon.between(C.NYC_BBOX["lon_min"], C.NYC_BBOX["lon_max"])
        & lat.between(C.NYC_BBOX["lat_min"], C.NYC_BBOX["lat_max"])
    )
    df["lon"] = lon.astype("float64")
    df["lat"] = lat.astype("float64")
    df["coord_valid"] = coord_valid.astype("boolean")

    # --- borough / geographic keys --------------------------------------- #
    df["borough"] = (
        df["borough"].str.strip().str.upper().map(C.BOROUGH_MAP)
        .fillna(df["borough"].map(_clean_text))
        .astype("string")
    )
    for col in ("orig_id", "nta_code", "nta_name", "census_tract", "bbl",
                "status_raw", "health_raw"):
        df[col] = df[col].map(_clean_text).astype("string")

    # --- finalise schema -------------------------------------------------- #
    df = df.reindex(columns=C.CANONICAL_COLUMNS)
    logger.info("  -> %s: %d records", year, len(df))
    return df


# --------------------------------------------------------------------------- #
# Panel builder
# --------------------------------------------------------------------------- #
def build_panel(
    years: tuple[int, ...] = C.CENSUS_YEARS,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Concatenate the harmonised censuses into one panel, newest rows last."""
    frames = [load_census(y, data_dir=data_dir) for y in years]
    panel = pd.concat(frames, ignore_index=True)
    panel["source_year"] = panel["source_year"].astype("int16")
    _validate_panel(panel)
    return panel


def _validate_panel(panel: pd.DataFrame) -> None:
    """Fail fast on structural invariants that must hold for every build."""
    assert list(panel.columns) == list(C.CANONICAL_COLUMNS), "column drift"
    assert panel["source_year"].isin(C.CENSUS_YEARS).all(), "unexpected source_year"
    # (source_year, orig_id) must uniquely identify a record.
    dup = panel.duplicated(subset=["source_year", "orig_id"]).sum()
    if dup:
        logger.warning("%d duplicate (source_year, orig_id) keys detected", dup)
    # A living tree should have a valid DBH; warn (do not fail) if many do not.
    live = panel["alive"].eq(True)
    bad = (live & ~panel["dbh_valid"].fillna(False)).sum()
    if bad:
        logger.warning("%d living records have invalid/zero DBH", int(bad))


def save_panel(panel: pd.DataFrame, path: Path = C.CANONICAL_PANEL_PARQUET) -> Path:
    """Write the panel to Parquet, creating the processed directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(path, index=False)
    logger.info("Wrote %d rows to %s", len(panel), path)
    return path


# --------------------------------------------------------------------------- #
# QA summary + CLI
# --------------------------------------------------------------------------- #
def qa_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Return a per-year quality-control table for logging and the data appendix."""
    g = panel.groupby("source_year", observed=True)
    summary = pd.DataFrame(
        {
            "records": g.size(),
            "trees_present": g["tree_present"].sum(),
            "alive": g["alive"].sum(),
            "species_identified": g["species_key"].apply(lambda s: s.notna().sum()),
            "distinct_species": g["species_key"].nunique(),
            "dbh_valid": g["dbh_valid"].sum(),
            "coord_valid": g["coord_valid"].sum(),
            "median_dbh_cm": g["dbh_cm"].median().round(1),
        }
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Override data directory (expects a raw/ subfolder).",
    )
    parser.add_argument(
        "--out", type=Path, default=C.CANONICAL_PANEL_PARQUET,
        help="Output Parquet path for the canonical panel.",
    )
    parser.add_argument(
        "--years", type=int, nargs="+", default=list(C.CENSUS_YEARS),
        help="Census years to include.",
    )
    parser.add_argument("--no-save", action="store_true", help="Build but do not write output.")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging to warnings.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    panel = build_panel(tuple(args.years), data_dir=args.data_dir)

    summary = qa_summary(panel)
    with pd.option_context("display.width", 140, "display.max_columns", None):
        logger.info("QA summary:\n%s", summary.to_string())

    if not args.no_save:
        save_panel(panel, args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
