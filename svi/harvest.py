"""Sample locations and harvest street-view image metadata (Mapillary).

Workflow: choose sample points (the coordinate-valid street-tree locations are a
natural proxy for "where street greenery is seen"), query a provider for nearby
images with timestamps, bin each image to a study epoch, and record where its
pixels will be scored for greenery. Image *scoring* happens in
:mod:`stgreen.svi.green`; aggregation to the Green View Index in
:mod:`stgreen.metrics.gvi`.

Mapillary is the recommended provider (free, open licence, non-Chinese,
timestamped). Access needs a token supplied at run time via the
``MAPILLARY_TOKEN`` environment variable -- it is never written to disk or stored
in config. Real harvesting also needs network access to the Mapillary Graph API,
which may be unavailable in restricted environments; in that case use
:func:`make_synthetic_records` to exercise the rest of the pipeline offline.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from stgreen import config as cfg

logger = logging.getLogger(__name__)

# ~111 km per degree latitude; used to turn a metre radius into a small bbox.
_M_PER_DEG = 111_000.0


def assign_epoch(year: int | float, epochs=cfg.SVI_EPOCHS,
                 max_gap: int = cfg.SVI_EPOCH_MAX_GAP_YEARS) -> int | None:
    """Nearest epoch to ``year`` within ``max_gap`` years, else ``None``."""
    if year is None or (isinstance(year, float) and np.isnan(year)):
        return None
    diffs = [(abs(int(year) - e), e) for e in epochs]
    gap, epoch = min(diffs)
    return epoch if gap <= max_gap else None


def sample_points(panel: pd.DataFrame, unit: str, n_per_unit: int | None = None,
                  seed: int = 0) -> pd.DataFrame:
    """Candidate SVI sample points from coordinate-valid tree locations.

    Returns one row per point with ``point_id``, ``lon``, ``lat`` and ``unit_id``.
    ``n_per_unit`` optionally caps the points sampled per spatial unit.
    """
    unit_cols = list(cfg.SPATIAL_UNITS[unit])
    pts = (
        panel[panel["coord_valid"].eq(True)]
        .dropna(subset=unit_cols)
        .drop_duplicates(subset=["lon", "lat"])
        [["lon", "lat", *unit_cols]]
        .copy()
    )
    if len(unit_cols) == 1:
        pts["unit_id"] = pts[unit_cols[0]].astype("string")
    else:
        pts["unit_id"] = pts[unit_cols].astype("string").agg("|".join, axis=1)

    if n_per_unit is not None:
        # Global shuffle, then take up to n_per_unit per unit (caps small units;
        # retains all columns, unlike groupby.apply under pandas 2.x).
        pts = (
            pts.sample(frac=1.0, random_state=seed)
            .groupby("unit_id", group_keys=False, sort=False)
            .head(n_per_unit)
        )
    pts = pts.reset_index(drop=True)
    pts["point_id"] = pts.index.astype(str)
    return pts[["point_id", "lon", "lat", "unit_id"]]


# --------------------------------------------------------------------------- #
# Mapillary provider
# --------------------------------------------------------------------------- #
class MapillaryClient:
    """Minimal Mapillary Graph API client for image discovery.

    Token is read from the ``MAPILLARY_TOKEN`` environment variable unless passed
    explicitly. No credentials are persisted.
    """

    def __init__(self, token: str | None = None, base_url: str = cfg.MAPILLARY_GRAPH_URL) -> None:
        self.token = token or os.environ.get("MAPILLARY_TOKEN")
        if not self.token:
            raise RuntimeError(
                "No Mapillary token. Set MAPILLARY_TOKEN or pass token=... "
                "(use make_synthetic_records for offline runs)."
            )
        self.base_url = base_url

    def images_near(self, lon: float, lat: float, radius_m: float = 20.0,
                    start_year: int | None = None, end_year: int | None = None) -> list[dict]:
        """Return image records (id, lon, lat, captured_at) within a small bbox."""
        import requests  # local import keeps requests optional

        dlat = radius_m / _M_PER_DEG
        dlon = radius_m / (_M_PER_DEG * max(np.cos(np.radians(lat)), 1e-6))
        bbox = f"{lon - dlon},{lat - dlat},{lon + dlon},{lat + dlat}"
        params = {"fields": "id,captured_at,geometry", "bbox": bbox,
                  "access_token": self.token, "limit": 50}
        r = requests.get(f"{self.base_url}/images", params=params, timeout=30)
        r.raise_for_status()
        out = []
        for d in r.json().get("data", []):
            coords = (d.get("geometry") or {}).get("coordinates") or [lon, lat]
            ts = d.get("captured_at")
            year = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).year if ts else None
            if start_year and year and year < start_year:
                continue
            if end_year and year and year > end_year:
                continue
            out.append({"image_id": str(d["id"]), "lon": coords[0], "lat": coords[1], "year": year})
        return out


def harvest(points: pd.DataFrame, client: MapillaryClient, radius_m: float = 20.0) -> pd.DataFrame:
    """Harvest image metadata for every sample point. Returns an image-record table.

    Columns: point_id, unit_id, image_id, lon, lat, year, epoch. (Pixel scoring and
    GVI are computed downstream.)
    """
    rows = []
    for p in points.itertuples():
        for img in client.images_near(p.lon, p.lat, radius_m=radius_m):
            rows.append({"point_id": p.point_id, "unit_id": p.unit_id, **img})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["epoch"] = df["year"].map(assign_epoch)
        df = df.dropna(subset=["epoch"])
        df["epoch"] = df["epoch"].astype(int)
    logger.info("Harvested %d images for %d points", len(df), len(points))
    return df


def make_synthetic_records(points: pd.DataFrame, epochs=cfg.SVI_EPOCHS,
                           images_per_point: int = 3, seed: int = 0) -> pd.DataFrame:
    """Offline fixture: image records with a known ``green_truth`` fraction.

    Lets the GVI aggregation run and be tested without network access or imagery.
    The green fraction is a deterministic function of location, so spatial pattern
    is reproducible (but it is NOT real greenery).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for p in points.itertuples():
        base = 0.15 + 0.3 * (abs(np.sin(p.lat * 7) * np.cos(p.lon * 7)))  # 0.15–0.45
        for e in epochs:
            for k in range(images_per_point):
                g = float(np.clip(base + rng.normal(0, 0.03), 0.0, 1.0))
                rows.append({"point_id": p.point_id, "unit_id": p.unit_id,
                             "image_id": f"{p.point_id}-{e}-{k}", "lon": p.lon, "lat": p.lat,
                             "year": e, "epoch": e, "green_truth": round(g, 4)})
    return pd.DataFrame(rows)
