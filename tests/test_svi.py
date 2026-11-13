"""Tests for the street-view / GVI layer (computable parts, offline)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stgreen.svi.green import exg_green_fraction, ExGSegmenter
from stgreen.svi.harvest import assign_epoch, make_synthetic_records
from stgreen.metrics.gvi import score_records, aggregate_gvi


def test_exg_green_fraction():
    green = np.zeros((10, 10, 3), dtype="uint8"); green[..., 1] = 255   # pure green
    gray = np.full((10, 10, 3), 128, dtype="uint8")                     # neutral
    half = np.concatenate([green[:5], gray[5:]], axis=0)                # 50/50
    assert exg_green_fraction(green) == pytest.approx(1.0)
    assert exg_green_fraction(gray) == pytest.approx(0.0)
    assert exg_green_fraction(half) == pytest.approx(0.5)
    assert isinstance(ExGSegmenter().vegetation_fraction(green), float)


def test_assign_epoch():
    assert assign_epoch(2014, epochs=(1995, 2005, 2015), max_gap=3) == 2015
    assert assign_epoch(2003, epochs=(1995, 2005, 2015), max_gap=3) == 2005
    assert assign_epoch(2000, epochs=(1995, 2005, 2015), max_gap=3) is None  # 5-yr gap
    assert assign_epoch(None) is None


def test_gvi_aggregation_bounds_and_counts():
    pts = pd.DataFrame({"point_id": ["0", "1"], "lon": [-73.95, -73.90],
                        "lat": [40.70, 40.72], "unit_id": ["A", "A"]})
    recs = make_synthetic_records(pts, epochs=(2015,), images_per_point=3)
    gvi = aggregate_gvi(score_records(recs), unit="nta")
    row = gvi.iloc[0]
    assert 0.0 <= row["gvi_mean"] <= 100.0
    assert row["n_points"] == 2
    assert row["n_images"] == 6                     # 2 points x 3 images
    assert row["source_year"] == 2015               # epoch -> source_year
