"""Vegetation extraction from street-view images for the Green View Index.

The default extractor is a **colour index** -- the Excess Green index
``ExG = 2g - r - b`` on channels normalised to [0, 1] -- which classifies a pixel
as vegetation when ExG exceeds a threshold. This needs no model weights, no GPU
and no Hugging Face access, and it follows the colour-index tradition of the
Green View Index (Yang et al., 2009; Li et al., 2015).

For higher accuracy (separating trees from painted-green surfaces, etc.) a
deep-learning semantic segmenter can be dropped in: implement the
:class:`Segmenter` protocol with any model and pass it where a segmenter is
accepted. The GVI arithmetic downstream is identical either way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from stgreen import config as cfg


def _to_unit_rgb(image: np.ndarray) -> np.ndarray:
    """Return an (H, W, 3) float array in [0, 1] from a uint8 or float image."""
    arr = np.asarray(image, dtype="float64")
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError("Expected an (H, W, 3) RGB image.")
    arr = arr[..., :3]
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr


def exg_green_fraction(image: np.ndarray, threshold: float = cfg.SVI_EXG_THRESHOLD) -> float:
    """Fraction of pixels classified as vegetation by the Excess Green index."""
    rgb = _to_unit_rgb(image)
    exg = 2.0 * rgb[..., 1] - rgb[..., 0] - rgb[..., 2]
    return float((exg > threshold).mean())


@runtime_checkable
class Segmenter(Protocol):
    """Anything that maps an RGB image to a vegetation pixel fraction in [0, 1]."""

    def vegetation_fraction(self, image: np.ndarray) -> float: ...


class ExGSegmenter:
    """Default colour-index segmenter (no external model)."""

    def __init__(self, threshold: float = cfg.SVI_EXG_THRESHOLD) -> None:
        self.threshold = threshold

    def vegetation_fraction(self, image: np.ndarray) -> float:
        return exg_green_fraction(image, self.threshold)


def load_image(path: str | Path) -> np.ndarray:
    """Load an image file to an (H, W, 3) uint8 RGB array (needs Pillow)."""
    from PIL import Image  # local import keeps Pillow optional

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def image_green_fraction(source: str | Path | np.ndarray,
                         segmenter: Segmenter | None = None) -> float:
    """Vegetation fraction of an image given as an array or a file path."""
    seg = segmenter or ExGSegmenter()
    img = source if isinstance(source, np.ndarray) else load_image(source)
    return seg.vegetation_fraction(img)
