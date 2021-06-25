import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np
from napari.layers import Image
from skimage.io import imread

FilePath = Union[Path, str]
ArrayLike = Union[np.ndarray]  # could add other array types if needed


def transpose(array: ArrayLike) -> ArrayLike:
    return array.T


def select_channel(array: ArrayLike, channel=0) -> ArrayLike:
    return np.expand_dims(array[channel, ...], axis=0)


def crop_black_border(array: ArrayLike, border_width: int = 12) -> ArrayLike:
    """
    Crops away the band of black pixels on our nikon camera.
    """
    right = -border_width if border_width > 0 else None
    return array[:, :right]


def subsample(array: ArrayLike, factor: int = 4, method="slice") -> ArrayLike:
    """
    Subsamples the input array along all dimensions using the given
    factor. 'slice' method simply slices with factor as the stride,
    which is fast but leads to sampling artefact. Other methods
    should be added (just dispatch to skimage).
    """
    if method == "slice":
        return array[:, ::factor, ::factor]  # todo: support generic nD
    else:
        raise NotImplementedError("only supporting slice for now")


def load_image(
    file: FilePath, transforms: List[Callable[[ArrayLike], ArrayLike]] = None
) -> np.ndarray:
    img = imread(file)
    #if img.ndim == 2:
    #    img = np.expand_dims(img, axis=0)
    if transforms is not None:
        for t in transforms:
            img = t(img)
    return img


def create_image_layer_name(path: Path, channel: Optional[int] = None) -> str:
    if channel is None:
        return path.stem
    else:
        return f"{path.stem}_[{channel}]"


def find_layer_by_metadata(
    layers: Iterable[Image], metadata: Dict[str, Any]
) -> Optional[Image]:
    candidates = [
        l
        for l in layers
        if all(
            k in l.metadata and l.metadata[k] == v for k, v in metadata.items()
        )
    ]
    assert (
        len(candidates) <= 1
    ), "Multiple matching layers found. Every Layer should be unique for given metadata."
    return candidates[0] if candidates else None

