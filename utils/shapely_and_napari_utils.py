import numpy as np
from typing import Tuple
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import Polygon
from shapely.geometry import LineString


def numpy_shape_to_shapely(coords: np.ndarray, shape_type: str = "polygon") -> BaseGeometry:
    """
    Convert an individual shape represented as a numpy array of coordinates
    to a shapely object
    """
    _coords = coords[:, ::-1].copy()  # shapely has col,row order, numpy row,col
    _coords[:, 1] *= -1  # axis direction flipped between shapely and napari
    if shape_type in ("rectangle", "polygon", "ellipse"):
        return Polygon(_coords)
    elif shape_type in ("line", "path"):
        return LineString(_coords)
    else:
        raise ValueError


def get_transformed_array_corners(shape: Tuple[int,int], affine_matrix: np.ndarray) -> np.ndarray:
    """
    returns the corner coordinates of a 2D array with shape shape
    after applying the transform represented by affine_matrix.
    """
    h, w = shape
    # create homogeneous coordinates for corner points
    baserect = np.array([[0, 0], [h, 0], [h, w], [0, w]])
    augmented_baserect = np.concatenate(
        (baserect, np.ones((baserect.shape[0], 1))), axis=1
    )
    # see where the corner points map to
    transformed_rect = (affine_matrix @ augmented_baserect.T).T[:, :-1]
    return transformed_rect
