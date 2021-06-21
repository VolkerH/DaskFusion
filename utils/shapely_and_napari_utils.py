import numpy as np
from typing import Union, Tuple, Sequence, List
import copy
from itertools import product
from shapely.affinity import affine_transform, translate, scale
from shapely.geometry.polygon import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.geometry import  GeometryCollection, LineString
from skimage.transform import AffineTransform


def create_grid_of_shapes(grid_shape: Tuple[int, int], grid_spacing: Tuple[float, float], elements: Union[BaseGeometry, Sequence[BaseGeometry]]) -> List[BaseGeometry]:
    """Create a grid of elements

    Args:
        grid_shape: shape of the grid
        grid_spacing: spacing between elements of the grid along directions
        elements: either a single shapely object derive from BaseGeometry that is used as a template, or a sequence of such objects

    Returns:
        [type]: [description]
    """

    grid = []
    spacing = np.array(grid_spacing)
    if not hasattr(type(elements), '__iter__'):
        elements = [copy.deepcopy(elements) for i in range(grid_shape[0]*grid_shape[1])]
    for ((y, x), e) in zip(product(range(grid_shape[1]), range(grid_shape[0])), elements):
        displacement = np.array((x,y)) * spacing
        grid.append(translate(e, displacement[0], displacement[1]))
    return grid


# from jni's affinder
def calculate_transform(src, dst, model_class=AffineTransform):
    """Calculate transformation matrix from matched coordinate pairs.
    Parameters
    ----------
    src : ndarray
        Matched row, column coordinates from source image.
    dst : ndarray
        Matched row, column coordinates from destination image.
    model_class : scikit-image transformation class, optional.
        By default, model=AffineTransform().
    Returns
    -------
    transform
        scikit-image Transformation object
    """
    model = model_class()
    model.estimate(dst, src)  # we want the inverse
    return model


def napari_shape_to_shapely(coords: np.ndarray, shape_type: str="polygon"):
    """ 
    Convert an individual napari shape from a shapes layer to a shapely object
    
    
    There is no direct correspondence between a 'rectangle' in napari and a 'box' in 
    shapely. The command for creating a box only requires minimum and maximum x and y
    coordinates, i.e. this is for axis aligned rectangles. In contrast, a napari rectangle
    can be arbitrarily rotated and still be a rectangle. 
    So for converting a rectangle from napari to shapely we have to go to a polygon
    """
    _coords  = coords[:,::-1].copy() # shapely has col,row order, numpy row,col
    _coords[:,1] *= -1 # axis direction flipped between shapely and napari
    if shape_type in ("rectangle", "polygon", "ellipse"):
        return Polygon(_coords)
    elif shape_type in ("line", "path"):
        return LineString(_coords)
    else:
        raise ValueError
        
def napari_shape_layer_to_shapely(s):
    """
    Convert all shapes in a napari shape layer to shapely objects
    and return a GeometryCollection
    """
    shapes = []
    for _coord, _st in zip(s.data, s.shape_type):
        shapes.append(napari_shape_to_shapely(_coord, _st))
    return GeometryCollection(shapes)

def get_transformed_bbox(shape, affine_matrix: np.ndarray) -> np.ndarray:
    """
    returns the corner coordinates  of a 2D array with shape shape
    after applying the affine transform transform.
    This corresponds to the oriented bounding box
    """
    h,w = shape
    # create homogeneous coordinates for corner points
    baserect = np.array([[0,0],[h,0],[h,w],[0,w]])
    augmented_baserect = np.concatenate((baserect,np.ones((baserect.shape[0],1))), axis=1)
    # see where the corner points map to 
    transformed_rect = (affine_matrix @ augmented_baserect.T).T[:,:-1]
    return transformed_rect

def get_image_layer_rect(layer):
    """given a napari image or labels layer return the oriented bounding box
    coordinates that can be added to a napari shape layer"""
    im = layer.data
    if layer.data.ndim>2:
        im=np.squeeze(im)
    
    if im.ndim != 2:
        raise ValueError("Layer is not a 2D single channel layer")
    
    return get_transformed_bbox(im.shape, layer.affine.affine_matrix)

