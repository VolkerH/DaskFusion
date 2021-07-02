from functools import partial
from utils.imutils import apply_transform_chain
from utils.chunks import get_chunk_coordinates, get_rect_from_chunk_slice, tile_chunk_intersections
import numpy as np
from typing import Callable, Union, Tuple, Sequence, List
import copy
from itertools import product
from shapely.affinity import translate
from shapely.geometry.polygon import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.geometry import GeometryCollection, LineString
from skimage.transform import AffineTransform
from dask.array.core import normalize_chunks

def create_grid_of_shapes(
    grid_shape: Tuple[int, int],
    grid_spacing: Tuple[float, float],
    elements: Union[BaseGeometry, Sequence[BaseGeometry]],
) -> List[BaseGeometry]:
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
    if not hasattr(type(elements), "__iter__"):
        elements = [
            copy.deepcopy(elements) for i in range(grid_shape[0] * grid_shape[1])
        ]
    for ((y, x), e) in zip(
        product(range(grid_shape[1]), range(grid_shape[0])), elements
    ):
        displacement = np.array((x, y)) * spacing
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


def napari_shape_to_shapely(coords: np.ndarray, shape_type: str = "polygon"):
    """
    Convert an individual napari shape from a shapes layer to a shapely object


    There is no direct correspondence between a 'rectangle' in napari and a 'box' in
    shapely. The command for creating a box only requires minimum and maximum x and y
    coordinates, i.e. this is for axis aligned rectangles. In contrast, a napari rectangle
    can be arbitrarily rotated and still be a rectangle.
    So for converting a rectangle from napari to shapely we have to go to a polygon
    """
    _coords = coords[:, ::-1].copy()  # shapely has col,row order, numpy row,col
    _coords[:, 1] *= -1  # axis direction flipped between shapely and napari
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
    h, w = shape
    # create homogeneous coordinates for corner points
    baserect = np.array([[0, 0], [h, 0], [h, w], [0, w]])
    augmented_baserect = np.concatenate(
        (baserect, np.ones((baserect.shape[0], 1))), axis=1
    )
    # see where the corner points map to
    transformed_rect = (affine_matrix @ augmented_baserect.T).T[:, :-1]
    return transformed_rect


def get_image_layer_rect(layer):
    """given a napari image or labels layer return the oriented bounding box
    coordinates that can be added to a napari shape layer"""
    im = layer.data
    if layer.data.ndim > 2:
        im = np.squeeze(im)

    if im.ndim != 2:
        raise ValueError("Layer is not a 2D single channel layer")

    return get_transformed_bbox(im.shape, layer.affine.affine_matrix)


def get_mosaics_bboxes_transforms(images: List[np.ndarray], 
                transform_chain: Sequence[Callable], 
                normalized_coordinates,
                factor):


    assert len(images)            
    _apply_transform_chain = partial(apply_transform_chain, transforms=transform_chain)
    images = list(map(_apply_transform_chain, images))
    tile_shape= images[0].shape

    bboxes = []
    transforms = []
    
    scale_up = AffineTransform(scale=(factor,factor))
    scale_down  = AffineTransform(scale=(1.0/factor,1.0/factor))
    
    # get bboxes of image tiles after moving them to stage position and scaling
    for coord in normalized_coordinates:
        translate = AffineTransform(translation=coord)
        transform = scale_down.params @ translate.params @ scale_up.params
        bboxes.append(
            get_transformed_bbox(tile_shape[:2], transform))
        transforms.append(transform)

    # determine overall shape (size of fused image)
    all_bboxes = np.vstack(bboxes)
    # minimum  & maximum extents of tile collection
    all_min = all_bboxes.min(axis=0)
    all_max = all_bboxes.max(axis=0)
    stitched_shape=tuple(np.ceil(all_max-all_min).astype(int))

    # determine required shift to origin and update transforms
    shift_to_origin = AffineTransform(translation=-all_min)
    transforms = [shift_to_origin.params @ t  for t in transforms]

    shifted_bboxes = []
    for t, coord in zip(transforms,
                        normalized_coordinates):
        shifted_bboxes.append(
            get_transformed_bbox(tile_shape, t))
    mosaic_shifted = GeometryCollection([napari_shape_to_shapely(s) for s in shifted_bboxes])

    return {'mosaic_shifted': mosaic_shifted, 'stitched_shape': stitched_shape, 'transforms': transforms ,  'transformed_images': images}


def get_chunk_slices_and_shapes(chunk_size: Tuple[int,int], array_shape: Tuple[int,int]):
    chunks = normalize_chunks((4096, 4096), shape=array_shape)
    # sanity check, can be removed eventually
    computed_shape = np.array(list(map(sum, chunks)))
    assert np.all(np.array(array_shape) == computed_shape)
    chunk_slices = list(get_chunk_coordinates(array_shape, chunk_size))
    chunk_shapes = list(map(get_rect_from_chunk_slice, chunk_slices))
    chunks_shapely = GeometryCollection([napari_shape_to_shapely(c) for c in chunk_shapes])
    return {'chunk_slices': chunk_slices, 'chunks_shapely': chunks_shapely}
