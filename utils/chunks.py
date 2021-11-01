from typing import Dict, List, Tuple

import numpy as np
from dask.array.core import normalize_chunks
from shapely.strtree import STRtree


def get_chunk_coordinates(shape: Tuple[int, int], chunk_size: Tuple[int, int]):
    """Iterator that returns the bounding coordinates
    for the individual chunks of a dask array of size
    shape with chunk size chunk_size.


    return_np_slice determines the output format. If True,
    a numpy slice object is returned for each chunk, that can be used
    directly to slice a dask array to return the desired chunk region.
    If False, a Tuple of Tuples ((row_min, row_max+1),(col_min, col_max+1))
    is returned.
    """
    chunksy, chunksx = normalize_chunks(chunk_size, shape=shape)
    y = 0
    for cy in chunksy:
        x = 0
        for cx in chunksx:
            yield ((y, y + cy), (x, x + cx))
            x = x + cx
        y = y + cy


def get_rect_from_chunk_boundary(chunk_boundary):
    """given a chunk boundary tuple, return a numpy
    array that can be added as a shape to napari"
    """
    ylim, xlim = chunk_boundary
    miny, maxy = ylim[0], ylim[1] - 1
    minx, maxx = xlim[0], xlim[1] - 1
    return np.array([[miny, minx], [maxy, minx], [maxy, maxx], [miny, maxx]])


def find_chunk_tile_intersections(
    tiles_shapely: List["shapely.geometry.base.BaseGeometry"],
    chunks_shapely: List["shapely.geometry.base.BaseGeometry"],
) -> Dict[Tuple[int, int], Tuple[str, np.ndarray]]:
    """
    For each output array chunk, find the intersecting image tiles

    Args:
        tile_shapes: Contains the shapely objects corresponding to transformed image outlines.
                    Each shape in tile_shapes must have a .fuse_info dictionary with
                    keys "file" and "transform".
        chunk_shapes: Contains the shapely objects representing dask array chunks.
                    Each shape in chunk_shapes must have a .fuse_info dictionary with
                    key "chunk_boundary", containing a tuple of chunk boundaries

    Returns:
         The chunk_to_tiles dictionary, which has the chunk anchor points as keys and tuples of
         image file names and their corresponding affine transform matrix as values.
    """
    chunk_to_tiles = {}
    tile_tree = STRtree(tiles_shapely)

    for chunk_shape in chunks_shapely:
        chunk_boundary = chunk_shape.fuse_info["chunk_boundary"]
        anchor_point = (chunk_boundary[0][0], chunk_boundary[1][0])
        intersecting_tiles = tile_tree.query(chunk_shape)
        chunk_to_tiles[anchor_point] = [
            ((t.fuse_info["file"], t.fuse_info["transform"]))
            for t in intersecting_tiles
        ]
    return chunk_to_tiles
