from typing import List, Tuple

import numpy as np
from dask.array.core import normalize_chunks


def get_chunk_coordinates(
    shape: Tuple[int, int], chunk_size: Tuple[int, int], return_np_slice: bool = False
):
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
            if return_np_slice:
                yield np.s_[y : y + cy, x : x + cx]
            else:
                yield ((y, y + cy), (x, x + cx))
            x = x + cx
        y = y + cy


def get_rect_from_chunk_slice(chunk_slice):
    """given a chunk slice tuple, return a numpy
    array that can be added as a shape to napari"
    """
    ylim, xlim = chunk_slice
    miny, maxy = ylim[0], ylim[1] - 1
    minx, maxx = xlim[0], xlim[1] - 1
    return np.array([[miny, minx], [maxy, minx], [maxy, maxx], [miny, maxx]])


def tile_chunk_intersections(
    mosaic_shifted: "shapely.geometry.GeometryCollection",
    files: List[str],
    transforms: List[np.ndarray],
    chunks_shapely,
    chunk_slices,
):
    """
    Finds intersections between image tiles and chunks

    mosaic_shifted: contains the shapely objects corresponding to transformed images
    mosaic_layers: the napari layers with the images corresponding to the tiles (used to get the names)
    chunks_shapely: contains the shapely objects representing dask array chunk coordinate
    chunk_slices: chunk coordinate objects

    returns the chunk_tiles dictionary, which has the chunk anchor point tuples as keys
    and tuples of image filenames and corresponding transform as values.
    """
    chunk_tiles = {}
    for i, tile in enumerate(mosaic_shifted):
        for j, (chunk, chunk_slice) in enumerate(zip(chunks_shapely, chunk_slices)):
            anchor_point = (chunk_slice[0][0], chunk_slice[1][0])
            if anchor_point not in chunk_tiles.keys():
                chunk_tiles[anchor_point] = []
            if tile.intersects(chunk):
                chunk_tiles[anchor_point].append((files[i], transforms[i]))
    return chunk_tiles
