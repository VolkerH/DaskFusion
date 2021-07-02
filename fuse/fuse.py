from typing import Dict
import numpy as np
from skimage.transform import AffineTransform
from scipy.ndimage import affine_transform
from typing import Dict, Tuple, List, Callable, Union, Optional
from pathlib import Path
from utils.imutils import load_image, transpose, select_channel, crop_black_border
from skimage.io import imread

def fuse_func(input_tile_info: Dict[Tuple[int, int], List[Tuple[Union[str, np.ndarray], np.ndarray]]],
         imload_func: Optional[Callable] = imread,
         block_info = None, 
         dtype = np.uint16,
         ) -> np.ndarray:
    
    """Fuses the tiles that intersect the current chunk of a dask array
    using maximum projection.
    
    Pass this function to dask.array.map_blocks,
    after partial evaluation of the required image_folder
    and (if needed) optional arguments.

    
    Returns:
        np.ndarray: array of chunk-shape containing max projection of 
                    tiles falling into chunk
    """
    array_location = block_info[None]['array-location']
    # the anchor point is the key to the input_tile_info dictionary
    anchor_point = (array_location[0][0], array_location[1][0])
    chunk_shape = block_info[None]['chunk-shape']
    tiles_info = input_tile_info[anchor_point]
    print(f"Processing chunk at {anchor_point}")
    fused = np.zeros(chunk_shape, dtype=dtype)
    for image_representation, tile_affine in tiles_info:
        if imload_func is not None: 
            # when imload_func is provided we assume we
            # have been given strings representing files
            tile_path = image_representation
            im = imload_func(tile_path)
        else:
            # without imload function we assume images are passed
            im = image_representation
        shift = AffineTransform(translation=(-anchor_point[0], -anchor_point[1]))
        tile_shifted = affine_transform(im, matrix=np.linalg.inv(tile_affine@shift.params), output_shape=chunk_shape)
        stack = np.stack([fused,tile_shifted])
        fused = np.max(stack,axis=0)
    return fused
