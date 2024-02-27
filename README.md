[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10715850.svg)](https://doi.org/10.5281/zenodo.10715850)


# DaskFusion

This repo contains proof-of-concept code that fuses many image tiles from a microscopy scan,
where the position of each tile is known from the microscopy stage metadat, into 
a large "fused" array. 

This is achieved using the `map_blocks` functionality from `dask.array` which takes care of
out-of-core processing, i.e. we can generate an array of mosaiced tiles that is much larger
than what we can hold in RAM on a particular machine. 

The notebook [`DaskFusion_Example.ipynb`](./DaskFusion_Example.ipynb) explains the approach.
We use `shapely` to visualize the locations of the tiles in the notebook and also to find
tiles that overlap the individual blocks (chunks) that we process.

In principle, this approach can be used to generate very large photomosaics, from collections
of image tiles with known affine or even perspective transforms.

Caveats:

* Regions with multiple ovberlapping tiles are simply merged using a maximum projection. In practice, you would want to amend the fuse function with a smooth blending method.
* We assume the affine transforms are known. We disregard fine tuning the transformations to get pixel-perfect registration. Situations where the transformations are not known or require fine-tuning can be handled for exampl by SIFT-feature point matching between overlapping tiles and calculation of the transformations from feature point correspondences.
* This is not production-quality code.
 
# Additional reading and related work

We wrote up a small blog post on this proof of concept for the [Dask Blog](https://blog.dask.org/2021/12/01/mosaic-fusion).

Some [feedback on the blog post](https://twitter.com/TobiasAdeJong/status/1466719789280382977) by [Tobias deJong](https://github.com/TAdeJong) points out a very similar approach that allows incorporates optimization of tile positions, [see this notebook](https://github.com/TAdeJong/LEEM-analysis/blob/master/6%20-%20Stitching.ipynb). 

# Citation information

See: https://doi.org/10.5281/zenodo.10715850 for citation information
