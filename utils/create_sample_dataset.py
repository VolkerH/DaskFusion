# script that was used to convert the original dataset consisting of tiff files
# to  compressed .jpg files, in order to crete a sample dataset that can easily
# be shared (for my own reference)

from functools import partial
from pathlib import Path

from skimage import img_as_ubyte
from skimage.io import imread, imsave

src = "~/Desktop/W2"
dst = "~/Desktop/Stiching_Example_Dataset"


def convert(filepath: Path, outfolder: Path):
    print(f"converting {filepath}")

    im = imread(filepath)
    print(im.shape)
    outpath = outfolder / (filepath.stem + ".jpg")

    imsave(str(outpath), img_as_ubyte(im), quality=30)


inputfiles = Path(src).expanduser().glob("*.tif")
_convert = partial(convert, outfolder=Path(dst))
list(map(_convert, inputfiles))
