#  batch convert to .jpg, to create a small sample dataset that can easily be shared

from pathlib import Path
from functools import partial
from skimage import img_as_ubyte
from skimage.io import imread, imsave

src = "~/Desktop/W2"
dst = "~/Desktop/Stiching_Example_Dataset"


def convert(filepath: Path, outfolder: Path):
    print(f"converting {filepath}")
    
    im = imread(filepath)
    print(im.shape)
    outpath = outfolder / (filepath.stem + ".jpg") 
    
    imsave(str(outpath), img_as_ubyte(im), quality = 30)

inputfiles = Path(src).expanduser().glob("*.tif")
_convert = partial(convert, outfolder=Path(dst))
list(map(_convert, inputfiles))