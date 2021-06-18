import codecs
import re
from pathlib import Path
from typing import Tuple, Union

import chardet
import pandas as pd


def _detect_encoding(path: Union[str, Path]) -> str:
    with open(path, "rb") as txt_file:
        return chardet.detect(txt_file.read())["encoding"]


def extract_coordinates(metadata_file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Extract coordinates and scale calibration from Nikon .txt metadata file.
    """

    with codecs.open(
        metadata_file_path, "r", encoding=_detect_encoding(metadata_file_path)
    ) as f:
        text = f.read()

    # Regular expression to extract points
    # https://regex101.com/r/iGNxWz/1
    # TODO: maybe use an additional regex to make sure we are only applying
    #  this in the points region
    cre_stageposition = re.compile(
        r"(?P<name>^[#0-9_]+)\s+(?P<X>[-0-9,.]+)\s+(?P<Y>[-0-9,.]+)\s+(?P<Z>[-0-9,.]+)\s",  # noqa: E501
        re.MULTILINE,
    )
    matchdict = [m.groupdict() for m in cre_stageposition.finditer(text)]
    coords = pd.DataFrame(matchdict)

    # Settings on the Nikon computer seem to be German, resulting in commas in
    # the stage coordinates. We need to fix this.
    for col in ("X", "Y", "Z"):
        coords[col] = coords[col].apply(lambda s: float(s.replace(",", ".")))

    # Regular expression to extract calibration:
    # https://regex101.com/r/Tf1MIh/1
    cre_calibration = re.compile(
        r"Calibration \(µm/px\):\s(?P<um_per_pix>[0-9,.]+)"
    )
    _ = cre_calibration.findall(text)
    calibration = float(_[0].replace(",", "."))

    # although it duplicates the calibration for each image
    # we can afford an extra column that will simplify things
    # later
    coords["um/px"] = calibration

    return coords


def normalize_coords_to_pixel(
    coords: pd.DataFrame,
    xy_keys: Tuple[str, str] = ("X", "Y"),
    conversion_factor_key: str = "um/px",
    relative_to_first: bool = False,
) -> pd.DataFrame:
    """
    Normalizes stage coordinates to pixels.
    """

    if relative_to_first:
        origin = coords[list(xy_keys)].iloc[0]
    else:
        origin = [0, 0]
    offset_coords = coords[list(xy_keys)] - origin
    px_offset_coords = offset_coords.div(coords[conversion_factor_key], axis=0)

    return px_offset_coords
