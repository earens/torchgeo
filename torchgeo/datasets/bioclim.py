# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Bioclim Rasters."""

from collections.abc import Callable, Sequence
from typing import Any
import glob
import re
import os

from rasterio.crs import CRS

from torchgeo.datasets.geo import RasterDataset

from torchgeo.datasets.utils import tile_tif




class Bioclim(RasterDataset):
    r"""Bioclimatic variables.
    TODO: add more details about the dataset
    """

    filename_glob = "*bio*.*"

    filename_regex = r"""
        (?P<type>(\d+)?\w{3})
        (?P<band>(\d+|_all)) 
        (?P<tile>_\d+)? #optional tile number
        \.
    """

    date_format = "%Y"
    all_bands = [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
    ]
    separate_files = True

    def __init__(
        self,
        paths: str | list[str] = "data",
        crs: CRS | None = None,
        res: float | None = None,
        bands: Sequence[str] = None,
        insert_band: str = None,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        cache: bool = True,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            paths: one or more root directories to search or files to load
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            bands: bands to return (defaults to ["VV", "VH"])
            tile: boolean flag to indicate if the dataset should be tiled
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling

        Raises:
            AssertionError: if ``bands`` is invalid
            DatasetNotFoundError: If dataset is not found.

        .. versionchanged:: 0.5
           *root* was renamed to *paths*.
        """
        bands = bands or self.all_bands
        assert len(bands) > 0, "'bands' cannot be an empty list"
        assert len(bands) == len(set(bands)), "'bands' contains duplicate bands"

        for band in bands:
            assert band in self.all_bands, f"invalid band '{band}'"

        self.filename_glob = self.filename_glob.format(bands[0])

        super().__init__(paths, crs, res, bands, insert_band, transforms, cache)
