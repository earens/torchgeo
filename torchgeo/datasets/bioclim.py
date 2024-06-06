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

    filename_glob = "bio*.*"

    filename_tiled_glob = "bio*_*_*.*"  

    # filename_regex = r"""
    #    (?P<type>\w+)
    #    (?P<band>\d{1}|\d{2})
    #    _(?P<start>\d{4})
    #    _(?P<stop>\d{4})
    #    \.
    # """

    filename_regex = r"""
        (?P<type>\w{3})
        (?P<band>\d+)
        \.
    """

    filename_tiled_regex = r"""
        (?P<type>\w{3})
        (?P<band>\d+)
        _(?P<tile_x>\d+)
        _(?P<tile_y>\d+)
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
        tile: bool = False,
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
       

        #check if data is tiled
        if tile:
            filename_tiled_regex = re.compile(self.filename_tiled_regex, re.VERBOSE)
            matches = [re.match(filename_tiled_regex, os.path.basename(filepath)) for filepath in glob.glob(f"{paths[0]}*.tif")]
            #if all None, then the data is not tiled
            if all(match is None for match in matches):
                #tile the data
                for path in paths:
                    tile_tif(path,path, 1024)
            
            self.filename_regex = self.filename_tiled_regex
            self.filename_glob = self.filename_tiled_glob
        

        """
        #check if the file name is in the correct format, if not rename the file
        filename_regex = re.compile(self.filename_regex, re.VERBOSE)
        self.paths = paths
        for i, filepath in enumerate(self.files):
     
            match = re.match(filename_regex, os.path.basename(filepath))
            #if match none, rename the file
            if match is None:
                #rename the file
                new_file_name = os.path.basename(filepath)[:-4]+ "_1981_2010.tif" #add the correct time range
                os.rename(filepath, os.path.join(os.path.dirname(filepath), new_file_name))  
                self.paths[i] = os.path.join(os.path.dirname(filepath), new_file_name)

        """

        super().__init__(paths, crs, res, bands, transforms, cache)
