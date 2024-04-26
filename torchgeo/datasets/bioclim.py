# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Bioclim Rasters."""

from collections.abc import Callable, Iterable, Sequence
from typing import Any, Optional

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from rasterio.crs import CRS

from torchgeo.datasets.geo import RasterDataset, GeoDataset
from torchgeo.datasets.utils import RGBBandsMissingError

from src.data.datasets.geo import AdaptedIntersectionDataset

 

import re
import os



class Bioclim(RasterDataset):
    r"""Bioclimatic variables.
    TODO: add more details about the dataset
    """

    filename_glob = "bio*{}.*"

    #filename_regex = r"""
    #    (?P<type>\w+)
    #    (?P<band>\d{1}|\d{2})
    #    _(?P<start>\d{4})
    #    _(?P<stop>\d{4})
    #    \.
    #"""

    filename_regex = r"""
        (?P<type>\w{3})
        (?P<band>\d+)
        \.
    """
    
    date_format = "%Y"
    all_bands = ["1", "2", "3", "4","5","6","7","8","9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19"]
    separate_files = True

    def __init__(
        self,
        paths: str | list[str] = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        bands: Sequence[str] = None,
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

        '''
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

        '''
              
        super().__init__(paths, crs, res, bands, transforms, cache)

    

