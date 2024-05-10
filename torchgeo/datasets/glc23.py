# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Dataset for Location-based species presence prediction."""

import glob
import os
import sys
from collections.abc import Callable
from typing import Any

import torch



import numpy as np
import pandas as pd
from rasterio.crs import CRS

from torchgeo.datasets.geo import PointDataset
from torchgeo.datasets.utils import (
    BoundingBox,
    DatasetNotFoundError,
    disambiguate_timestamp,
    sum_samples
)

import math


class GLC23(PointDataset):
    """GeoLifeCLef2023 species observations.
    
    The `GeoLifeClef2023 <https://www.imageclef.org/GeoLifeCLEF2023>` dataset is a collection of species observations and environmental rasters in Europe for biodiversity monitoring. This dataset comprises a subset only covering point wise species occurence data.

    Dataset features:
    * Presence-Only Data retreived from GBIF covering the whole EU territory
    * Presence-Absence Surveys in areas of 10 to 400 m²

    Dataset format:
    * CSV files with location and species ID

    Dataset classes:
    * 10040 species IDs

    """

    res = 1 
    _crs = CRS.from_epsg(4326)  # Lat/Lon

    def __init__(
        self,
        root: str = "data",
        single_instance: bool = False,
        centered: bool = False,
        prediction: bool = False,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found

        Raises:
            DatasetNotFoundError: If dataset is not found.
        """

        self.root = root
        self.prediction = prediction

        files = glob.glob(os.path.join(root, '**.csv'))
        if not files:
            raise DatasetNotFoundError(self)
        
        # receptive field
        self.single_instance = single_instance
        self.centered = centered

        self.transforms = transforms

        # Read tab-delimited CSV file --> # TODO: remove this and replace with pre-processing option (load precomputed r-tree)
        columns = ["lat", "lon", "dayOfYear", "year"]
        if not prediction:
            columns.append("speciesId")
        data = pd.read_table(
            files[0],
            engine="c",
            usecols=columns,
            sep=";",
            nrows=1000,  
        )
        if not prediction: 
            data = (
                data.groupby(["dayOfYear", "lat", "lon"])
                .agg({"speciesId": lambda x: list(x), "year": "first"})
                .reset_index()
            )
        self.data = data[columns]

        super().__init__()

        

        

    
