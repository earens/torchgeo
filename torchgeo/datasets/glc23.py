# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Dataset for Location-based species presence prediction."""

import glob
import os
import sys
from datetime import datetime, timedelta
from collections.abc import Callable
from typing import Any, cast, Dict

import numpy as np
import pandas as pd
from rasterio.crs import CRS

from torchgeo.datasets.geo import GeoDataset
from torchgeo.datasets.geo import PointDataset
from torchgeo.datasets.utils import  BoundingBox, DatasetNotFoundError, disambiguate_timestamp


class GLC23(PointDataset):
    """Dataset for Location-based species presence prediction.

    TODO: add more details about the dataset
    """

    res = 0 # TODO: check if this breaks something
    _crs = CRS.from_epsg(4326)  # Lat/Lon

    def __init__(
            self, root: str = "data", 
            single_instance: bool = False,
            centered: bool = False,
            transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found

        Raises:
            DatasetNotFoundError: If dataset is not found.
        """
        super().__init__()

        self.root = root

        self.single_instance = single_instance
        self.centered = centered

        self.transforms = transforms


        files = glob.glob(os.path.join(root, "**.csv"))
        if not files:
            raise DatasetNotFoundError(self)

        # Read tab-delimited CSV file
        data = pd.read_table(
            files[0],
            engine="c",
            usecols=["lat", "lon", "dayOfYear", "year","speciesId"],
            sep=";",
            nrows=10000, #TODO: remove this and replace with pre-processing option (load precomputed r-tree)
        )
        data = data[["lat", "lon", "dayOfYear", "year","speciesId"]]

        # Convert from pandas DataFrame to rtree Index
        i = 0
        for y, x, dayOfYear, year, speciesId in data.itertuples(index=False, name=None):
            # Skip rows without lat/lon
            if np.isnan(y) or np.isnan(x):
                continue

            if not pd.isna(dayOfYear) and not pd.isna(year):
                mint, maxt = disambiguate_timestamp("-".join((str(dayOfYear), str(year))), "%j-%Y")
            else:
                mint, maxt = 0, sys.maxsize

            coords = (x, x, y, y, mint, maxt)
            self.index.insert(i, coords,speciesId)  #insert into r-tree with speciesId as object
            i += 1

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        """Retrieve metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        
        # if center only look for one point near center 
        if query.area > 0 and self.single_instance and self.centered:
            center = query.center
            hits = list(self.index.nearest(tuple(center), 1, objects=True))   
        #otherwise retrieve all points in the box
        else:
            hits = list(self.index.intersection(tuple(query), objects=True))

        #allow nodata returns (base case)
        if len(hits) == 0:
            bboxes, labels, location = [], [], []
        else:
            bboxes, labels = zip(*[(hit.bounds, hit.object) for hit in hits])

            # group labels by bbox
            unique_bboxes = set(tuple(b) for b in bboxes)
            grouped_labels = [[labels[i] for i, b in enumerate(bboxes) if tuple(b) == bbox] for bbox in unique_bboxes]

            #if single instance, pick random location
            if query.area > 0 and self.single_instance and not self.centered:
                #pick random point from the list
                idx = np.random.randint(0, len(unique_bboxes))
                bboxes = [list(unique_bboxes)[idx]]
                labels = [grouped_labels[idx]]
            else:
                bboxes = list(unique_bboxes)
                labels = grouped_labels


            #location transforms
            if self.transforms is not None:
                location = [np.mean(np.array(bboxes)[:,2:4], axis=1), np.mean(np.array(bboxes)[:,0:2],axis=1)]
                location = self.transforms(location)

        #return not only metadata but also encoded location and label (speciesId)
        sample = {"crs": self.crs, "res": self.res, "bbox": bboxes, "label": labels, "location": location}

        return sample
    

    