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

    res = 0  # TODO: check if this breaks something
    _crs = CRS.from_epsg(4326)  # Lat/Lon


    def __init__(
        self,
        root: str = "data",
        single_instance: bool = False,
        centered: bool = False,
        prediction: bool = False,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        collate_fn: Callable[[list[dict[str, Any]]], dict[str, Any]] = sum_samples,
    ) -> None:
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
        self.collate_fn = collate_fn


        files = root
        if not files:
            raise DatasetNotFoundError(self)
        
        
        columns = ["lat", "lon", "dayOfYear", "year"]
        if not prediction:
            columns.append("speciesId")
        

        # Read tab-delimited CSV file
        data = pd.read_table(
            files,
            engine="c",
            usecols=columns,
            sep=";",
            nrows=1000,  # TODO: remove this and replace with pre-processing option (load precomputed r-tree)
        )
        if not prediction: 
            data = (
                data.groupby(["dayOfYear", "lat", "lon"])
                .agg({"speciesId": lambda x: list(x), "year": "first"})
                .reset_index()
            )  # TODO: remove this and replace with pre-processing option (group and write to file)
        data = data[columns]
        

        # Convert from pandas DataFrame to rtree Index
        i = 0
        for y,x,dayOfYear,year, *speciesId in data.itertuples(index=False, name=None):
          
            if not prediction:
                speciesId = speciesId[0]
            else:
                speciesId = []
                
            if np.isnan(y) or np.isnan(x):
                continue

            if not pd.isna(dayOfYear) and not pd.isna(year):
                mint, maxt = disambiguate_timestamp(
                    "-".join((str(dayOfYear), str(year))), "%j-%Y"
                )
            else:
                mint, maxt = 0, sys.maxsize

            coords = (x, x, y, y, mint, maxt)
            self.index.insert(
                i, coords, speciesId
            )  # insert into r-tree with speciesId as object
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
        # otherwise retrieve all points in the box
        else:
            hits = list(self.index.intersection(tuple(query), objects=True))

        # allow nodata returns (base case)
        if len(hits) == 0:
            bboxes, labels, location = [], [], []
        else:
            bboxes, labels = map(list,zip(*[(hit.bounds, hit.object) for hit in hits]))
            bboxes = [BoundingBox(*bbox) for bbox in bboxes]

            # if single instance, pick random location
            if query.area > 0 and self.single_instance and not self.centered:
                idx = np.random.randint(0, len(bboxes))
                bboxes = [bboxes[idx]]
                labels = [labels[idx]]

            location = (query.center.minx, query.center.miny) #for multiple samples, location defaults to center of bbox
            location = torch.tensor(location, dtype=torch.float32)
        
            #one hot encode labels
            labels = [torch.nn.functional.one_hot(torch.tensor(label), num_classes=10040) for label in labels]
            labels = [(torch.sum(label, dim=0)>0).double() for label in labels if len(label) > 0]

            if query.area > 0 and not self.single_instance and not self.centered:
                #collate multiple labels
                labels = self.collate_fn(labels)
            else:
                labels = labels[0]


            # location transforms
            if self.transforms is not None:                
                location = self.transforms(location)

        # return not only metadata but also encoded location and label (speciesId)
        sample = {
            "crs": self.crs,
            "bbox": bboxes,
            "label": labels,
            "location": location,
        }
        return sample
