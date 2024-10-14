# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Dataset for Location-based species presence prediction."""

from collections.abc import Callable, Sequence
from typing import Any

from rasterio.crs import CRS

from torchgeo.datasets.geo import PointDataset


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

    date_format = "%j-%Y"
    res = 0
    crs = CRS.from_epsg(4326)  # Lat/Lon
    all_metadata_columns = [
        "Id",
        "lat",
        "lon",
        "dayOfYear",
        "year",
        "glcID",
        "gbifID",
        "observer",
        "datasetName",
        "geoUncertaintyInM",
        "speciesId",
        "patchID",
        "timeSerieID",
    ]

    def __init__(
        self,
        paths: str = "data",
        crs: CRS | None = None,
        res: float = 0,
        metadata_columns: Sequence[str] | None = ["speciesId"],
        location_columns: Sequence[str] = ["lon", "lat"],
        time_columns: Sequence[str] = ["dayOfYear", "year"],
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        cache: bool = False,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            paths: one or more root directories to search or files to load
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            metadata_columns: columns with label information to return (defaults to ["speciesId"])
            location_columns: columns with location information (defaults to ["lon", "lat"])
            time_columns: columns with time information (defaults to ["dayOfYear", "year"])
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
        """
        if metadata_columns is not None:
            assert len(metadata_columns) == len(
                set(metadata_columns)
            ), "Metadata columns must be unique"
            for metadata_column in metadata_columns:
                assert (
                    metadata_column in self.all_metadata_columns
                ), f"Metadata column {metadata_column} not found in dataset"

        assert len(location_columns) == len(
            set(location_columns)
        ), "Location columns must be unique"
        for location_column in location_columns:
            assert (
                location_column in self.all_metadata_columns
            ), f"Location column {location_column} not found in dataset"

        assert len(time_columns) == len(
            set(time_columns)
        ), "Time columns must be unique"
        for time_column in time_columns:
            assert (
                time_column in self.all_metadata_columns
            ), f"Time column {time_column} not found in dataset"

        super().__init__(
            paths,
            crs,
            res,
            metadata_columns,
            location_columns,
            time_columns,
            transforms,
            cache,
        )
