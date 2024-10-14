# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Dataset for iNaturalist."""

from collections.abc import Callable, Sequence
from typing import Any

from rasterio.crs import CRS

from .geo import GeoDataset


class INaturalist(GeoDataset):
    """Dataset for iNaturalist.

    `iNaturalist <https://www.inaturalist.org/>`__ is a joint initiative of the
    California Academy of Sciences and the National Geographic Society. It allows
    citizen scientists to upload observations of organisms that can be downloaded by
    scientists and researchers.

    If you use an iNaturalist dataset in your research, please cite it according to:

    * https://www.inaturalist.org/pages/help#cite

    .. versionadded:: 0.3
    """

    date_format = "%Y-%m-%d"
    res = 0
    _crs = CRS.from_epsg(4326)  # Lat/Lon
    all_metadata_columns = [
        "observation_uuid",
        "observer_id",
        "positional_accuracy",
        "taxon_id",
        "quality_grade",
    ]

    def __init__(
        self,
        paths: str = "data",
        crs: CRS | None = None,
        res: float = 0,
        metadata_columns: Sequence[str] | None = ["taxon_id"],
        location_columns: Sequence[str] = ["longitude", "latitude"],
        time_columns: Sequence[str] = ["observed_on"],
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        cache: bool = False,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            paths: Path to one or more CSV files.
            crs: The CRS of the dataset.
            res: The resolution of the dataset.
            metadata_columns: The metadata columns to include.
            location_columns: The location columns to include.
            time_columns: The time columns to include.
            transforms: A function/transform that takes in a sample and returns a transformed version.
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
