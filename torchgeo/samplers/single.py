# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo samplers."""

import abc
from collections.abc import Callable, Iterable, Iterator
import time

import torch
from rtree.index import Index, Property
from torch.utils.data import Sampler

import rasterio
import numpy as np

from ..datasets import BoundingBox, GeoDataset
from .constants import Units
from .utils import _to_tuple, get_random_bounding_box, tile_to_chips, prepare_complex_roi, extract_valid_bboxes, image_coordinates_to_geocoordinates, extract_valid_tiles, check_tile_validity


class GeoSampler(Sampler[BoundingBox], abc.ABC):
    """Abstract base class for sampling from :class:`~torchgeo.datasets.GeoDataset`.

    Unlike PyTorch's :class:`~torch.utils.data.Sampler`, :class:`GeoSampler`
    returns enough geospatial information to uniquely index any
    :class:`~torchgeo.datasets.GeoDataset`. This includes things like latitude,
    longitude, height, width, projection, coordinate system, and time.
    """

    def __init__(self, dataset: GeoDataset, roi: BoundingBox | None = None) -> None:
        """Initialize a new Sampler instance.

        Args:
            dataset: dataset to index from
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
        """
        if roi is None:
            self.index = dataset.index
            roi = BoundingBox(*self.index.bounds)
        else:
            self.index = Index(interleaved=False, properties=Property(dimension=3))
            hits = dataset.index.intersection(tuple(roi), objects=True)
            for hit in hits:
                bbox = BoundingBox(*hit.bounds) & roi
                self.index.insert(hit.id, tuple(bbox), hit.object)

        self.res = dataset.res
        self.roi = roi

    @abc.abstractmethod
    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """


class RandomGeoSampler(GeoSampler):
    """Samples elements from a region of interest randomly.

    This is particularly useful during training when you want to maximize the size of
    the dataset and return as many random :term:`chips <chip>` as possible. Note that
    randomly sampled chips may overlap.

    This sampler is not recommended for use with tile-based datasets. Use
    :class:`RandomBatchGeoSampler` instead.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        size: tuple[float, float] | float,
        length: int | None = None,
        roi: BoundingBox | None = None,
        units: Units = Units.PIXELS,
    ) -> None:
        """Initialize a new Sampler instance.

        The ``size`` argument can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

        .. versionchanged:: 0.3
           Added ``units`` parameter, changed default to pixel units

        .. versionchanged:: 0.4
           ``length`` parameter is now optional, a reasonable default will be used

        Args:
            dataset: dataset to index from
            size: dimensions of each :term:`patch`
            length: number of random samples to draw per epoch
                (defaults to approximately the maximal number of non-overlapping
                :term:`chips <chip>` of size ``size`` that could be sampled from
                the dataset)
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
            units: defines if ``size`` is in pixel or CRS units
        """
        super().__init__(dataset, roi)
        self.size = _to_tuple(size)

        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res, self.size[1] * self.res)

        self.length = 0
        self.hits = []
        areas = []
        for hit in self.index.intersection(tuple(self.roi), objects=True):
            bounds = BoundingBox(*hit.bounds)
            if (
                bounds.maxx - bounds.minx >= self.size[1]
                and bounds.maxy - bounds.miny >= self.size[0]
            ):
                if bounds.area > 0:
                    rows, cols = tile_to_chips(bounds, self.size)
                    self.length += rows * cols
                else:
                    self.length += 1
                self.hits.append(hit)
                areas.append(bounds.area)
        if length is not None:
            self.length = length

        # torch.multinomial requires float probabilities > 0
        self.areas = torch.tensor(areas, dtype=torch.float)
        if torch.sum(self.areas) == 0:
            self.areas += 1

    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        for _ in range(len(self)):
            # Choose a random tile, weighted by area
            idx = torch.multinomial(self.areas, 1)
            hit = self.hits[idx]
            bounds = BoundingBox(*hit.bounds)

            # Choose a random index within that tile
            bounding_box = get_random_bounding_box(bounds, self.size, self.res)

            yield bounding_box

    def __len__(self) -> int:
        """Return the number of samples in a single epoch.

        Returns:
            length of the epoch
        """
        return self.length
    
class ExtendedRandomGeoSampler(GeoSampler):
    """Samples elements from a region of interest randomly.

    This is particularly useful during training when you want to maximize the size of
    the dataset and return as many random :term:`chips <chip>` as possible. Note that
    randomly sampled chips may overlap.

    This sampler is not recommended for use with tile-based datasets. Use
    :class:`RandomBatchGeoSampler` instead.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        size: tuple[float, float] | float,
        length: int | None = None,
        roi: BoundingBox | None = None,
        complex_roi: str | None = None,
        units: Units = Units.PIXELS,
        resample: int = 1,
        thresh: float = 0.8,
    ) -> None:
        """Initialize a new Sampler instance.

        The ``size`` argument can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

        .. versionchanged:: 0.3
           Added ``units`` parameter, changed default to pixel units

        .. versionchanged:: 0.4
           ``length`` parameter is now optional, a reasonable default will be used

        Args:
            dataset: dataset to index from
            size: dimensions of each :term:`patch`
            length: number of random samples to draw per epoch
                (defaults to approximately the maximal number of non-overlapping
                :term:`chips <chip>` of size ``size`` that could be sampled from
                the dataset)
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
            units: defines if ``size`` is in pixel or CRS units
        """
        super().__init__(dataset, roi)
        self.size = _to_tuple(size)
        self.resample = resample
        self.thresh = thresh
        self.units = units
        self.dataset = dataset

        if complex_roi is not None:
            self.complex_roi = rasterio.open(complex_roi)

        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res, self.size[1] * self.res)

        self.length = 0
        self.hits = []
        areas = []
        validity = True
        extraction_size = (int(self.size[0]/self.res),int(self.size[1]/self.res) ) if self.units == Units.PIXELS else (int(self.size[0]),int(self.size[1]))
        
        start = time.time()
        if complex_roi is not None:
            #find indices of duplicate bounds in hits
            hits = [hit for hit in self.index.intersection(tuple(self.roi), objects=True)]
            print(len(hits))
            bounds = [BoundingBox(*hit.bounds) for hit in hits]
            unique_bounds = list(set(bounds))

            #for each bound, if there are duplicates, remove all but the first
            for bound in unique_bounds:
                #get all indices of the bound
                indices = [j for j, b in enumerate(bounds) if b == bound]
                #if there are duplicates, remove all but the first
                if len(indices) > 1:
                    for idx in sorted(indices[1:], reverse=True):

                        hits.pop(idx)
                        bounds.pop(idx)
            validities = [check_tile_validity(self.complex_roi, dataset, BoundingBox(*hit.bounds), extraction_size, self.thresh) for hit in hits]

        print("Time taken to check validity: ", time.time()-start)

        for hit in self.index.intersection(tuple(self.roi), objects=True):
            bounds = BoundingBox(*hit.bounds)
            

            if complex_roi is not None:
                
                #find bound in unique bounds
                idx = unique_bounds.index(bounds)
                validity = validities[idx]

            if validity:
                if (
                    bounds.maxx - bounds.minx >= self.size[1]
                    and bounds.maxy - bounds.miny >= self.size[0]
                ):
                    if bounds.area > 0:
                        rows, cols = tile_to_chips(bounds, self.size)
                        self.length += rows * cols
                    else:
                        self.length += 1
                    self.hits.append(hit)
                    areas.append(bounds.area)
  
        if length is not None:
            self.length = length

        # torch.multinomial requires float probabilities > 0
        self.areas = torch.tensor(areas, dtype=torch.float)
        if torch.sum(self.areas) == 0:
            self.areas += 1

    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        print(len(self.areas))
        print(len(self.hits))
        for _ in range(len(self)):

            valid_indices = []

            while len(valid_indices) == 0:

                # Choose a random tile, weighted by area
                
                idx = torch.multinomial(self.areas, 1)
                hit = self.hits[idx]
                bounds = BoundingBox(*hit.bounds)

                if self.complex_roi is None:
                    # Choose a random index within that tile
                    bounding_box = get_random_bounding_box(bounds, self.size, self.res)
                    valid_indices = [0]

                    yield bounding_box
                
                else:

                    extraction_size = (int(self.size[0]/self.res),int(self.size[1]/self.res) ) if self.units == Units.PIXELS else (int(self.size[0]),int(self.size[1]))
                    valid_indices = extract_valid_bboxes(self.complex_roi, self.dataset, bounds,extraction_size, resample=self.resample,tolerance = self.thresh)
                    if len(valid_indices) > 0:
                        idx = torch.randint(0, len(valid_indices), (1,)).item()
                        bounding_box = image_coordinates_to_geocoordinates(valid_indices[idx][1], valid_indices[idx][0], self, bounds, extraction_size)
                        yield bounding_box
                    else:
                        print("No valid indices found, removing tile from list")
                        #remove all hits with the same bounds
                        indices = [i for i, hit in enumerate(self.hits) if BoundingBox(*hit.bounds) == bounds]
                        for idx in sorted(indices, reverse=True):
                            self.hits.pop(idx)
                        self.areas = torch.tensor([area for i, area in enumerate(self.areas) if i not in indices], dtype=torch.float)


            yield bounding_box

    def __len__(self) -> int:
        """Return the number of samples in a single epoch.

        Returns:
            length of the epoch
        """
        return self.length


class RandomGeoPointSampler(GeoSampler):
    """Extension of the RandomGeoSampler class to  sample around points"""

    def __init__(
        self,
        scene_dataset: GeoDataset,
        point_dataset: Index,
        size: tuple[float, float] | float,
        length: int | None = None,
        roi: BoundingBox | None = None,
        complex_roi: str | None = None,
        units: Units = Units.PIXELS,
        centered: bool = False,
        resample: int = 1,
        thresh: float = 0.8,
    ) -> None:
        super().__init__(point_dataset, roi)

        self.res = scene_dataset.res
        self.scene_dataset = scene_dataset

        self.size = _to_tuple(size)
        self.centered = centered
        self.resample = resample
        self.thresh = thresh
        self.units = units

        # bbox size wrt res and pixel size
        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res, self.size[1] * self.res)

        # list all points that can potentially be sampled
        self.hits = list(self.index.intersection(self.index.bounds, objects=True))

        # filter out points that are not within the complex ROI
        if complex_roi is not None: 
            self.complex_roi = rasterio.open(complex_roi)

            valid_indices = extract_valid_tiles(self.complex_roi, self, self.centered, self.hits, int(self.size[0]//self.res), self.thresh)
            self.hits = [self.hits[i] for i in valid_indices]

        if length is not None:
            self.length = length
        else:
            self.length = len(self.hits)

        
    def __len__(self) -> int:
        """Return the number of samples in a single epoch.

        Returns:
            length of the epoch
        """
        return self.length

    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset

        """
        
        for _ in range(len(self)):

            valid_indices = []

            while len(valid_indices) == 0:
    
                idx = torch.randint(0, len(self.hits), (1,)).item()
                hit = self.hits[idx]

                bounds = BoundingBox(*hit.bounds)

                extent = 2 if self.centered else 1


                bounds = BoundingBox(
                    bounds.minx - ((self.size[1] / extent)),
                    bounds.maxx + ((self.size[1] / extent)),
                    bounds.miny - ((self.size[0] / extent)),
                    bounds.maxy + ((self.size[0] / extent)),
                    bounds.mint,
                    bounds.maxt,
                )


                if self.complex_roi is None or self.centered:
                    bounding_box = get_random_bounding_box(bounds, self.size, self.res)
                    valid_indices = [0]

                    yield bounding_box

                else: 
                    extraction_size = (int(self.size[0]/self.res),int(self.size[1]/self.res) ) if self.units == Units.PIXELS else (int(self.size[0]),int(self.size[1]))
                    valid_indices = extract_valid_bboxes(self.complex_roi, self, bounds,extraction_size, resample=self.resample,tolerance = self.thresh)
                    if len(valid_indices) > 0:
                        idx = torch.randint(0, len(valid_indices), (1,)).item()
                        bounding_box = image_coordinates_to_geocoordinates(valid_indices[idx][1], valid_indices[idx][0], self, bounds, extraction_size)
                        yield bounding_box
                    else: 
                        print("No valid indices found, removing point from list")
                        self.hits.pop(idx)


class GridGeoSampler(GeoSampler):
    """Samples elements in a grid-like fashion.

    This is particularly useful during evaluation when you want to make predictions for
    an entire region of interest. You want to minimize the amount of redundant
    computation by minimizing overlap between :term:`chips <chip>`.

    Usually the stride should be slightly smaller than the chip size such that each chip
    has some small overlap with surrounding chips. This is used to prevent `stitching
    artifacts <https://arxiv.org/abs/1805.12219>`_ when combining each prediction patch.
    The overlap between each chip (``chip_size - stride``) should be approximately equal
    to the `receptive field <https://distill.pub/2019/computing-receptive-fields/>`_ of
    the CNN.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        size: tuple[float, float] | float,
        stride: tuple[float, float] | float,
        roi: BoundingBox | None = None,
        units: Units = Units.PIXELS,
    ) -> None:
        """Initialize a new Sampler instance.

        The ``size`` and ``stride`` arguments can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

        .. versionchanged:: 0.3
           Added ``units`` parameter, changed default to pixel units

        Args:
            dataset: dataset to index from
            size: dimensions of each :term:`patch`
            stride: distance to skip between each patch
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
            units: defines if ``size`` and ``stride`` are in pixel or CRS units
        """
        super().__init__(dataset, roi)
        self.size = _to_tuple(size)
        self.stride = _to_tuple(stride)

        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res, self.size[1] * self.res)
            self.stride = (self.stride[0] * self.res, self.stride[1] * self.res)

        self.hits = []
        for hit in self.index.intersection(tuple(self.roi), objects=True):
            bounds = BoundingBox(*hit.bounds)
            if (
                bounds.maxx - bounds.minx >= self.size[1]
                and bounds.maxy - bounds.miny >= self.size[0]
            ):
                self.hits.append(hit)

        self.length = 0
        for hit in self.hits:
            bounds = BoundingBox(*hit.bounds)
            rows, cols = tile_to_chips(bounds, self.size, self.stride)
            self.length += rows * cols

    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        # For each tile...
        for hit in self.hits:
            bounds = BoundingBox(*hit.bounds)
            rows, cols = tile_to_chips(bounds, self.size, self.stride)
            mint = bounds.mint
            maxt = bounds.maxt

            # For each row...
            for i in range(rows):
                miny = bounds.miny + i * self.stride[0]
                maxy = miny + self.size[0]

                # For each column...
                for j in range(cols):
                    minx = bounds.minx + j * self.stride[1]
                    maxx = minx + self.size[1]

                    yield BoundingBox(minx, maxx, miny, maxy, mint, maxt)

    def __len__(self) -> int:
        """Return the number of samples over the ROI.

        Returns:
            number of patches that will be sampled
        """
        return self.length


class PreChippedGeoSampler(GeoSampler):
    """Samples entire files at a time.

    This is particularly useful for datasets that contain geospatial metadata
    and subclass :class:`~torchgeo.datasets.GeoDataset` but have already been
    pre-processed into :term:`chips <chip>`.

    This sampler should not be used with :class:`~torchgeo.datasets.NonGeoDataset`.
    You may encounter problems when using an :term:`ROI <region of interest (ROI)>`
    that partially intersects with one of the file bounding boxes, when using an
    :class:`~torchgeo.datasets.IntersectionDataset`, or when each file is in a
    different CRS. These issues can be solved by adding padding.
    """

    def __init__(
        self, dataset: GeoDataset, roi: BoundingBox | None = None, shuffle: bool = False
    ) -> None:
        """Initialize a new Sampler instance.

        .. versionadded:: 0.3

        Args:
            dataset: dataset to index from
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
            shuffle: if True, reshuffle data at every epoch
        """
        super().__init__(dataset, roi)
        self.shuffle = shuffle

        self.hits = []
        for hit in self.index.intersection(tuple(self.roi), objects=True):
            self.hits.append(hit)

    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        generator: Callable[[int], Iterable[int]] = range
        if self.shuffle:
            generator = torch.randperm

        for idx in generator(len(self)):
            yield BoundingBox(*self.hits[idx].bounds)

    def __len__(self) -> int:
        """Return the number of samples over the ROI.

        Returns:
            number of patches that will be sampled
        """
        return len(self.hits)
