# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Common sampler utilities."""

import math
from typing import overload

import torch
from torchgeo.datasets.geo import RasterDataset, PointDataset, IntersectionDataset
import torchvision
import time
import rasterio
from rasterio.crs import CRS
from rasterio.vrt import WarpedVRT
from tqdm import tqdm
import numpy as np

from ..datasets import BoundingBox


@overload
def _to_tuple(value: tuple[int, int] | int) -> tuple[int, int]: ...


@overload
def _to_tuple(value: tuple[float, float] | float) -> tuple[float, float]: ...


def _to_tuple(value: tuple[float, float] | float) -> tuple[float, float]:
    """Convert value to a tuple if it is not already a tuple.

    Args:
        value: input value

    Returns:
        value if value is a tuple, else (value, value)
    """
    if isinstance(value, float | int):
        return (value, value)
    else:
        return value
    

def generate_area_indices(bounds: BoundingBox, size: tuple[float, float], res:float):

    area_indicesa = []

    

    return area_indices


def get_random_bounding_box(
    bounds: BoundingBox, size: tuple[float, float] | float, res: float
) -> BoundingBox:
    """Returns a random bounding box within a given bounding box.

    The ``size`` argument can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

    Args:
        bounds: the larger bounding box to sample from
        size: the size of the bounding box to sample
        res: the resolution of the image

    Returns:
        randomly sampled bounding box from the extent of the input
    """
    t_size = _to_tuple(size)

    # May be negative if bounding box is smaller than patch size
    width = (bounds.maxx - bounds.minx - t_size[1]) / res
    height = (bounds.maxy - bounds.miny - t_size[0]) / res

    minx = bounds.minx
    miny = bounds.miny

    # Use an integer multiple of res to avoid resampling
    minx += int(torch.rand(1).item() * width) * res
    miny += int(torch.rand(1).item() * height) * res

    maxx = minx + t_size[1]
    maxy = miny + t_size[0]

    mint = bounds.mint
    maxt = bounds.maxt

    query = BoundingBox(minx, maxx, miny, maxy, mint, maxt)
    return query


def tile_to_chips(
    bounds: BoundingBox,
    size: tuple[float, float],
    stride: tuple[float, float] | None = None,
) -> tuple[int, int]:
    r"""Compute number of :term:`chips <chip>` that can be sampled from a :term:`tile`.

    Let :math:`i` be the size of the input tile. Let :math:`k` be the requested size of
    the output patch. Let :math:`s` be the requested stride. Let :math:`o` be the number
    of output chips sampled from each tile. :math:`o` can then be computed as:

    .. math::

       o = \left\lceil \frac{i - k}{s} \right\rceil + 1

    This is almost identical to relationship 5 in
    https://doi.org/10.48550/arXiv.1603.07285. However, we use ceiling instead of floor
    because we want to include the final remaining chip in each row/column when bounds
    is not an integer multiple of stride.

    Args:
        bounds: bounding box of tile
        size: size of output patch
        stride: stride with which to sample (defaults to ``size``)

    Returns:
        the number of rows/columns that can be sampled

    .. versionadded:: 0.4
    """
    if stride is None:
        stride = size

    assert stride[0] > 0
    assert stride[1] > 0

    rows = math.ceil((bounds.maxy - bounds.miny - size[0]) / stride[0]) + 1
    cols = math.ceil((bounds.maxx - bounds.minx - size[1]) / stride[1]) + 1

    return rows, cols


def prepare_complex_roi(filepath: str, new_crs: CRS):
    """Prepare a complex region of interest for a dataset.

    Args:
        path: path to a file containing a complex region of interest

    Returns:
        a complex region of interest
    """

    src = rasterio.open(filepath)

    if src.crs != new_crs:
        vrt = WarpedVRT(src, crs=new_crs)
        print(f"Reprojecting {filepath} to {new_crs}")
        src.close()
        return vrt
    else:
        return src

    
def extract_valid_bboxes(complex_roi: rasterio.DatasetReader, dataset: IntersectionDataset | RasterDataset | PointDataset, bounds: BoundingBox, size: int | tuple[int, int] = 1, resample: int = 4, tolerance: float = 0.8):

    """Extract valid bounding boxes from a complex region of interest.
    
    Args:
        complex_roi: complex region of interest
        dataset: dataset to extract resolution from
        bounds: bounding box of the image
        size: size of the patch
        stride: stride with which to sample
        tolerance: minimum valid proportion of the patch
    
    """
    size = _to_tuple(size)

    #check if resmple factor is valid
    if size[0]/resample < 1 or size[1]/resample < 1:
        warnings.warn("Resample factor is too high. Setting to 1.")
        resample = 1



    #complex roi should already be rasterio dataset
    res = dataset.res

    #offset only needed if raster because for point its constructed anyways if not centered
    #print dataset instance
    if isinstance(dataset, RasterDataset) or isinstance(dataset, IntersectionDataset):
        x_offset = (size[1]/2)*res
        y_offset = (size[0]/2)*res
    else: 
        x_offset,y_offset = 0,0

    extraction_bounds = (bounds.minx-x_offset, bounds.miny-y_offset, bounds.maxx+x_offset, bounds.maxy+y_offset)
    start = time.time()
    crop, _ = rasterio.merge.merge([complex_roi], bounds=extraction_bounds, res=res)
    crop = (crop > 0).astype(np.int8)

    crop = torch.tensor(crop, dtype=torch.float64)
    crop_size = crop.shape[1], crop.shape[2]
    filter = torch.ones((size[0]//resample, size[1]//resample), dtype=torch.float64)

    #downsample crop
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((crop_size[0]//resample,crop_size[1]//resample))])
    crop = transform(crop.unsqueeze(0)).squeeze(0)
    valids = torch.nn.functional.conv2d(crop.unsqueeze(0), filter.unsqueeze(0).unsqueeze(0), stride = 1)/((size[0]/resample)*(size[1]/resample))
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((crop_size[0]-(size[0]-1),crop_size[1]-(size[1]-1)))])
    valids = (transform(valids).squeeze(0).squeeze(0) >= tolerance)*1


    #get non-zero indices
    valids = torch.argwhere(valids)
    valids = valids.numpy()


    return valids

def image_coordinates_to_geocoordinates(
    x: int, y: int, dataset: RasterDataset | PointDataset, bounds: BoundingBox, size: int | tuple[int, int] = 1
) -> tuple[float, float]:
    """Convert image coordinates to geocoordinates.

    Args:
        x: x-coordinate in image space
        y: y-coordinate in image space
        dataset: dataset to extract resolution from
        bounds: bounding box of the image

    Returns:
        geocoordinates
    """
    res = dataset.res
    size = _to_tuple(size)

    minx = bounds.minx + (res*x)
    maxx = minx + size[1]*res
    maxy = bounds.maxy - (res*y)
    miny = maxy - size[0]*res


    return BoundingBox(minx, maxx, miny, maxy, bounds.mint, bounds.maxt)

def extract_valid_tiles(
    complex_roi: rasterio.DatasetReader,
    dataset: RasterDataset | PointDataset | IntersectionDataset,
    centered: bool,
    hits: list,
    size: int | tuple[int, int] = 1,
    tolerance: float = 0.8,
) -> list[int]:
    """Extract valid tiles from a dataset.

    Args:
        complex_roi: complex region of interest
        dataset: dataset to extract resolution from
        hits: list of files
        size: size of the patch
        tolerance: minimum valid proportion of the patch

    Returns:
        list of valid tiles
    """
 
    res = dataset.res if dataset.res > 0 else complex_roi.res #for edge case location only
    size = _to_tuple(size)

    if isinstance(dataset, PointDataset):
        extent = 2 if centered else 1
        x_offset = size[1]/extent*res
        y_offset = size[0]/extent*res
    else:
        x_offset = size[1]/2*res
        y_offset = size[0]/2*res
    
    bounds = [(BoundingBox(*hit.bounds).minx-x_offset, BoundingBox(*hit.bounds).miny-y_offset, BoundingBox(*hit.bounds).maxx+x_offset, BoundingBox(*hit.bounds).maxy+y_offset) for hit in hits]
    print("Extracting valid tiles")
    valid_tiles = [i for i, bound in tqdm(enumerate(bounds), total=len(bounds)) if (rasterio.merge.merge([complex_roi], bounds=bound, res=res)[0] > 0).astype(np.int8).sum()/(size[0]*size[1]) >= tolerance]
    return valid_tiles

def check_tile_validity(complex_roi: rasterio.DatasetReader, dataset: RasterDataset, bounds: BoundingBox, size: int | tuple[int, int] = 1, tolerance: float = 0.8) -> bool:
    """Check if a tile is valid.

    Args:
        complex_roi: complex region of interest
        dataset: dataset to extract resolution from
        bounds: bounding box of the tile
        size: size of the patch
        tolerance: minimum valid proportion of the patch

    Returns:
        True if the tile is valid, False otherwise
    """
    res = dataset.res
    #size = _to_tuple(size)

    x_offset = size[1]/2*res
    y_offset = size[0]/2*res

    bounds = (bounds.minx-x_offset, bounds.miny-y_offset, bounds.maxx+x_offset, bounds.maxy+y_offset)
    #start = time.time()
    crop, _ = rasterio.merge.merge([complex_roi], bounds=bounds, res=res)
    #print(f"Time Crop: {time.time()-start}")
    crop = (crop > 0).astype(np.int8)
    return crop.sum()/(size[0]*size[1]) >= tolerance
