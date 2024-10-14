from torchgeo.datasets.geo import GeoDataset
from torchgeo.datasets.utils import BoundingBox
import sys
from rasterio.crs import CRS
from typing import Any, Callable, cast



class Location(GeoDataset):
    """Dummy dataset when only location serves as a predictor"""

    def __init__(
        self, 
        bounds: BoundingBox = BoundingBox(-180, 180, -90, 90, 0, sys.maxsize),
        res: float = 1,
        crs: CRS = CRS.from_epsg(4326),
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        
        """Initialize a new Dataset instance.
        
        Args:
            bounds (BoundingBox): Bounding box to sample from
            res (float): Resolution of the dataset
            crs (CRS): Coordinate Reference System of the dataset
        """

        super().__init__(transforms=transforms)
        
        self.res = res
        self.crs = crs

        coords = (bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5])
        self.index.insert(0, coords)

        self._crs = cast(CRS, self.crs)
        self._res = cast(float, self.res)

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:

        #NOTE could be extended to return location
        # it could then be treated as a separate predictor (remove location from point dataset)

        """
        sample = {
            "crs": self.crs,
            "location": [[query.center.minx, query.center.miny]],
            "center": [[center.minx, center.miny]]
        }
        

        return sample"""
        return {}