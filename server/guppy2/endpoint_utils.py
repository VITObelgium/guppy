import numpy as np
import rasterio
from rasterio.mask import mask, raster_geometry_mask

from guppy2.db import schemas as s


def get_overview_factor(bounds, native, path):
    if not native:
        with rasterio.open(path) as src:
            res_x, res_y = src.res
            overview_factor, overview_bin = get_overview(res_x, res_y, src.overviews(1), bounds)
    else:
        overview_factor, overview_bin = None, None
    return overview_factor, overview_bin


def create_stats_response(rst: np.array, mask_array: np.array, nodata: float, type: str):
    rst = rst.astype(float)
    rst[rst == nodata] = np.nan
    q2, q5, q95, q98 = np.nanquantile(rst, [0.02, 0.05, 0.95, 0.98])
    response = s.StatsResponse(type=type,
                               min=float(np.nanmin(rst)),
                               max=float(np.nanmax(rst)),
                               sum=float(np.nansum(rst)),
                               mean=float(np.nanmean(rst)),
                               count_no_data=int(np.sum((~np.isfinite(rst)) & (~mask_array))),
                               count_total=int(np.sum(~mask_array)),
                               count_data=int(np.sum(np.isfinite(rst))),
                               q02=float(q2),
                               q05=float(q5),
                               q95=float(q95),
                               q98=float(q98),
                               )
    return response


def _extract_area_from_dataset(raster_ds, geom, crop=True, all_touched=False):
    crop_arr, crop_transform = mask(raster_ds, geom, crop=crop, all_touched=all_touched)
    crop_arr = crop_arr[0]
    return crop_arr, crop_transform


def _extract_shape_mask_from_dataset(raster_ds, geom, crop=True, all_touched=False):
    shape_mask, transform, window = raster_geometry_mask(raster_ds, geom, all_touched=all_touched, crop=crop)
    return shape_mask


def get_overview(res_x: float, res_y: float, overviews: [int], bounds: (float,)):
    bbox_bottom, bbox_left, bbox_top, bbox_right = bounds
    pixels = (bbox_right - bbox_left) / res_x * (bbox_top - bbox_bottom) / res_y
    factor = int(pixels / 4000000)
    overview_level = 0
    for i, value in enumerate(overviews):
        if value > factor:
            overview_level = i - 1
            break
    if overview_level < 0:
        return None, None
    return overview_level, overviews[overview_level]


def _decode(data, base, interval):
    """
    Utility to decode RGB encoded data
    """
    data = data.astype(np.float64)
    return base + (((data[0] * 256 * 256) + (data[1] * 256) + data[2]) * interval)