import math
import os.path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize, shapes
from rasterio.transform import from_bounds
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry import box
from tqdm import tqdm


def to_mbtiles(layer_name: str, input_file_path: str, output_file_path: str, max_zoom=15, min_zoom=10):
    """
    Args:
        name: A string representing the name of the MBTiles file.
        input_file_path: A string representing the path to the input file.
        output_file_path: A string representing the path to save the output MBTiles file.
    """
    import subprocess

    cmd = [
        'ogr2ogr',
        '-f', 'MBTILES',
        '-dsco', 'MAX_FEATURES=5000000',
        '-dsco', 'MAX_SIZE=5000000',
        '-dsco', f'MINZOOM={min_zoom}',
        '-dsco', f'MAXZOOM={max_zoom}',
        '-dsco', f'NAME={layer_name}',
        '-lco', f'NAME={layer_name}',
        '-preserve_fid',
        output_file_path,
        input_file_path
    ]

    subprocess.run(cmd)


def round_coorinates_of_gpkg(df, limit):
    df['geometry'] = df['geometry'].apply(lambda x: x.simplify(limit, preserve_topology=True))
    return df


def combine_mbtiles(base_mbtile_file, to_add_mbtile_file):
    import sqlite3
    conn = sqlite3.connect(base_mbtile_file)
    conn.execute(f'ATTACH DATABASE "{to_add_mbtile_file}" AS db2')
    conn.execute('BEGIN')
    conn.execute('INSERT INTO main.tiles SELECT * FROM db2.tiles')
    conn.execute('COMMIT')
    # update min and max zoom lvls in metadata table
    min_zoom = conn.execute('SELECT min(zoom_level) FROM tiles').fetchone()[0]
    max_zoom = conn.execute('SELECT max(zoom_level) FROM tiles').fetchone()[0]
    conn.execute(f'UPDATE main.metadata SET value = {min_zoom} WHERE name = "minzoom"')
    conn.execute(f'UPDATE main.metadata SET value = {max_zoom} WHERE name = "maxzoom"')

    # also update the json field in metadata table
    json = conn.execute('SELECT value FROM main.metadata WHERE name = "json"').fetchone()[0]
    json = json.replace('"maxzoom":14', f'"maxzoom":{max_zoom}')
    json = json.replace('"minzoom":11', f'"minzoom":{min_zoom}')
    json = json.replace('"minzoom":9', f'"minzoom":{min_zoom}')
    conn.execute(f'''UPDATE main.metadata SET value = '{json}' WHERE name = "json"''')
    # analyze tiles index
    conn.execute('ANALYZE main.tiles')
    conn.execute('COMMIT')
    conn.execute('DETACH DATABASE db2')

    conn.commit()
    conn.close()


def create_raster_from_gpkg(input_file, column, output_file, resolution_px, dtype=np.float32):
    print("create_raster_from_gpkg")
    gdf = gpd.read_file(input_file, engine='pyogrio')
    bounds = gdf.total_bounds

    # Define the transformation
    transform = from_bounds(*bounds, width=resolution_px, height=resolution_px)
    gdf.fillna(-9999, inplace=True)
    # get pixel area:
    pixel_area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]) / resolution_px ** 2
    # Rasterize the geometries
    shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf[column]))
    raster = rasterize(shapes, out_shape=(resolution_px, resolution_px), fill=-9999, transform=transform, all_touched=True, dtype=dtype if dtype else None)

    # Create a new raster
    with rasterio.open(
            output_file,
            'w',
            driver='GTiff',
            height=resolution_px,
            width=resolution_px,
            count=1,
            dtype=dtype if dtype else None,
            crs='EPSG:4326',
            transform=transform,
            compress='deflate',
            nodata=-9999,
            tiled=True,
            blockxsize=256,
            blockysize=256
    ) as dst:
        dst.write(raster, 1)
    return pixel_area
    # # Open the raster and plot it
    # with rasterio.open(output_file) as src:
    #     show(src)


# Adjust X to your specific threshold

# Function to create grid and cut large polygon
def cut_large_polygon(polygon, value, cell_width, cell_height):
    xmin, ymin, xmax, ymax = polygon.bounds
    rows = math.ceil((ymax - ymin) / cell_height)
    cols = math.ceil((xmax - xmin) / cell_width)
    grid = []
    for i in range(cols):
        for j in range(rows):
            grid.append(box(xmin + i * cell_width, ymin + j * cell_height, xmin + (i + 1) * cell_width, ymin + (j + 1) * cell_height))
    grid_gdf = gpd.GeoDataFrame(geometry=grid, crs='EPSG:3857')
    large_poly_gdf = gpd.GeoDataFrame({'geometry': [polygon], 'VALUE': [value]}, crs='EPSG:3857')
    cut_pieces = gpd.overlay(grid_gdf, large_poly_gdf, how='intersection')
    return cut_pieces


# Apply cutting operation only to large polygons

def process_polygons(row, column, area_threshold, cell_width=10000, cell_height=10000):
    if row.geometry.area > area_threshold:
        return cut_large_polygon(row.geometry, row[column], cell_width, cell_height)
    else:
        return gpd.GeoDataFrame({'geometry': [row.geometry], column: [row[column]]}, crs='EPSG:3857')


def subdivide_input_df(input_file, output_file, column, area_threshold=100000000):
    print("subdivide_input_df")

    df = gpd.read_file(input_file, engine='pyogrio')
    df.to_crs('EPSG:3857', inplace=True)
    # Assuming you want to store the result in a list
    resulting_polygons = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        cut_or_original = process_polygons(row, column, area_threshold, cell_width=int(area_threshold / 10000), cell_height=int(area_threshold / 10000))
        resulting_polygons.append(cut_or_original)
    # Concatenate all GeoDataFrames from the resulting_polygons list into a single GeoDataFrame
    final_gdf = gpd.GeoDataFrame(pd.concat(resulting_polygons, ignore_index=True), crs='EPSG:3857')
    final_gdf.to_crs('EPSG:4326', inplace=True)
    final_gdf.to_file(output_file, driver='GPKG')


def remove_small_holes(geometry, pixel_area):
    if geometry.type == 'Polygon':
        return Polygon(geometry.exterior, [hole for hole in geometry.interiors if Polygon(hole).area > pixel_area])
    elif geometry.type == 'MultiPolygon':
        return MultiPolygon([Polygon(poly.exterior, [hole for hole in poly.interiors if Polygon(hole).area > pixel_area]) for poly in geometry])
    return geometry


def raster_to_polygon(input_file, output_file, pixel_area=0.000001):
    with rasterio.open(input_file) as src:
        image = src.read(1)
        if src.nodata is None:
            src.nodata = -9999
        nodata = src.nodata
        results = (
            {'properties': {'VALUE': v}, 'geometry': s}
            for i, (s, v) in enumerate(
            shapes(image, mask=image != nodata, transform=src.transform)) if v != nodata)

    gdf = gpd.GeoDataFrame.from_features(list(results), crs=src.crs)

    gdf['geometry'] = gdf['geometry'].apply(remove_small_holes, pixel_area=pixel_area)

    gdf.to_file(output_file, driver='GPKG')


if __name__ == '__main__':
    input_file = r'C:\RMAbuild\Projects\basf_basagran_map_api\lbpercelen_2022_bentazonrestricties_2024_wgs84\lbpercelen_2022_bentazonrestricties_2024_wgs84.shp'
    input_df = gpd.read_file(input_file, engine='pyogrio')
    input_df['parcel_id'] = np.where(input_df['layer'].str.contains('_FL_'), "F_", "W_") + input_df['OBJECTID'].astype(str)
    input_df['layer'] = np.where(input_df['layer'].str.contains('_FL_'), 'Landbouwgebruikspercelen LV (2022)', 'Parcellaire agricole anonyme (situation 2022)')
    bounds_df = input_df.bounds
    input_df['bounds'] = bounds_df.apply(lambda row: f"{round(row['minx'], 6)}, {round(row['miny'], 6)}, {round(row['maxx'], 6)}, {round(row['maxy'], 6)}", axis=1)

    input_file_extra = rf'{os.path.join(os.path.dirname(input_file), os.path.basename(input_file).split(".")[0])}_extra.gpkg'
    input_df.to_file(input_file_extra, driver='GPKG')
    #
    input_file_extra_simple = rf'{os.path.join(os.path.dirname(input_file), os.path.basename(input_file).split(".")[0])}_extra_simple.gpkg'
    input_df = round_coorinates_of_gpkg(input_df, limit=0.00005)
    input_df.to_file(input_file_extra_simple, driver='GPKG')
    input_df = None
    path = os.path.join(os.path.dirname(input_file), os.path.basename(input_file).split('.')[0])
    # area1 = create_raster_from_gpkg(input_file=input_file_extra, column='VALUE',
    #                                 output_file=(rf'{path}_simple1.tif'), resolution_px=6000, dtype=np.int32)
    # area2 = create_raster_from_gpkg(input_file=input_file_extra, column='VALUE',
    #                                 output_file=(rf'{path}_simple2.tif'), resolution_px=3000, dtype=np.int32)
    #
    # raster_to_polygon(input_file=(rf'{path}_simple1.tif'),
    #                   output_file=(rf'{path}_simple1.gpkg'), pixel_area=area1 * 4)
    # raster_to_polygon(input_file=(rf'{path}_simple2.tif'),
    #                   output_file=(rf'{path}_simple2.gpkg'), pixel_area=area2 * 4)
    # # #
    # subdivide_input_df(input_file=(rf'{path}_simple1.gpkg'),
    #                    output_file=(rf'{path}_simple1_subdiv.gpkg'),
    #                    column="VALUE", area_threshold=20000000)
    # #
    # subdivide_input_df(input_file=(rf'{path}_simple2.gpkg'),
    #                    output_file=(rf'{path}_simple2_subdiv.gpkg'),
    #                    column="VALUE", area_threshold=40000000)

    to_mbtiles(layer_name='basf', max_zoom=14, min_zoom=13,
               input_file_path=input_file_extra,
               output_file_path=(rf'{path}.mbtiles'))
    to_mbtiles(layer_name='basf', max_zoom=12, min_zoom=11,
               input_file_path=input_file_extra_simple,
               output_file_path=(rf'{path}_0.mbtiles'))
    to_mbtiles(layer_name='basf', max_zoom=10, min_zoom=9,
               input_file_path=(rf'{path}_simple1_subdiv.gpkg'),
               output_file_path=(rf'{path}_1.mbtiles'))
    to_mbtiles(layer_name='basf', max_zoom=8, min_zoom=0,
               input_file_path=(rf'{path}_simple2_subdiv.gpkg'),
               output_file_path=(rf'{path}_2.mbtiles'))

    combine_mbtiles(base_mbtile_file=(rf'{path}.mbtiles'), to_add_mbtile_file=(rf'{path}_0.mbtiles'))
    combine_mbtiles(base_mbtile_file=(rf'{path}.mbtiles'), to_add_mbtile_file=(rf'{path}_1.mbtiles'))
    combine_mbtiles(base_mbtile_file=(rf'{path}.mbtiles'), to_add_mbtile_file=(rf'{path}_2.mbtiles'))
