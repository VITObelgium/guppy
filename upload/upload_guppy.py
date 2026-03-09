import glob
import os
import sqlite3

import mercantile
import numpy as np
import rasterio
import requests
from fastapi import HTTPException
from joblib import Parallel, delayed
from osgeo import gdal
from pyproj import Transformer
from rio_tiler.errors import TileOutsideBounds
from rio_tiler.io import Reader
from rio_tiler.models import ImageData
from rio_tiler.profiles import img_profiles
from starlette.responses import Response
from tqdm import tqdm
import xml.etree.ElementTree as ET

def data_to_rgba(data: np.ndarray, nodata):
    """
    Converts a 2D numpy array to an RGBA masked array.
    Encodes the float bytes into the rgba channels of the ouput array.

    Args:
        data (np.ndarray): The input 2D numpy array.
        nodata: The value representing no data.

    Returns:
        np.ma.MaskedArray: The converted RGBA masked array.
    """
    data = data.astype(np.float32)
    if np.isnan(nodata):
        data = np.where(np.isnan(data), -9999, data)
        nodata = -9999
    rows, cols = data.shape
    rgb = np.frombuffer(data.astype("<f4").tobytes(), dtype=np.uint8).reshape(-1, 4).transpose().reshape(4, rows, cols).copy()

    rgb[3] = np.where(data == nodata, 255, rgb[3])
    rgb[2] = np.where(data == nodata, 255, rgb[2])
    rgb[1] = np.where(data == nodata, 255, rgb[1])
    rgb[0] = np.where(data == nodata, 255, rgb[0])

    return np.ma.MaskedArray(rgb)


def get_tile(file_path: str, z: int, x: int, y: int, style: str = None, values: str = None, colors: str = None) -> Response:
    """
    Args:
        file_path: A string representing the path to the file.
        z: An integer representing the zoom level.
        x: An integer representing the x-coordinate of the tile.
        y: An integer representing the y-coordinate of the tile.
        style: An optional string representing the style of the tile.
        values: An optional string representing the values for the custom style.
        colors: An optional string representing the colors for the custom style.

    Returns:
        A Response object containing the rendered tile image in PNG format, or raises an HTTPException with a status code of 404 and a corresponding detail message.

    """
    try:
        img = None
        nodata = None
        with Reader(file_path) as cog:
            try:
                img = cog.tile(x, y, z,indexes=1)
                nodata = cog.dataset.nodata
            except TileOutsideBounds:
                return None
            except Exception as e:
                print(e)
            if img and img.dataset_statistics is None:
                # gdal.Info(file_path, computeMinMax=True, stats=True)
                stats = cog.statistics()["b1"]
        if img:
            colormap = None
            add_mask = True
            if style:
                if style == "shader_rgba":
                    img.array = data_to_rgba(img.data[0], nodata)
                    add_mask = False
                else:
                    if img.dataset_statistics:
                        min_val = img.dataset_statistics[0][0]
                        max_val = img.dataset_statistics[0][1]
                    else:
                        min_val = stats.min
                        max_val = stats.max

                    if style in ["custom", "values", "intervals"]:
                        if not values or not colors:
                            raise HTTPException(status_code=400, detail="values and colors must be provided for a custom style")
                        value_points = [float(x) for x in values.split(",")]

                        if "_" in colors:
                            colors_points = [(int(x), int(y), int(z), int(a)) for x, y, z, a in [color.split(",") for color in colors.split("_")]]
                        else:
                            colors_points = [str(x) for x in colors.split(",")]
                        try:
                            # Sort values and colors together, maintaining their relative positions
                            sorted_pairs = sorted(zip(value_points, colors_points), key=lambda pair: pair[0])
                            value_points = [pair[0] for pair in sorted_pairs]
                            colors_points = [pair[1] for pair in sorted_pairs]
                        except ValueError as e:
                            raise HTTPException(
                                status_code=400, detail="values and colors must be the same length. colors must be sets of 4 values r,g,b,a separated by commas and sets of colors separated by _"
                            )

                        # OPTIMIZATION: For discrete data, skip interpolation-based rescaling.
                        if style == "values":
                            processed_colors = []
                            if isinstance(colors_points[0], str):
                                processed_colors = [hex_to_rgb(c) for c in colors_points]
                            else:
                                processed_colors = colors_points

                            if all(0 <= v <= 255 for v in value_points):
                                # Fast path: values already fit in byte range, use directly
                                colormap = {}
                                for v, color in zip(value_points, processed_colors):
                                    r, g, b = color[:3]
                                    a = color[3] if len(color) > 3 else 255
                                    colormap[int(v)] = (r, g, b, a)
                                # Do NOT rescale — raw pixel values map directly to colormap keys
                            else:
                                # Remap unique raw values to 0-N to avoid collisions after rescaling
                                unique_vals = sorted(set(value_points))
                                if len(unique_vals) <= 256:
                                    val_to_idx = {v: i for i, v in enumerate(unique_vals)}
                                    # Remap image data: replace each raw value with its compact index
                                    raw_data = img.array.data[0].astype(np.float32)
                                    remapped = np.full_like(raw_data, 255, dtype=np.uint8)
                                    for raw_v, idx in val_to_idx.items():
                                        remapped[raw_data == raw_v] = idx
                                    # Replace img array with remapped data (keep mask)
                                    orig_mask = img.array.mask
                                    if orig_mask is np.ma.nomask:
                                        orig_mask = np.zeros((1,) + raw_data.shape, dtype=bool)
                                    elif orig_mask.ndim == 2:
                                        orig_mask = orig_mask[np.newaxis, :, :]
                                    # Build a fresh ImageData with uint8 dtype — avoids int32 PNG warning
                                    img = ImageData(
                                        np.ma.MaskedArray(remapped[np.newaxis, :, :], mask=orig_mask)
                                    )
                                    # Build colormap from compact indices
                                    colormap = {}
                                    for v, color in zip(value_points, processed_colors):
                                        idx = val_to_idx.get(v)
                                        if idx is not None:
                                            r, g, b = color[:3]
                                            a = color[3] if len(color) > 3 else 255
                                            colormap[idx] = (r, g, b, a)
                                else:
                                    # Too many unique values, fall back to rescale path
                                    min_val = min(min_val, min(value_points))
                                    max_val = max(max_val, max(value_points))
                                    img.rescale(in_range=[(min_val, max_val)])
                                    colormap = generate_colormap_discrete(min_val, max_val, value_points, colors_points)
                        elif style == "intervals":
                            processed_colors = []
                            if isinstance(colors_points[0], str):
                                processed_colors = [hex_to_rgb(c) for c in colors_points]
                            else:
                                processed_colors = colors_points

                            n_intervals = len(value_points)
                            if n_intervals <= 256:
                                # Remap each pixel to the interval index it falls into
                                raw_data = img.array.data[0].astype(np.float32)
                                breakpoints = np.array(value_points, dtype=np.float32)
                                # searchsorted gives index of first breakpoint > pixel value
                                # subtract 1 → index of the interval the pixel falls into
                                interval_idx = np.searchsorted(breakpoints, raw_data, side='right') - 1
                                # Clamp: pixels below first breakpoint → 0, above last → last interval
                                interval_idx = np.clip(interval_idx, 0, n_intervals - 1).astype(np.uint8)
                                orig_mask = img.array.mask
                                if orig_mask is np.ma.nomask:
                                    orig_mask = np.zeros((1,) + raw_data.shape, dtype=bool)
                                elif orig_mask.ndim == 2:
                                    orig_mask = orig_mask[np.newaxis, :, :]
                                # Build a fresh ImageData with uint8 dtype — avoids int32 PNG warning
                                img = ImageData(
                                    np.ma.MaskedArray(interval_idx[np.newaxis, :, :], mask=orig_mask)
                                )
                                # Build colormap: interval index → color
                                colormap = {}
                                for i, color in enumerate(processed_colors):
                                    r, g, b = color[:3]
                                    a = color[3] if len(color) > 3 else 255
                                    colormap[i] = (r, g, b, a)
                        else:
                            min_val = min(min_val, min(value_points))
                            max_val = max(max_val, max(value_points))
                            img.rescale(in_range=[(min_val, max_val)])
                            colormap = generate_colormap(min_val, max_val, value_points, colors_points)

            elif img.dataset_statistics:
                img.rescale(in_range=img.dataset_statistics)
            else:
                img.rescale(in_range=[(stats.min, stats.max)])
            content = img.render(img_format="PNG", colormap=colormap, add_mask=add_mask, **img_profiles.get("png"))
            img = None
            del img
            return Response(content, media_type="image/png")
    except TileOutsideBounds:
        pass


def is_hex_color(input):
    """
    Function to check if a string is a valid hex color.
    Args:
        input: The string to check.

    Returns:
        True if the input is a valid hex color, False otherwise.

    """
    return len(input) == 6 and all(c in "0123456789ABCDEF" for c in input.upper())


def hex_to_rgb(hex_color):
    """
    Function to convert a hex color to an RGB color.
    Args:
        hex_color: The hex color to convert.

    Returns:
        A tuple containing the RGB color.

    """
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def generate_colormap_discrete(min_val, max_val, value_points, colors):
    # Map the provided value_points from [min_val, max_val] to [0, 255]
    rescaled_points = np.interp(value_points, (min_val, max_val), (0, 255)).astype(int)

    if len(colors) != len(value_points):
        raise ValueError("values and colors must be the same length")

    if isinstance(colors[0], str) and is_hex_color(colors[0]):
        colors = [hex_to_rgb(color) for color in colors]

    colors = np.array(colors)

    # Create a step-based colormap for intervals
    # Every integer from 0-255 gets assigned the color of the highest rescaled_point <= itself
    final_colormap = {}
    for i in range(256):
        # Find the index of the largest point <= current value
        idx = np.searchsorted(rescaled_points, i, side='right') - 1
        if idx >= 0:
            r, g, b = colors[idx][:3]
            a = colors[idx][3] if colors.shape[1] == 4 else 255
            final_colormap[i] = (int(r), int(g), int(b), int(a))
        else:
            # Fallback for values below the first point (e.g., nodata)
            final_colormap[i] = (0, 0, 0, 0)

    return final_colormap

def generate_colormap(min_val, max_val, value_points, colors):
    # Map the provided value_points from [min_val, max_val] to [0, 255]
    rescaled_points = np.interp(value_points, (min_val, max_val), (0, 255))
    # Generate colormap over 256 values
    all_values = np.linspace(0, 255, 256)

    colormap = {}
    if len(colors) != len(value_points):
        raise ValueError("values and colors must be the same length")
    if isinstance(colors[0], str) and is_hex_color(colors[0]):
        colors = [hex_to_rgb(color) for color in colors]

    colors = np.array(colors)
    # Interpolate color channels
    r = np.interp(all_values, rescaled_points, colors[:, 0])
    g = np.interp(all_values, rescaled_points, colors[:, 1])
    b = np.interp(all_values, rescaled_points, colors[:, 2])
    if colors.shape[1] == 4:
        a = np.interp(all_values, rescaled_points, colors[:, 3])
    else:
        a = np.full_like(all_values, 255)
    final_colormap = {int(v): (int(r[i]), int(g[i]), int(b[i]), int(a[i])) for i, v in enumerate(all_values)}
    return final_colormap


def process_tile(coord, z, geotiff_path, style, values, colors):
    x, y = coord
    try:
        response = get_tile(geotiff_path, z, x, y, style, values, colors)
        if response and response.status_code == 200:
            tms_y = (1 << z) - 1 - y
            tile_data = response.body
            return (z, x, tms_y, tile_data)
    except HTTPException:
        return None


def create_mbtiles(layer_name, file_path, geotiff_path, min_zoom, max_zoom, style=None, values=None, colors=None):
    # Open the dataset to get EPSG:3857 bounds
    with rasterio.open(geotiff_path) as dataset:
        bounds = dataset.bounds
        gdal.Info(geotiff_path, computeMinMax=True, stats=True)

    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    min_lon, min_lat = transformer.transform(bounds.left, bounds.bottom)
    max_lon, max_lat = transformer.transform(bounds.right, bounds.top)
    if os.path.exists(file_path):
        os.remove(file_path)
    # Open an SQLite connection to create MBTiles
    conn = sqlite3.connect(file_path)
    cursor = conn.cursor()

    # Create tables as per the MBTiles specification
    cursor.execute("CREATE TABLE IF NOT EXISTS tiles (zoom_level INTEGER, tile_column INTEGER, tile_row INTEGER, tile_data BLOB)")
    cursor.execute("CREATE TABLE IF NOT EXISTS metadata (name TEXT, value TEXT)")
    cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS tile_index ON tiles (zoom_level, tile_column, tile_row)")

    # Set metadata
    cursor.execute("INSERT INTO metadata (name, value) VALUES (?, ?)", ("name", layer_name))
    cursor.execute("INSERT INTO metadata (name, value) VALUES (?, ?)", ("format", "png"))
    cursor.execute("INSERT INTO metadata (name, value) VALUES (?, ?)", ("minzoom", str(min_zoom)))
    cursor.execute("INSERT INTO metadata (name, value) VALUES (?, ?)", ("maxzoom", str(max_zoom)))

    # Iterate over each zoom level, x, y within bounds
    for z in range(min_zoom, max_zoom + 1):
        # Calculate tile bounds at this zoom level using mercantile

        tiles = list(mercantile.tiles(min_lon, min_lat, max_lon, max_lat, z))
        min_x = min(tile.x for tile in tiles)
        max_x = max(tile.x for tile in tiles)
        min_y = min(tile.y for tile in tiles)
        max_y = max(tile.y for tile in tiles)
        coords = [(x, y) for x in range(min_x, max_x + 1) for y in range(min_y, max_y + 1)]
        results = Parallel(n_jobs=-1)(delayed(process_tile)(coord, z, geotiff_path, style, values, colors) for coord in tqdm(coords))
        coords = None
        for result in results:
            z, x, tms_y, tile_data = result
            cursor.execute("INSERT INTO tiles (zoom_level, tile_column, tile_row, tile_data) VALUES (?, ?, ?, ?)", (z, x, tms_y, tile_data))
        results = None
    # Commit changes and close the connection
    cursor.close()
    conn.commit()
    conn.close()


def save_geotif_tiled_overviews(input_file: str, output_file: str, nodata: int) -> str:
    """
    Saves a GeoTIFF file with tiled overviews.

    Args:
        input_file: Path to the input file.
        output_file: Path to save the output GeoTIFF file.
        nodata: Pixel value used to represent no data in the output file.

    Returns:
        The path to the saved output GeoTIFF file.
    """
    if os.path.exists(output_file):
        return output_file
    with rasterio.open(input_file) as src:
        target_crs = rasterio.crs.CRS.from_epsg(code=3857)
        tmp_input_file = None
        if src.crs != target_crs or nodata != src.nodata:
            transform, width, height = rasterio.warp.calculate_default_transform(src.crs, target_crs, src.width, src.height, *src.bounds)
            profile = src.profile
            profile.update(crs=target_crs, transform=transform, width=width, height=height)
            tmp_input_file = os.path.join("c:/dev/", os.path.basename(input_file).replace(".tif", "_tmp.tif"))
            with rasterio.open(tmp_input_file, "w", **profile) as dst:
                for i in range(1, src.count + 1):
                    rasterio.warp.reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=rasterio.enums.Resampling.nearest,
                        dst_nodata=nodata,
                    )
    if tmp_input_file:
        input_file = tmp_input_file

    translate_options = gdal.TranslateOptions(gdal.ParseCommandLine(f"-of COG -co COMPRESS=ZSTD -co BIGTIFF=YES -a_nodata {nodata} -co BLOCKSIZE=256 -co RESAMPLING=NEAREST"))
    gdal.Translate(output_file, input_file, options=translate_options)
    gdal.Info(output_file, computeMinMax=True, stats=True)
    if tmp_input_file:
        os.remove(tmp_input_file)
    return output_file


def send_file_to_guppy(server, file_path, data_path, layer_name, layer_label, max_zoom=15):
    url = f"https://{server}.marvintest.vito.be/guppy/admin/layer/{layer_name}"
    response = requests.delete(url)
    print(response.status_code, response.text)
    url = f"https://{server}.marvintest.vito.be/guppy/admin/upload"  # Update with your actual endpoint URL
    files = {"file": open(file_path, "rb")}
    if data_path:
        files["data"] = open(data_path, "rb")
    data = {
        "layerName": layer_name,
        "layerLabel": layer_label,
        "isRgb": "false",  # Change to 'false' if needed
        "isMbtile": "true",  # Change to 'false' if needed
        "maxZoom": max_zoom,
        "metadata": str({"max_zoom": max_zoom}),
    }

    response = requests.post(url, data=data, files=files)
    print(response.status_code, response.text)


def upload_layer(layer_name, file_path):
    url = "https://zmkempen.marvintest.vito.be/guppy/admin/layer"  # Update with your actual endpoint URL
    body = {"layerName": layer_name, "label": layer_name, "filePath": file_path, "isRgb": False, "isMbtile": False}
    response = requests.post(url, json=body)
    print(response.status_code, response.text)
mapping = [
    {
        "layer": "nwv_landgebruik_v3",
        "sld": "v2/sld/landuse"
    },
    {
        "layer": "nwv_drainage_v2",
        "sld": "v2/sld/soilMoisture"
    },
    {
        "layer": "nwv_profiel_v2",
        "sld": "v2/sld/soilStructure"
    },
    {
        "layer": "nwv_textuur_v2",
        "sld": "v2/sld/soilTexture"
    },
    {
        "layer": "nwv_wrb_v2",
        "sld": "v2/sld/soilClassification"
    },
    {
        "layer": "nwv_veen_v2",
        "sld": "v2/sld/peat"
    },
    {
        "layer": "nwv_ghg_v2",
        "sld": "v2/sld/ghg"
    },
    {
        "layer": "nwv_glg_v2",
        "sld": "v2/sld/glg"
    },
    {
        "layer": "nwv_helling_v2",
        "sld": "v2/sld/slopePercentage"
    },
    {
        "layer": "nwv_lsfactor_v2",
        "sld": "v2/sld/lsFactor"
    },
    {
        "layer": "nwv_ferrarisbos_v2",
        "sld": "v2/sld/ferrarisPercentage"
    },
    {
        "layer": "nwv_publiek-prive-bos_v2",
        "sld": "v2/sld/privateWoodPercentage"
    },
    {
        "layer": "nwv_pm10_v2",
        "sld": "v2/sld/concentrationPM10"
    },
    {
        "layer": "nwv_pm25_v2",
        "sld": "v2/sld/concentrationPM25"
    },
    {
        "layer": "nwv_rfactor_v2",
        "sld": "v2/sld/rFactor"
    },
    {
        "layer": "nwv_kfactor_v2",
        "sld": "v2/sld/kFactor"
    },
    {
        "layer": "nwv_regenval_v2",
        "sld": "v2/sld/rainfall"
    },
    {
        "layer": "nwv_street-canyons_v2",
        "sld": "v2/sld/canyon"
    },
    {
        "layer": "nwv_bwk_v2",
        "sld": "v2/sld/biologicalValuation"
    },
    {
        "layer": "nwv_relief-score_v2",
        "sld": "v2/sld/reliefScore"
    },
    {
        "layer": "nwv_cultuurhistorisch-score_v2",
        "sld": "v2/sld/culturalHistoricalValueScore"
    },
    {
        "layer": "nwv_horizon-score_v2",
        "sld": "v2/sld/visualPollutionScore"
    },
    {
        "layer": "nwv_geluid_v2",
        "sld": "v2/sld/noiseLevelScore"
    },
    {
        "layer": "nwv_padendensiteit_v2",
        "sld": "v2/sld/pathDensityScore"
    },
    {
        "layer": "nwv_recreatie-wandelaars_v2",
        "sld": "v2/sld/visitsWalking"
    },
    {
        "layer": "nwv_recreatie-fietsen_v2",
        "sld": "v2/sld/visitsCycling"
    },
    {
        "layer": "nwv_recreatie-auto_v2",
        "sld": "v2/sld/visitsCar"
    },
    {
        "layer": "nwv_recreatie-toeristen_v2",
        "sld": "v2/sld/visitsTourism"
    },
    {
        "layer": "nwv_publiek-prive-bos_v3",
        "sld": "v2/sld/privateWoodPercentage"
    },
    {
        "layer": "nwv_kfactor_v3",
        "sld": "v2/sld/kFactor"
    },
    {
        "layer": "nwv_profiel_v3",
        "sld": "v2/sld/soilStructure"
    },
    {
        "layer": "nwv_drainage_v3",
        "sld": "v2/sld/soilMoisture"
    },
    {
        "layer": "nwv_textuur_v3",
        "sld": "v2/sld/soilTexture"
    },

]


def get_color_values_from_sld(sld_path):
    url = f"https://natuurwaardeverkenner.marvintest.vito.be/api/{sld_path}"
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to get sld for layer {sld_path}")
            return None, None, None
        root = ET.fromstring(response.content)
        # SLD uses namespaces, we need to define them to find elements
        namespaces = {
            'sld': 'http://www.opengis.net/sld',
            'ogc': 'http://www.opengis.net/ogc'
        }
        colormap_element = root.find(".//sld:ColorMap", namespaces)
        colormap_type = colormap_element.get('type') if colormap_element is not None else None
        type = colormap_type if colormap_type in ["values", "intervals"] else "custom"
        entries = root.findall(".//sld:ColorMapEntry", namespaces)

        if colormap_type == 'intervals':
            out_values = [-9999] # first boundary
            out_colors = ['FFFFFF']  # first color (nodata start)

            for i in range(0, len(entries)):
                next_color = entries[i + 1].attrib['color'].lstrip('#') if i < len(entries) - 1 else entries[i].attrib['color'].lstrip('#')
                curr_quantity = float(entries[i].attrib['quantity'])
                curr_color = entries[i].attrib['color'].lstrip('#')

                out_values.append(curr_quantity)  # end of previous interval (flat close)
                out_colors.append(next_color)
            out_values.append(9999)  # start of next interval
            return ','.join(out_colors), ','.join(str(v) for v in out_values), 'intervals'
        else:
            out_colors = [entry.get('color').lstrip('#') for entry in entries]
            out_values = [entry.get('quantity') for entry in entries]
        print(out_colors, out_values, type)
        return ",".join(out_colors), ",".join(out_values), type

    except Exception as e:
        print(f"Error parsing SLD: {e}")
        return None, None, None

if __name__ == "__main__":
    root_folder = r"C:\RMAbuild\Projects\guppy2\server\guppy\content\tifs\nwv"
    for map_item in mapping:
        files = glob.glob(rf"{root_folder}/*{map_item['layer'].replace('nwv_','')}*.tif")
        if len(files) == 0 or len(files) > 1:
            print(f"Skipping layer {map_item['layer']} found files: {files}")
            continue
        input_file = files[0]
        laye_name = f"""{map_item["layer"]}_mbtile"""
        tif_path = os.path.join("c:/dev/", os.path.basename(input_file))
        mbtile_path = os.path.join("c:/dev/", os.path.basename(input_file).replace(".tif", ".mbtiles"))
        colors, values, style_type = get_color_values_from_sld(map_item["sld"])
        print(colors, values, style_type)
        # if style_type == "intervals":
        print(laye_name,style_type)
        save_geotif_tiled_overviews(input_file, tif_path, -9999)
        create_mbtiles(laye_name, mbtile_path, tif_path, min_zoom=8, max_zoom=13, style=style_type, values=values, colors=colors)
        send_file_to_guppy(server='natuurwaardeverkenner', file_path=mbtile_path, data_path=None, layer_name=laye_name, layer_label=laye_name, max_zoom=13)
    # upload existing file from geoserver
    # upload_layer('nwv_pm10_v2','/content/tifs/nwv/pm10_v2.tif')
    # send_file_to_guppy("zmkempen",r"C:\RMAbuild\Projects\zmkempen_deploy/zn-grondwater_v1.tif",None,"zn-grondwater_v1","zn-grondwater_v1")
    # send_file_to_guppy("zmkempen",r"C:\RMAbuild\Projects\zmkempen_deploy/zn-bodem_v1.tif",None,"zn-bodem_v1","zn-bodem_v1")
    # send_file_to_guppy("zmkempen",r"C:\RMAbuild\Projects\zmkempen_deploy/pb-bodem_v1.tif",None,"pb-bodem_v1","pb-bodem_v1")
    # send_file_to_guppy("zmkempen",r"C:\RMAbuild\Projects\zmkempen_deploy/cd-grondwater_v1.tif",None,"cd-grondwater_v1","cd-grondwater_v1")
    # send_file_to_guppy("zmkempen",r"C:\RMAbuild\Projects\zmkempen_deploy/cd-bodem_v1.tif",None,"cd-bodem_v1","cd-bodem_v1")
