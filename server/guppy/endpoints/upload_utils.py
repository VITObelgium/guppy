import logging
import os
import re
import shutil

import geopandas as gpd
import rasterio
from fastapi import UploadFile
from osgeo import gdal
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from guppy.config import config as cfg
from guppy.db.models import LayerMetadata
from guppy.error import create_error

logger = logging.getLogger(__name__)


def validate_raster(file_path: str) -> list:
    """
    Validates a raster file. Checks for the following:
    - EPSG/CRS required
    - NoData required
    - Transform required
    - Single band required
    - Scale/Offset forbidden
    Args:
        file_path (str): The file path of the raster file to be validated.

    Returns:
        List[str]: A list of error messages indicating the validation errors found in the raster file.

    """
    errors = []
    if os.path.exists(file_path):
        with rasterio.open(file_path) as src:
            # Validate EPSG/CRS
            if src.crs is None:
                errors.append("Invalid CRS")
            # Validate NoData
            if src.nodatavals[0] is None:
                errors.append("Invalid NoData")
            # Validate transform
            if src.transform is None:
                errors.append("Invalid Transform")
            # Validate single band
            if src.count != 1:
                errors.append("More than one band")
            # Validate scale/offset
            if src.scales[0] != 1.0 or src.offsets[0] != 0.0:
                errors.append("Has scale/offset")
    else:
        logger.error("file not exists")
        errors.append("file not exists")
    return errors


def sanitize_input_str(input: str) -> str:
    """
    Sanitizes input string by removing any non-alphanumeric characters.

    Args:
        input: Input string to be sanitized.

    Returns:
        Sanitized input string with non-alphanumeric characters removed.
    """
    sanitized_input = re.sub(r'[^a-zA-Z0-9_-]', '', input)
    return sanitized_input


def validate_input_str(input: str) -> bool:
    """
    Args:
        input (str): The input string to be validated.

    Returns:
        bool: Returns True if the input string contains only alphanumeric characters and underscores, otherwise False.
    """
    if re.search(r'[^a-zA-Z0-9_-]', input):
        return False
    return True


def check_disk_space(temp_file_size: int):
    """Checks the disk space usage to determine if it would exceed 90% after file upload.

    Args:
        temp_file_size (int): The size of the temporary file to be uploaded.

    Raises:
        Error: If the disk space usage would exceed 90% after file upload.

    """
    total, used, free = shutil.disk_usage(f"{cfg.deploy.content}")
    used_percentage = ((used + temp_file_size) / total) * 100
    if used_percentage > 90:
        raise create_error(code=403, message="Upload failed: Disk space usage would exceed 90% after file upload")


def check_layer_exists(layer_name: str, db: Session):
    """
    Checks if a layer exists in the database.

    Args:
        layer_name (str): The name of the layer.
        db (Session): The database connection object.

    Raises:
        Error: If the layer already exists in the database.
    """
    layer = db.query(LayerMetadata).filter_by(layer_name=layer_name).first()
    if layer:
        raise create_error(code=400, message=f"Upload failed: Layer {layer_name} already exists.")


def create_preprocessed_layer_file(ext: str, file_location: str, sanitized_filename: str, sanitized_layer_name: str, tmp_file_location: str, max_zoom: int = 17) -> bool:
    """
    Args:
        ext (str): The extension of the file.
        file_location (str): The location of the output file.
        sanitized_filename (str): The sanitized filename.
        sanitized_layer_name (str): The sanitized layer name.
        tmp_file_location (str): The temporary file location.

    Returns:
        bool: True if the file is converted to mbtiles, False otherwise.
    """
    is_mbtile = False
    if ext.lower() in ['.tif', '.tiff', '.asc']:
        error_list = validate_raster(file_path=tmp_file_location)
        if error_list:
            raise create_error(message=f"Upload failed: {', '.join(error_list)}", code=400)
        else:
            save_geotif_tiled_overviews(input_file=tmp_file_location, output_file=file_location, nodata=-9999)
    elif ext.lower() in ['.mbtiles']:
        is_mbtile = True
    else:
        df = gpd.read_file(tmp_file_location)
        df.to_crs(epsg=4326, inplace=True)
        if len(df) == 1:  # explode if there is only one row
            df = df.explode().reset_index(drop=True)
        df['fid'] = df.index
        gpkg_loc = f"{cfg.deploy.content}/shapefiles/uploaded/{sanitized_layer_name}_{sanitized_filename}_tmp.geojson"
        if 'bounds' not in df.columns:
            df['bounds'] = df.geometry.envelope.to_wkt()
        df.to_file(gpkg_loc, index=False, layer=sanitized_layer_name, driver='GeoJSON')  # needs to be geojson to preserve the fid field in the mbtile
        to_mbtiles(sanitized_layer_name, gpkg_loc, file_location, max_zoom)
        os.remove(tmp_file_location)
        df.drop(columns=['geometry'], inplace=True)
        engine = create_engine(f'sqlite:///{file_location.replace(".mbtiles", ".sqlite")}')
        df.to_sql('tiles', con=engine, index=False)
        if os.path.exists(gpkg_loc):
            os.remove(gpkg_loc)
        is_mbtile = True
    return is_mbtile


def create_location_paths_and_check_if_exists(ext: str, sanitized_filename: str, sanitized_layer_name: str, is_raster=False) -> tuple[str, str]:
    """
    Creates file paths for the uploaded file and checks if the file already exists.

    Args:
        ext (str): The file extension.
        sanitized_filename (str): The sanitized name of the file.
        sanitized_layer_name (str): The sanitized name of the layer.
        is_raster (bool): Optional. Indicates whether the file is a raster file. Default is False.

    Returns:
        tuple: A tuple containing the file location and temporary file location.

    Raises:
        create_error: If the file already exists.

    """
    if ext.lower() in ['.tif', '.tiff', '.asc', ]:
        tmp_file_location = f"{cfg.deploy.content}/tifs/uploaded/{sanitized_layer_name}_{sanitized_filename}_tmp{ext}"
        file_location = f"{cfg.deploy.content}/tifs/uploaded/{sanitized_layer_name}_{sanitized_filename}.tif"
        if os.path.exists(file_location):
            raise create_error(message=f"Upload failed: File {sanitized_layer_name}_{sanitized_filename}.tif already exists.", code=400)
    elif ext.lower() in ['.mbtiles']:
        if is_raster:
            folder = 'tifs'
        else:
            folder = 'shapefiles'
        file_location = f"{cfg.deploy.content}/{folder}/uploaded/{sanitized_layer_name}_{sanitized_filename}.mbtiles"
        tmp_file_location = file_location
        if os.path.exists(file_location):
            raise create_error(message=f"Upload failed: File {sanitized_layer_name}_{sanitized_filename}.mbtiles already exists.", code=400)
    else:
        tmp_file_location = f"{cfg.deploy.content}/shapefiles/uploaded/{sanitized_layer_name}_{sanitized_filename}{ext}"
        file_location = f"{cfg.deploy.content}/shapefiles/uploaded/{sanitized_layer_name}_{sanitized_filename}.mbtiles"
        if os.path.exists(file_location):
            raise create_error(message=f"Upload failed: File {sanitized_layer_name}_{sanitized_filename}.mbtiles already exists.", code=400)
    return file_location, tmp_file_location


def write_input_file_to_disk(file: UploadFile, tmp_file_location: str):
    """
    Args:
        file: The file object containing the input file to be written to disk.
        tmp_file_location: The temporary file location where the input file will be written.

    Raises:
        Exception: If there is an error uploading the file.

    """
    if not os.path.exists(os.path.dirname(tmp_file_location)):
        os.makedirs(os.path.dirname(tmp_file_location))
    try:
        with open(tmp_file_location, "wb+") as file_object:
            file_object.write(file.file.read())
    except Exception as e:
        logger.error(e)
        raise create_error(message=f"Upload failed: There was an error uploading the file.", code=400)
    finally:
        file.file.close()


def to_mbtiles(name: str, input_file_path: str, output_file_path: str, max_zoom=17):
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
        '-dsco', 'MINZOOM=0',
        '-dsco', f'MAXZOOM={max_zoom}',
        '-dsco', f'NAME={name}',
        '-lco', f'NAME={name}',
        '-preserve_fid',
        output_file_path,
        input_file_path
    ]

    subprocess.run(cmd)


def validate_file_input(ext: str, file: UploadFile, filename_without_extension: str, layer_name: str):
    """
    Args:
        ext (str): The file extension.
        file (File): The file object.
        filename_without_extension (str): The name of the file without the extension.
        layer_name (str): The name of the layer.

    Raises:
        HTTPException: If the layer name or file name contains invalid characters.
        HTTPException: If the file extension is not supported.
        HTTPException: If the file size is too large or too small.
    """
    if not validate_input_str(layer_name):
        raise create_error(message=f"Upload failed: Layer name {layer_name} contains invalid characters.", code=400)
    if not validate_input_str(filename_without_extension):
        raise create_error(message=f"Upload failed: File name {file.filename} contains invalid characters.", code=400)
    if ext.lower() not in ['.tif', '.tiff', '.asc', '.gpkg', '.geojson', '.mbtiles']:
        raise create_error(message=f"Upload failed: File extension {ext} is not supported.", code=400)
    if file.size > 1000000000:  # 1000mb
        raise create_error(message=f"Upload failed: File size {file.size} is too large.", code=400)
    if file.size < 1000:  # 1kb
        raise create_error(message=f"Upload failed: File size {file.size} is too small.", code=400)
    check_disk_space(file.size)


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
    with rasterio.open(input_file) as src:
        target_crs = rasterio.crs.CRS.from_epsg(code=3857)
        tmp_input_file = None
        if src.crs != target_crs or nodata != src.nodata:
            transform, width, height = rasterio.warp.calculate_default_transform(src.crs, target_crs, src.width, src.height, *src.bounds)
            profile = src.profile
            profile.update(crs=target_crs, transform=transform, width=width, height=height)
            tmp_input_file = input_file.replace('.tif', '_tmp.tif')
            with rasterio.open(tmp_input_file, 'w', **profile) as dst:
                for i in range(1, src.count + 1):
                    rasterio.warp.reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=rasterio.enums.Resampling.nearest,
                        dst_nodata=nodata
                    )
    if tmp_input_file:
        os.remove(input_file)
        input_file = tmp_input_file

    translate_options = gdal.TranslateOptions(gdal.ParseCommandLine(f"-of COG -co COMPRESS=ZSTD -co BIGTIFF=YES -a_nodata {nodata} -co BLOCKSIZE=256 -co RESAMPLING=NEAREST"))
    gdal.Translate(output_file, input_file, options=translate_options)
    gdal.Info(output_file, computeMinMax=True, stats=True)
    os.remove(input_file)
    logger.info('Done transforming tif')
    return output_file


def insert_into_layer_metadata(layer_uuid: str, label: str, file_path: str, data_path: str, db: Session, is_rgb: bool = False, is_mbtile: bool = False, metadata: dict = None):
    """
    Inserts a record into the layer_metadata table.

    Args:
        layer_uuid: The UUID of the layer.
        file_path: The file path of the layer.
        db: The database connection object.
        is_rgb: Optional. Indicates whether the layer is an RGB layer. Default is False.
    """
    new_layer = LayerMetadata(layer_name=layer_uuid, label=label, file_path=file_path, data_path=data_path, is_rgb=is_rgb, is_mbtile=is_mbtile, metadata_str=str(metadata))
    db.add(new_layer)
    db.commit()
    logger.info("Record inserted into layer metadata")
