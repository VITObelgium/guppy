# coding: utf-8
import logging
import os
import re
import shutil

import rasterio
from fastapi import UploadFile
from osgeo import gdal
from sqlalchemy.orm import Session

from guppy2.db.models import LayerMetadata
from guppy2.error import create_error

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
    sanitized_input = re.sub(r'[^a-zA-Z0-9_]', '', input)
    return sanitized_input


def validate_input_str(input: str) -> bool:
    """
    Args:
        input (str): The input string to be validated.

    Returns:
        bool: Returns True if the input string contains only alphanumeric characters and underscores, otherwise False.
    """
    if re.search(r'[^a-zA-Z0-9_]', input):
        return False
    return True


def check_disk_space(temp_file_size: int):
    """Checks the disk space usage to determine if it would exceed 90% after file upload.

    Args:
        temp_file_size (int): The size of the temporary file to be uploaded.

    Raises:
        Error: If the disk space usage would exceed 90% after file upload.

    """
    total, used, free = shutil.disk_usage("c:/temp/content")
    used_percentage = ((used + temp_file_size) / total) * 100
    if used_percentage > 90:
        raise create_error(code=507, message="Upload failed: Disk space usage would exceed 90% after file upload")


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


def upload_file(layer_name: str, file: UploadFile, db: Session, is_rgb: bool = False):
    """
    Args:
        layer_name: A string representing the name of the layer.
        file: An instance of the UploadFile class representing the file to be uploaded.
        db: An instance of the Session class representing the database session.
        is_rgb: A boolean representing whether the file is an RGB image. Default is False.
    Raises:
        HTTPException: If there was an error uploading the file.
        HTTPException: If there was an error validating the file.
    """

    filename_without_extension, ext = os.path.splitext(file.filename)

    validate_file_input(ext, file, filename_without_extension, layer_name)

    sanitized_layer_name = sanitize_input_str(layer_name)
    sanitized_filename = sanitize_input_str(filename_without_extension)

    check_layer_exists(layer_name=f"{sanitized_layer_name}_{sanitized_filename}", db=db)

    tmp_file_location = f"c:/temp/content/tifs/uploaded/{sanitized_layer_name}_{sanitized_filename}_tmp.{ext}"
    file_location = f"c:/temp/content/tifs/uploaded/{sanitized_layer_name}_{sanitized_filename}.tif"
    if os.path.exists(file_location):
        raise create_error(message=f"Upload failed: File {sanitized_layer_name}_{sanitized_filename}.tif already exists.", code=400)
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
    error_list = validate_raster(file_path=tmp_file_location)
    if error_list:
        raise create_error(message=f"Upload failed: {', '.join(error_list)}", code=400)
    else:
        save_geotif_tiled_overviews(input_file=tmp_file_location, output_file=file_location, nodata=-9999)
        insert_into_layer_metadata(layer_uuid=f"{sanitized_layer_name}_{sanitized_filename}", file_path=file_location, db=db, is_rgb=is_rgb)


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
    if ext not in ['.tif', '.tiff', '.asc']:
        raise create_error(message=f"Upload failed: File extension {ext} is not supported.", code=400)
    if file.size > 100000000:  # 100mb
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
    translate_options = gdal.TranslateOptions(gdal.ParseCommandLine(f"-of Gtiff -co COMPRESS=DEFLATE -co TILED=YES -co BIGTIFF=YES -a_nodata {nodata}"))
    gdal.Translate(output_file, input_file, options=translate_options)
    image = gdal.Open(output_file, 1)
    gdal.SetConfigOption('COMPRESS_OVERVIEW', 'DEFLATE')
    algo = 'NEAREST'
    image.BuildOverviews(algo, [2, 4, 8, 16, 32, 64, 128, 256, 512])
    del image
    os.remove(input_file)
    logger.info('Done transforming tif')
    return output_file


def insert_into_layer_metadata(layer_uuid: str, file_path: str, db: Session, is_rgb: bool = False):
    """
    Inserts a record into the layer_metadata table.

    Args:
        layer_uuid: The UUID of the layer.
        file_path: The file path of the layer.
        db: The database connection object.
        is_rgb: Optional. Indicates whether the layer is an RGB layer. Default is False.
    """
    new_layer = LayerMetadata(layer_name=layer_uuid, file_path=file_path, is_rgb=is_rgb)
    db.add(new_layer)
    db.commit()
    logger.info("Record inserted into layer metadata")
