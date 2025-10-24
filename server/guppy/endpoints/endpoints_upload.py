# coding: utf-8
import logging
import os

import geopandas as gpd
from fastapi import UploadFile
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from guppy.endpoints.endpoint_utils import validate_layer_and_get_file_path
from guppy.endpoints.upload_utils import sanitize_input_str, check_layer_exists, create_preprocessed_layer_file, create_location_paths_and_check_if_exists, write_input_file_to_disk, \
    validate_file_input, insert_into_layer_metadata
from guppy.error import create_error

logger = logging.getLogger(__name__)


def upload_file(layer_name: str, label: str, file: UploadFile, data: UploadFile | None, db: Session, is_rgb: bool = False,
                max_zoom: int = 17, metadata: dict = None, process: bool = True):
    """
    Args:
        layer_name (str): The name of the layer.
        label (str): The label of the layer.
        file (UploadFile): The file to upload.
        data (UploadFile): The data to upload (if the input is a mbtile raster, we need the data geotiff aswell for sampling functions).
        db (Session): The database session.
        is_rgb (bool, optional): Indicates whether the file is in RGB format.
        max_zoom (int, optional): The maximum zoom level of the layer.
        metadata (dict, optional): The metadata of the layer.

    """
    data_location = None
    filename_without_extension, ext = os.path.splitext(file.filename)
    if data:
        dataname_without_extension, d_ext = os.path.splitext(data.filename)
        sanitized_dataname = sanitize_input_str(filename_without_extension)
        validate_file_input(d_ext, data, dataname_without_extension, layer_name)

    validate_file_input(ext, file, filename_without_extension, layer_name)

    sanitized_layer_name = sanitize_input_str(layer_name)
    sanitized_filename = sanitize_input_str(filename_without_extension)

    check_layer_exists(layer_name=f"{sanitized_layer_name}", db=db)

    file_location, tmp_file_location = create_location_paths_and_check_if_exists(ext, sanitized_filename, sanitized_layer_name, is_raster=True if data else False)
    write_input_file_to_disk(file, tmp_file_location if process else file_location)

    if data:
        data_location, tmp_data_location = create_location_paths_and_check_if_exists(d_ext, sanitized_dataname, sanitized_layer_name, is_raster=True)
        write_input_file_to_disk(data, data_location)

    is_mbtile = create_preprocessed_layer_file(ext, file_location, sanitized_filename, sanitized_layer_name, tmp_file_location, max_zoom, process=process)

    insert_into_layer_metadata(layer_uuid=sanitized_layer_name, label=label, file_path=file_location, data_path=data_location, db=db, is_rgb=is_rgb, is_mbtile=is_mbtile, metadata=metadata)
    return f"Upload successful: Layer {sanitized_layer_name} uploaded with label {label}."


def generate_sqlite_file(layer_name, db):
    logger.info(f"Generating sqlite file for layer {layer_name}")
    mb_file = validate_layer_and_get_file_path(db, layer_name)
    if os.path.exists(mb_file):
        sqlite_file = mb_file.replace(".mbtiles", ".sqlite")
        df = gpd.read_file(mb_file, engine="pyogrio")
        if 'bounds' not in df.columns:
            bounds_df = df.bounds
            df['bounds'] = bounds_df.apply(lambda row: f"{round(row['minx'], 8)}, {round(row['miny'], 8)}, {round(row['maxx'], 8)}, {round(row['maxy'], 8)}", axis=1)
        df.drop(columns=['geometry'], inplace=True)
        df.to_sql('tiles', con=create_engine(f'sqlite:///{sqlite_file}'), if_exists='replace', index=False)
        return f"Generated sqlite file for layer {layer_name}."
    create_error(code=404, message=f"Layer {layer_name} not found.")
