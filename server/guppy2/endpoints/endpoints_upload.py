# coding: utf-8
import logging
import os

import geopandas as gpd
from fastapi import UploadFile
from sqlalchemy.orm import Session

from guppy2.endpoints.endpoint_utils import validate_layer_and_get_file_path
from guppy2.endpoints.upload_utils import sanitize_input_str, check_layer_exists, create_preprocessed_layer_file, create_location_paths_and_check_if_exists, write_input_file_to_disk, \
    validate_file_input, insert_into_layer_metadata
from guppy2.error import create_error

logger = logging.getLogger(__name__)


def upload_file(layer_name: str, label: str, file: UploadFile, db: Session, is_rgb: bool = False):
    """
    Args:
        layer_name (str): The name of the layer.
        label (str): The label of the layer.
        file (UploadFile): The file to upload.
        db (Session): The database session.
        is_rgb (bool, optional): Indicates whether the file is in RGB format.

    """

    filename_without_extension, ext = os.path.splitext(file.filename)

    validate_file_input(ext, file, filename_without_extension, layer_name)

    sanitized_layer_name = sanitize_input_str(layer_name)
    sanitized_filename = sanitize_input_str(filename_without_extension)

    check_layer_exists(layer_name=f"{sanitized_layer_name}_{sanitized_filename}", db=db)

    file_location, tmp_file_location = create_location_paths_and_check_if_exists(ext, sanitized_filename, sanitized_layer_name)

    write_input_file_to_disk(file, tmp_file_location)

    is_mbtile = create_preprocessed_layer_file(ext, file_location, sanitized_filename, sanitized_layer_name, tmp_file_location)

    insert_into_layer_metadata(layer_uuid=f"{sanitized_layer_name}_{sanitized_filename}", label=label, file_path=file_location, db=db, is_rgb=is_rgb, is_mbtile=is_mbtile)
    return f"Upload successful: Layer {sanitized_layer_name}_{sanitized_filename} uploaded with label {label}."


def generate_sqlite_file(layer_name, db):
    mb_file = validate_layer_and_get_file_path(db, layer_name)
    if os.path.exists(mb_file):
        sqlite_file = mb_file.replace(".mbtiles", ".sqlite")
        df = gpd.read_file(mb_file, engine="pyogrio")
        if 'bounds' not in df.columns:
            df['bounds'] = df.geometry.envelope
        df.drop(columns=['geometry'], inplace=True)
        df.to_file(sqlite_file, index=False)
        return f"Generated sqlite file for layer {layer_name}."
    create_error(code=404, message=f"Layer {layer_name} not found.")
