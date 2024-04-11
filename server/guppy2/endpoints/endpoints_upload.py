# coding: utf-8
import logging
import os

from fastapi import UploadFile
from sqlalchemy.orm import Session

from guppy2.endpoints.upload_utils import sanitize_input_str, check_layer_exists, create_preprocessed_layer_file, create_location_paths_and_check_if_exists, write_input_file_to_disk, \
    validate_file_input, insert_into_layer_metadata

logger = logging.getLogger(__name__)


def upload_file(layer_name: str, file: UploadFile, db: Session, is_rgb: bool = False):
    """
    Args:
        layer_name (str): The name of the layer.
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

    insert_into_layer_metadata(layer_uuid=f"{sanitized_layer_name}_{sanitized_filename}", file_path=file_location, db=db, is_rgb=is_rgb, is_mbtile=is_mbtile)
    return f"Upload successful: Layer {sanitized_layer_name}_{sanitized_filename} uploaded."
