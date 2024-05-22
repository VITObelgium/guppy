import logging
import os
import time

from fastapi import Response, status
from sqlalchemy.orm import Session

import guppy.db.models as m

logger = logging.getLogger(__name__)


def delete_layer_mapping(db: Session, layer_name: str):
    """
    Deletes a layer mapping from the database and removes the associated file.

    Args:
        db: The session object for database operations.
        layer_name: The name of the layer to delete the mapping for.

    Returns:
        If the layer mapping is successfully deleted and the associated file is removed, returns HTTP status code 200 (OK).
        If the layer mapping does not exist, returns HTTP status code 204 (NO CONTENT).
    """
    t = time.time()
    layer_model = db.query(m.LayerMetadata).filter_by(layer_name=layer_name).first()
    if layer_model:
        if os.path.exists(layer_model.file_path):
            os.remove(layer_model.file_path)
        db.delete(layer_model)
        db.commit()
        logger.info(f'delete_layer_mapping 200 {time.time() - t}')
        return status.HTTP_200_OK
    logger.info(f'get_layer_mapping 204 {time.time() - t}')
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def update_layer_mapping(db: Session, layer_name: str, label: str, file_path: str, is_rgb: bool, is_mbtile: bool):
    """
    Updates the mapping of a layer in the database.

    Args:
        db (Session): The session object for interacting with the database.
        layer_name (str): The name of the layer to be updated.
        file_path (str): The file path of the layer.
        is_rgb (bool): True if the layer is in RGB format, False otherwise.
        is_mbtile (bool): True if the layer is in MBTile format, False otherwise.

    Returns:
        int: HTTP status code 200 if the layer mapping is updated successfully.
        Response: HTTP response with status code 204 if the layer mapping is not found.
    """
    t = time.time()
    layer_model = db.query(m.LayerMetadata).filter_by(layer_name=layer_name).first()
    if layer_model:
        layer_model.label = label
        layer_model.file_path = file_path
        layer_model.is_rgb = is_rgb
        layer_model.is_mbtile = is_mbtile
        db.commit()
        logger.info(f'update_layer_mapping 200 {time.time() - t}')
        return status.HTTP_200_OK
    logger.info(f'update_layer_mapping 204 {time.time() - t}')
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def insert_layer_mapping(db: Session, layer_name: str, label: str, file_path: str, is_rgb: bool, is_mbtile: bool):
    """
    Inserts a layer mapping into the database.

    Args:
        db: The session object for the database connection.
        layer_name: The name of the layer.
        file_path: The file path of the layer.
        is_rgb: A boolean indicating whether the layer is an RGB layer.
        is_mbtile: A boolean indicating whether the layer is an MBTile layer.

    Returns:
        The HTTP status code 201 indicating successful insertion.

    """
    t = time.time()
    layer_model = m.LayerMetadata(layer_name=layer_name, label=label, file_path=file_path, is_rgb=is_rgb, is_mbtile=is_mbtile)
    db.add(layer_model)
    db.commit()
    logger.info(f'insert_layer_mapping 201 {time.time() - t}')
    return status.HTTP_201_CREATED
