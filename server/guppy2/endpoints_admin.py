import logging
import os
import time

from fastapi import Response, status
from sqlalchemy.orm import Session

import guppy2.db.models as m

logger = logging.getLogger(__name__)


def delete_layer_mapping(db: Session, layer_name: str):
    t = time.time()
    layer_model = db.query(m.LayerMetadata).filter_by(layer_name=layer_name).first()
    if layer_model:
        os.remove(layer_model.file_path)
        db.delete(layer_model)
        db.commit()
        logger.info(f'delete_layer_mapping 200 {time.time() - t}')
        return status.HTTP_200_OK
    logger.info(f'get_layer_mapping 204 {time.time() - t}')
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def update_layer_mapping(db: Session, layer_name: str, file_path: str, is_rgb: bool, is_mbtile: bool):
    t = time.time()
    layer_model = db.query(m.LayerMetadata).filter_by(layer_name=layer_name).first()
    if layer_model:
        layer_model.file_path = file_path
        layer_model.is_rgb = is_rgb
        layer_model.is_mbtile = is_mbtile
        db.commit()
        logger.info(f'update_layer_mapping 200 {time.time() - t}')
        return status.HTTP_200_OK
    logger.info(f'update_layer_mapping 204 {time.time() - t}')
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def insert_layer_mapping(db: Session, layer_name: str, file_path: str, is_rgb: bool, is_mbtile: bool):
    t = time.time()
    layer_model = m.LayerMetadata(layer_name=layer_name, file_path=file_path, is_rgb=is_rgb, is_mbtile=is_mbtile)
    db.add(layer_model)
    db.commit()
    logger.info(f'insert_layer_mapping 201 {time.time() - t}')
    return status.HTTP_201_CREATED
