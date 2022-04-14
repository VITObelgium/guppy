from sqlalchemy.orm import scoped_session

from guppy2.db.db_session import SessionLocal
from guppy2.db.models import WmsFeature

db_session = scoped_session(SessionLocal)


def init_wms_features():
    get_map_feature = WmsFeature(key='getMap')
    get_attributes_feature = WmsFeature(key='getAttributes')
    get_geo_feature = WmsFeature(key='getGeometry')

    db_session.add(get_map_feature)
    db_session.add(get_attributes_feature)
    db_session.add(get_geo_feature)


    db_session.commit()


if __name__ == '__main__':
    init_wms_features()
