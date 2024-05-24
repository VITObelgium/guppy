import logging
import math
import time
from threading import Lock

import numpy as np

from guppy.db.db_session import SessionLocal
from guppy.db.models import TileStatistics

logger = logging.getLogger(__name__)


def tile2lonlat(x, y, z):
    """
    Converts tile coordinates to longitude and latitude bounds.
    Args:
        x: The x-coordinate of the tile in the tile grid.
        y: The y-coordinate of the tile in the tile grid.
        z: The zoom level of the tile.

    Returns:
        A tuple containing the longitude and latitude bounds of the tile in degrees.
        The tuple includes the left longitude, bottom latitude, right longitude, and top latitude.
    """
    n = 2.0 ** z
    lon_left = x / n * 360.0 - 180.0
    lat_bottom_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat_bottom = math.degrees(lat_bottom_rad)
    lon_right = (x + 1) / n * 360.0 - 180.0
    lat_top_rad = math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n)))
    lat_top = math.degrees(lat_top_rad)
    return (lon_left, -lat_bottom, lon_right, -lat_top)


# In-memory request counter, structured by layer_name -> z/x/y -> count
request_counter = {}
request_counter_lock = Lock()


def save_request_counts_timer():
    """
    Saves the request counts periodically at a specific interval.

    This method runs in an infinite loop and sleeps for a specified time interval before saving the request counts.
    """
    while True:
        time.sleep(60)  # Update interval, e.g., every hour
        save_request_counts()


def save_request_counts():
    """
    Saves the request counts to the database.

    This method retrieves the request counts from the request_counter dictionary,
    clears the dictionary, and then saves the counts to the database.
    Each count is associated with a tile and stored in the TileStatistics table.
    If a tile already exists in the table, its count is updated.
    If a tile does not exist, a new TileStatistics entry is created.

    """
    db = SessionLocal()
    try:
        # Perform operations with db session
        with request_counter_lock:
            counts_to_save = request_counter.copy()
            request_counter.clear()
        if counts_to_save:
            logger.info(f"Saving {len(counts_to_save)} request counts.")
        for tile, count in counts_to_save.items():
            layer_name, z, x, y = tile.split('/')
            stat = db.query(TileStatistics).filter_by(layer_name=layer_name, x=int(x), y=int(y), z=int(z)).first()
            if stat:
                stat.count += count
            else:
                db.add(TileStatistics(layer_name=layer_name, x=x, y=y, z=z, count=count))
        db.commit()
    except Exception as e:
        logger.error(f"Exception occurred: {e}")
        db.rollback()
    finally:
        db.close()


def add_item_to_request_counter(layer_name, z, x, y):
    """
    Adds an item to the request counter.

    Args:
        layer_name (str): The name of the layer.
        z (int): The Z coordinate.
        x (int): The X coordinate.
        y (int): The Y coordinate.

    """
    key = f"{layer_name}/{z}/{x}/{y}"
    with request_counter_lock:
        if key not in request_counter:
            request_counter[key] = 0
        request_counter[key] += 1


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
    rgb = np.frombuffer(data.astype('<f4').tobytes(), dtype=np.uint8).reshape(-1, 4).transpose().reshape(4, rows, cols).copy()

    rgb[3] = np.where(data == nodata, 255, rgb[3])
    rgb[2] = np.where(data == nodata, 255, rgb[2])
    rgb[1] = np.where(data == nodata, 255, rgb[1])
    rgb[0] = np.where(data == nodata, 255, rgb[0])

    return np.ma.MaskedArray(rgb)


FUNCTION_MAP = {
    'strToLower': 'LOWER',  # Converts a string to lower case
    'strToUpper': 'UPPER',  # Converts a string to upper case
    'date': 'DATE',  # Extracts the date part of a datetime
    'now': 'CURRENT_TIMESTAMP',  # Returns the current date and time
    'substring': 'SUBSTR',  # Extracts a substring from a string (parameters: string, start, length)
    'concat': '||',  # SQL concatenation operator for strings
    'length': 'LENGTH',  # Returns the length of a string
    'mathAbs': 'ABS',  # Returns the absolute value of a number
    'mathRound': 'ROUND',  # Rounds a number to the nearest integer
    'add': '+',  # Addition operator for numbers
    'subtract': '-',  # Subtraction operator for numbers
    'multiply': '*',  # Multiplication operator for numbers
    'divide': '/',  # Division operator for numbers
    # Geospatial functions (if using SpatiaLite extension or similar)
    'within': 'ST_Within',  # Checks if geometry A is within geometry B
    'intersects': 'ST_Intersects',  # Checks if geometry A intersects geometry B
    'distance': 'ST_Distance',  # Computes distance between two geometries
}


def get_field_mapping(conn):
    """
    Args:
        conn: The connection object that represents the connection to the database.

    Returns:
        A dictionary containing the field mapping.
        The keys of the dictionary are the column names in the 'tiles' table, and the values are also the column names.
        This mapping provides a simple one-to-one mapping of column names to column names.
    """
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(tiles)")
    columns = cursor.fetchall()
    return {col[1]: col[1] for col in columns}  # Simple mapping of name to name
