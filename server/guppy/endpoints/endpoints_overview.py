import ast
import gzip
import hashlib
import json
import logging
import math
import numbers
import os
import sqlite3
import tempfile
import threading
import unicodedata
from contextlib import closing

import mapbox_vector_tile
import numpy as np
import rasterio
from fastapi import HTTPException
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import Affine
from rio_tiler.utils import render, resize_array
from shapely.affinity import affine_transform
from shapely.geometry import shape
from sqlalchemy.orm import Session
from starlette.responses import Response

from guppy.db.models import LayerMetadata

logger = logging.getLogger(__name__)
MBTILES_TILE_SIZE = 256
MBTILES_OVERVIEW_LEVEL_OFFSET = 0
OVERVIEW_WIDTH = 480
OVERVIEW_HEIGHT = 320
_PREVIEW_GENERATION_LOCK = threading.Lock()


def _get_layer_source(db: Session, layer_name: str) -> tuple[str, bool, str | None]:
    layer = db.query(LayerMetadata).filter_by(layer_name=layer_name).first()
    if not layer:
        raise HTTPException(status_code=404, detail=f"Layer not found: {layer_name}")

    file_path = layer.file_path
    if not file_path:
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    preview_path = os.path.splitext(file_path)[0] + ".png"
    if not os.path.isfile(file_path) and not os.path.isfile(preview_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    return file_path, bool(layer.is_mbtile), layer.metadata_str


def _preview_path(file_path: str) -> str:
    return os.path.splitext(file_path)[0] + ".png"


def _read_cached_preview(preview_path: str) -> bytes | None:
    try:
        with open(preview_path, "rb") as preview:
            return preview.read()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not read overview PNG: {exc}") from exc


def _save_preview(preview_path: str, content: bytes) -> None:
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=os.path.dirname(preview_path) or ".",
                prefix=f".{os.path.basename(preview_path)}.",
                suffix=".tmp",
                delete=False,
        ) as temporary:
            temporary_path = temporary.name
            temporary.write(content)
        os.replace(temporary_path, preview_path)
        temporary_path = None
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not save overview PNG: {exc}") from exc
    finally:
        if temporary_path:
            try:
                os.remove(temporary_path)
            except FileNotFoundError:
                pass


def _fitted_size(source_width: int, source_height: int, width: int, height: int) -> tuple[int, int]:
    scale = min(width / source_width, height / source_height)
    return (
        max(1, min(width, round(source_width * scale))),
        max(1, min(height, round(source_height * scale))),
    )


def _render_cog_overview(
        file_path: str, width: int, height: int, metadata_str: str | None = None
) -> bytes:
    try:
        with rasterio.open(file_path) as source:
            render_width, render_height = _fitted_size(source.width, source.height, width, height)
            data = source.read(
                1,
                out_shape=(render_height, render_width),
                masked=True,
                resampling=Resampling.average,
            )
    except rasterio.errors.RasterioError as exc:
        raise HTTPException(status_code=422, detail=f"Could not read COG: {exc}") from exc

    values = np.asarray(data.data, dtype=np.float64)
    valid = ~np.ma.getmaskarray(data) & np.isfinite(values)
    image = np.full((3, render_height, render_width), 255, dtype=np.uint8)

    if np.any(valid):
        ramp = _ramp_stops(metadata_str)
        if ramp:
            quantities, colors = ramp
            for channel in range(3):
                interpolated = np.interp(values[valid], quantities, colors[:, channel])
                image[channel, valid] = np.rint(interpolated).astype(np.uint8)
        else:
            minimum = float(np.min(values[valid]))
            maximum = float(np.max(values[valid]))
            grayscale = np.zeros(np.count_nonzero(valid), dtype=np.uint8)
            if maximum > minimum:
                scaled = np.clip((values[valid] - minimum) * 255.0 / (maximum - minimum), 0, 255)
                grayscale = scaled.astype(np.uint8)
            image[:, valid] = grayscale

        rows, columns = np.nonzero(valid)
        image = image[:, rows[0]:rows[-1] + 1, columns.min():columns.max() + 1]
        render_width, render_height = _fitted_size(
            image.shape[2], image.shape[1], width, height
        )
        if image.shape[1:] != (render_height, render_width):
            image = resize_array(
                image, render_height, render_width, resampling_method="bilinear"
            )

    canvas = np.full((3, height, width), 255, dtype=np.uint8)
    x_offset = (width - render_width) // 2
    y_offset = (height - render_height) // 2
    canvas[:, y_offset:y_offset + render_height, x_offset:x_offset + render_width] = image
    return render(canvas, img_format="PNG")


def _metadata_bounds_at_zoom(bounds: str, zoom: int) -> tuple[float, float, float, float] | None:
    """Convert WGS84 MBTiles bounds to fractional XYZ tile coordinates."""
    try:
        west, south, east, north = (float(value) for value in bounds.split(","))
    except (AttributeError, TypeError, ValueError):
        return None
    if west >= east or south >= north:
        return None

    world_size = 1 << zoom

    def longitude_to_x(longitude: float) -> float:
        return (min(180.0, max(-180.0, longitude)) + 180.0) / 360.0 * world_size

    def latitude_to_y(latitude: float) -> float:
        latitude = min(85.05112878, max(-85.05112878, latitude))
        radians = math.radians(latitude)
        return (1.0 - math.asinh(math.tan(radians)) / math.pi) / 2.0 * world_size

    return longitude_to_x(west), latitude_to_y(north), longitude_to_x(east), latitude_to_y(south)


def _level_crop(level: tuple[int, int, int, int, int], bounds: str | None) -> tuple[float, float, float, float]:
    zoom, min_column, max_column, min_tms_row, max_tms_row = level
    max_index = (1 << zoom) - 1
    min_row = max_index - max_tms_row
    max_row = max_index - min_tms_row
    tile_crop = (float(min_column), float(min_row), float(max_column + 1), float(max_row + 1))
    metadata_crop = _metadata_bounds_at_zoom(bounds, zoom) if bounds else None
    if not metadata_crop:
        return tile_crop

    left = max(tile_crop[0], metadata_crop[0])
    top = max(tile_crop[1], metadata_crop[1])
    right = min(tile_crop[2], metadata_crop[2])
    bottom = min(tile_crop[3], metadata_crop[3])
    return (left, top, right, bottom) if left < right and top < bottom else tile_crop


def _select_mbtiles_level(
        levels: list[tuple[int, int, int, int, int]], bounds: str | None, width: int, height: int
) -> tuple[tuple[int, int, int, int, int], tuple[float, float, float, float], int, int]:
    """Pick two available levels less detailed than the resolution-matched level."""
    candidates = []
    for level in levels:
        crop = _level_crop(level, bounds)
        source_width = max(1, round((crop[2] - crop[0]) * MBTILES_TILE_SIZE))
        source_height = max(1, round((crop[3] - crop[1]) * MBTILES_TILE_SIZE))
        render_width, render_height = _fitted_size(source_width, source_height, width, height)
        candidate = (level, crop, render_width, render_height)
        candidates.append(candidate)
        if source_width >= render_width and source_height >= render_height:
            return candidates[max(0, len(candidates) - 1 - MBTILES_OVERVIEW_LEVEL_OFFSET)]
    return candidates[max(0, len(candidates) - 1 - MBTILES_OVERVIEW_LEVEL_OFFSET)]


def _parse_color(color: object, opacity: object = 1) -> tuple[int, int, int] | None:
    if not isinstance(color, str) or not color.startswith("#"):
        return None
    hexadecimal: str = color[1:]
    if len(hexadecimal) in (3, 4):
        hexadecimal = "".join(character * 2 for character in hexadecimal)
    if len(hexadecimal) not in (6, 8):
        return None
    try:
        red = int(hexadecimal[0:2], 16)
        green = int(hexadecimal[2:4], 16)
        blue = int(hexadecimal[4:6], 16)
        color_alpha = int(hexadecimal[6:8], 16) / 255 if len(hexadecimal) == 8 else 1
        if not isinstance(opacity, (numbers.Real, str)):
            return None
        opacity_value = float(opacity)
        if not math.isfinite(opacity_value):
            return None
        fill_opacity = min(1.0, max(0.0, opacity_value)) * color_alpha
    except (TypeError, ValueError):
        return None
    return tuple(
        round(channel * fill_opacity + 255 * (1 - fill_opacity))
        for channel in (red, green, blue)
    )


def _decode_metadata(value: object) -> object:
    """Decode JSON/Python-literal metadata, including harmless extra string encoding."""
    for _ in range(3):
        if not isinstance(value, str):
            break
        serialized = value.strip()
        if not serialized or serialized == "None":
            return None
        try:
            decoded = json.loads(serialized)
        except (json.JSONDecodeError, TypeError):
            try:
                decoded = ast.literal_eval(serialized)
            except (SyntaxError, TypeError, ValueError):
                return value
        if decoded == value:
            break
        value = decoded
    return value


def _style_definition(metadata_str: str | None) -> dict:
    metadata = _decode_metadata(metadata_str)
    if not isinstance(metadata, dict):
        return {}
    if "styling" not in metadata:
        return metadata
    styling = _decode_metadata(metadata.get("styling"))
    return styling if isinstance(styling, dict) else {}


def _ramp_stops(metadata_str: str | None) -> tuple[np.ndarray, np.ndarray] | None:
    """Return sorted quantities and RGB colors from supported COG ramp metadata."""
    definition = _style_definition(metadata_str)
    entries = definition.get("style")
    if definition.get("type") != "ramp" or not isinstance(entries, list):
        return None

    stops = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        quantity = _numeric_value(entry.get("quantity"))
        color = _parse_color(entry.get("color"))
        if quantity is not None and math.isfinite(quantity) and color is not None:
            stops[quantity] = color
    if not stops:
        return None

    ordered = sorted(stops.items())
    return (
        np.asarray([quantity for quantity, _ in ordered], dtype=np.float64),
        np.asarray([color for _, color in ordered], dtype=np.uint8),
    )


def _style_rules(metadata_str: str | None) -> list[tuple[object, tuple[int, int, int]]]:
    """Extract supported fill filters and colors from layer metadata."""
    definition = _style_definition(metadata_str)
    layers = definition.get("layers", [])
    if not isinstance(layers, list):
        logger.warning("Could not parse layer style metadata")
        return [(None, (200, 200, 200))]

    rules: list[tuple[object, tuple[int, int, int]]] = []
    for layer in layers:
        if not isinstance(layer, dict) or layer.get("type") != "fill":
            continue
        paint = layer.get("paint", {})
        if not isinstance(paint, dict):
            continue
        color = _parse_color(paint.get("fill-color"), paint.get("fill-opacity", 1))
        if color is not None:
            rules.append((layer.get("filter"), color))
    return rules or [(None, (200, 200, 200))]


def _expression_value(expression: object, properties: dict) -> object:
    if isinstance(expression, list) and expression:
        if expression[0] == "get" and len(expression) >= 2:
            property_name = expression[1]
            if property_name in properties:
                return properties[property_name]
            if isinstance(property_name, str):
                canonical_name = _canonical_property_name(property_name)
                matches = [
                    value for key, value in properties.items()
                    if isinstance(key, str) and _canonical_property_name(key) == canonical_name
                ]
                if len(matches) == 1:
                    return matches[0]
            return None
        if expression[0] == "literal" and len(expression) >= 2:
            return expression[1]
    return expression


def _canonical_property_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKC", name).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _numeric_value(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        if "," in value and "." not in value:
            value = value.replace(",", ".")
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _matches_filter(expression: object, properties: dict) -> bool:
    if expression is None:
        return True
    if not isinstance(expression, list) or not expression:
        return bool(expression)

    operator = expression[0]
    operands = expression[1:]
    if operator == "all":
        return all(_matches_filter(operand, properties) for operand in operands)
    if operator == "any":
        return any(_matches_filter(operand, properties) for operand in operands)
    if operator == "!" and len(operands) == 1:
        return not _matches_filter(operands[0], properties)
    if operator == "has" and len(operands) == 1:
        return operands[0] in properties
    if operator == "!has" and len(operands) == 1:
        return operands[0] not in properties

    values = [_expression_value(operand, properties) for operand in operands]
    try:
        if operator == "==" and len(values) == 2:
            numeric_values = [_numeric_value(value) for value in values]
            if all(value is not None for value in numeric_values):
                return numeric_values[0] == numeric_values[1]
            return values[0] == values[1]
        if operator == "!=" and len(values) == 2:
            numeric_values = [_numeric_value(value) for value in values]
            if all(value is not None for value in numeric_values):
                return numeric_values[0] != numeric_values[1]
            return values[0] != values[1]
        if operator in (">", ">=", "<", "<=") and len(values) == 2:
            left, right = (_numeric_value(value) for value in values)
            if left is None or right is None:
                return False
            return {
                ">": left > right,
                ">=": left >= right,
                "<": left < right,
                "<=": left <= right,
            }[operator]
        if operator == "in" and len(values) >= 2:
            candidates = values[1] if len(values) == 2 and isinstance(values[1], list) else values[1:]
            return values[0] in candidates
        if operator == "!in" and len(values) >= 2:
            candidates = values[1] if len(values) == 2 and isinstance(values[1], list) else values[1:]
            return values[0] not in candidates
    except TypeError:
        return False
    return False


def _is_subpixel_geometry(
        geometry_data: dict, x_scale: float, y_scale: float, minimum_pixels: float = 1.0
) -> bool:
    """Return whether a Polygon/MultiPolygon bounding box is subpixel in both axes."""
    coordinates = geometry_data.get("coordinates")
    geometry_type = geometry_data.get("type")
    if not isinstance(coordinates, (list, tuple)) or not x_scale or not y_scale:
        return False

    polygons = (coordinates,) if geometry_type == "Polygon" else coordinates
    minimum_width = minimum_pixels / abs(x_scale)
    minimum_height = minimum_pixels / abs(y_scale)
    try:
        for polygon in polygons:
            min_x = min_y = math.inf
            max_x = max_y = -math.inf
            for ring in polygon:
                for coordinate in ring:
                    x, y = coordinate[:2]
                    if x != x or y != y:
                        return False
                    if x < min_x:
                        min_x = x
                    if x > max_x:
                        max_x = x
                    if y < min_y:
                        min_y = y
                    if y > max_y:
                        max_y = y
                    if max_x - min_x >= minimum_width or max_y - min_y >= minimum_height:
                        return False
            if not all(math.isfinite(value) for value in (min_x, min_y, max_x, max_y)):
                return False
    except (IndexError, TypeError, ValueError):
        return False

    return bool(polygons)


def _render_vector_mbtiles(
        rows: list[tuple[int, int, bytes]], zoom: int, crop: tuple[float, float, float, float],
        render_width: int, render_height: int,
        style_rules: list[tuple[object, tuple[int, int, int]]],
) -> np.ndarray:
    crop_left, crop_top, crop_right, crop_bottom = crop
    crop_width = crop_right - crop_left
    crop_height = crop_bottom - crop_top
    max_index = (1 << zoom) - 1
    fills = []
    palette = {}
    feature_count = 0
    matched_feature_count = 0
    property_names = set()

    for column, tms_row, tile_data in rows:
        row = max_index - tms_row
        data = gzip.decompress(tile_data) if tile_data.startswith(b"\x1f\x8b") else tile_data
        decoded = mapbox_vector_tile.decode(data, default_options={"y_coord_down": True})
        for layer in decoded.values():
            extent = layer.get("extent", 4096)
            pixel_tolerance = max(
                crop_width * extent / render_width,
                crop_height * extent / render_height,
            )
            x_scale = render_width / (crop_width * extent)
            y_scale = render_height / (crop_height * extent)
            x_offset = (column - crop_left) * render_width / crop_width
            y_offset = (row - crop_top) * render_height / crop_height
            for feature in layer.get("features", []):
                geometry_data = feature.get("geometry")
                if not geometry_data or geometry_data.get("type") not in ("Polygon", "MultiPolygon"):
                    continue
                properties = feature.get("properties", {})
                if _is_subpixel_geometry(geometry_data, x_scale, y_scale):
                    continue
                feature_count += 1
                property_names.update(properties)
                color = next(
                    (
                        rule_color for style_filter, rule_color in reversed(style_rules)
                        if _matches_filter(style_filter, properties)
                    ),
                    None,
                )
                if color is None:
                    continue
                matched_feature_count += 1
                geometry = shape(geometry_data)
                geometry = geometry.simplify(pixel_tolerance, preserve_topology=True)
                if geometry.is_empty:
                    continue
                geometry = affine_transform(
                    geometry, [x_scale, 0.0, 0.0, y_scale, x_offset, y_offset]
                )
                palette_index = palette.setdefault(color, len(palette) + 1)
                fills.append((geometry, palette_index))

    indexes = np.zeros((render_height, render_width), dtype=np.uint16)
    if fills:
        rasterize(
            fills,
            out=indexes,
            transform=Affine.identity(),
            all_touched=True,
            dtype=np.uint16,
        )
    image = np.full((3, render_height, render_width), 255, dtype=np.uint8)
    for color, palette_index in palette.items():
        image[:, indexes == palette_index] = np.asarray(color, dtype=np.uint8)[:, np.newaxis]
    if feature_count and not matched_feature_count:
        logger.warning(
            "No polygon features matched the overview style; available PBF properties: %s",
            sorted(property_names),
        )
    return image


def _render_mbtiles_overview(
        file_path: str, width: int, height: int, metadata_str: str | None = None
) -> bytes:
    try:
        uri = f"file:{file_path}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            levels = connection.execute(
                """SELECT zoom_level, MIN(tile_column), MAX(tile_column),
                          MIN(tile_row), MAX(tile_row)
                   FROM tiles GROUP BY zoom_level ORDER BY zoom_level"""
            ).fetchall()
            if not levels:
                raise HTTPException(status_code=404, detail="No tiles found in MBTiles file")

            has_metadata = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'metadata'"
            ).fetchone()
            metadata = dict(connection.execute("SELECT name, value FROM metadata")) if has_metadata else {}
            level, crop, render_width, render_height = _select_mbtiles_level(
                levels, metadata.get("bounds"), width, height
            )
            print(f"Selected MBTiles level: {level}, crop: {crop}, render size: {render_width}x{render_height}")
            zoom = level[0]
            rows = connection.execute(
                "SELECT tile_column, tile_row, tile_data FROM tiles WHERE zoom_level = ?",
                (zoom,),
            ).fetchall()
    except HTTPException:
        raise
    except (sqlite3.Error, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Could not read MBTiles: {exc}") from exc

    try:
        overview = _render_vector_mbtiles(
            rows, zoom, crop, render_width, render_height, _style_rules(metadata_str)
        )
    except Exception as exc:
        logger.warning("Could not render MBTiles %s: %s", file_path, exc)
        raise HTTPException(status_code=422, detail=f"Could not render MBTiles: {exc}") from exc

    x_offset = (width - render_width) // 2
    y_offset = (height - render_height) // 2
    canvas = np.full((3, height, width), 255, dtype=np.uint8)
    canvas[:, y_offset:y_offset + render_height, x_offset:x_offset + render_width] = overview

    return render(canvas, img_format="PNG")


def generate_layer_preview(
        file_path: str, is_mbtile: bool, metadata_str: str | None = None
) -> bytes:
    """Generate and cache a fixed-size preview, or return the existing cache."""
    preview_path = _preview_path(file_path)
    content = _read_cached_preview(preview_path)

    if content is None:
        with _PREVIEW_GENERATION_LOCK:
            content = _read_cached_preview(preview_path)
            if content is None:
                if not os.path.isfile(file_path):
                    raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
                logger.info("Rendering overview for %s", file_path)
                if is_mbtile:
                    content = _render_mbtiles_overview(
                        file_path, OVERVIEW_WIDTH, OVERVIEW_HEIGHT, metadata_str
                    )
                else:
                    content = _render_cog_overview(
                        file_path, OVERVIEW_WIDTH, OVERVIEW_HEIGHT, metadata_str
                    )
                _save_preview(preview_path, content)
    return content


def get_layer_overview(layer_name: str, db: Session) -> Response:
    file_path, is_mbtile, metadata_str = _get_layer_source(db, layer_name)
    content = generate_layer_preview(file_path, is_mbtile, metadata_str)

    return Response(
        content,
        media_type="image/png",
        headers={
            "Cache-Control": "public, max-age=300, must-revalidate",
            "ETag": f'"{hashlib.sha256(content).hexdigest()}"',
        },
    )
