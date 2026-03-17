
import logging
from urllib.parse import quote

import httpx
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import PlainTextResponse


from guppy.config import config as cfg

# Initialize FastMCP server
mcp = FastMCP("Guppy MCP Server")

logger = logging.getLogger(__name__)


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    return PlainTextResponse("OK")


@mcp.tool()
async def get_layers(limit: int = 100, offset: int = 0, filter_query: str = None) -> str:
    """
    Get a list of all available layers in the Guppy application.
    This tool returns metadata for all possible layers, including their name, label, and additional metadata
    that describes what each layer does, its source, and other properties.

    Args:
        limit: Maximum number of layers to return.
        offset: Number of layers to skip.
        filter_query: Optional SQL-like filter string (e.g., 'layer_name LIKE "%example%"').
    """
    # Use the configured deploy path and assuming it's running on localhost in the container
    # Port 8080 is what is exposed and used in CMD in Dockerfile
    base_url = f"http://guppy:8000{cfg.deploy.path}"
    url = f"{base_url}/layers"
    params = {
        "limit": limit,
        "offset": offset,
        "filter": filter_query
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=30.0)
            response.raise_for_status()
            layers = response.json()

            # Ensure metadata is descriptive for the LLM
            formatted_layers = []
            for layer in layers:
                # Extract layer name and label
                layer_info = {
                    "layer_name": layer.get("layerName"),
                    "label": layer.get("label"),
                    "is_rgb": layer.get("isRgb"),
                    "is_mbtile": layer.get("isMbtile"),
                }

                # Include metadata if present, which often contains descriptive information
                metadata = layer.get("metadata", {})
                if metadata:
                    layer_info["description"] = metadata.get("description") or metadata.get("abstract") or "No detailed description available."
                    layer_info["metadata"] = metadata

                formatted_layers.append(layer_info)

            return str(formatted_layers)
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error calling {url}: {e}")
        return f"Error: API returned status {e.response.status_code} - {e.response.text}"
    except Exception as e:
        logger.error(f"Unexpected error in get_layers tool: {e}")
        return f"Error retrieving layers via API: {str(e)}"


@mcp.tool()
async def get_layer_stats(layer_name: str, wkt_geometry: str, srs: str = "EPSG:4326") -> str:
    """
    Get zonal statistics for a specific layer within a given WKT (Well-Known Text) geometry.
    This tool calculates zonal statistics like min, max, mean, sum, count and quantiles for the area defined by the geometry.

    Args:
        layer_name: The name of the layer to get statistics for.
        wkt_geometry: The geometry in WKT format (e.g., 'POLYGON((...))').
        srs: The spatial reference system of the WKT geometry (default: 'EPSG:4326').
    """
    base_url = f"http://localhost:8000{cfg.deploy.path}"
    url = f"{base_url}/layers/{layer_name}/stats"

    payload = {
        "geometry": wkt_geometry,
        "srs": srs
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=60.0)
            response.raise_for_status()
            stats = response.json()
            return str(stats)
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error calling {url}: {e}")
        return f"Error: API returned status {e.response.status_code} - {e.response.text}"
    except Exception as e:
        logger.error(f"Unexpected error in get_layer_stats tool: {e}")
        return f"Error retrieving statistics via API: {str(e)}"


@mcp.tool()
async def get_layer_classification(layer_name: str, wkt_geometry: str, srs: str = "EPSG:4326", all_touched: bool = False) -> str:
    """
    Get classification for a specific layer within a given WKT (Well-Known Text) geometry.
    This tool returns the count of pixels for each value in the specified area.
    This is only useful on layers that are categorical.

    Args:
        layer_name: The name of the layer to get classification for.
        wkt_geometry: The geometry in WKT format (e.g., 'POLYGON((...))').
        srs: The spatial reference system of the WKT geometry (default: 'EPSG:4326').
        all_touched: Whether to include all pixels that touch the geometry (default: False).
    """
    base_url = f"http://localhost:8000{cfg.deploy.path}"
    url = f"{base_url}/layers/{layer_name}/classification"

    payload = {
        "geometry": wkt_geometry,
        "srs": srs
    }
    params = {
        "all_touched": all_touched
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, params=params, timeout=60.0)
            response.raise_for_status()
            classification = response.json()
            return str(classification)
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error calling {url}: {e}")
        return f"Error: API returned status {e.response.status_code} - {e.response.text}"
    except Exception as e:
        logger.error(f"Unexpected error in get_layer_classification tool: {e}")
        return f"Error retrieving classification via API: {str(e)}"


@mcp.tool()
async def get_bbox_for_place(location: str, limit: int = 1) -> str:
    """
    Get the bounding box for a specific place using the Nominatim API.
    use the field bounding_box_wkt to get the bounding box in WKT format, which can be directly used in other tools that require WKT input.

    Args:
        location: The name of the location to geocode.
        limit: Maximum number of results to return (default: 1).
    """
    encoded_location = quote(location)
    url = f"https://nominatim.openstreetmap.org/search?format=json&q={encoded_location}&limit={limit}&addressdetails=1"
    headers = {"User-Agent": "Guppy-MCP-Server/1.0"}

    try:
        async with httpx.AsyncClient(verify=False) as client:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            data = response.json()

            if not data:
                return str({
                    "error": "No coordinates found for the specified location",
                    "query": location,
                    "suggestions": [
                        "Try including more specific details (e.g., state, country)",
                        "Check spelling of the location name",
                        "Use a more general location (e.g., city instead of specific address)",
                    ],
                })

            results = []
            for item in data:
                result = {
                    "latitude": float(item["lat"]),
                    "longitude": float(item["lon"]),
                    "display_name": item["display_name"],
                    "place_id": item["place_id"],
                    "type": item.get("type", ""),
                    "class": item.get("class", ""),
                    "importance": item.get("importance", 0),
                    "bounding_box": {
                        "south": float(item["boundingbox"][0]),
                        "north": float(item["boundingbox"][1]),
                        "west": float(item["boundingbox"][2]),
                        "east": float(item["boundingbox"][3]),
                    },
                    "bounding_box_wkt": f"POLYGON(({item['boundingbox'][2]} {item['boundingbox'][0]}, {item['boundingbox'][2]} {item['boundingbox'][1]}, {item['boundingbox'][3]} {item['boundingbox'][1]}, {item['boundingbox'][3]} {item['boundingbox'][0]}, {item['boundingbox'][2]} {item['boundingbox'][0]}))"
                }
                results.append(result)

            return str({
                "query": location,
                "results_count": len(results),
                "coordinates": results,
            })
    except httpx.HTTPStatusError as e:
        logger.error(f"Nominatim API error: {e.response.status_code} {e.response.reason_phrase}")
        return f"Error: Nominatim API returned status {e.response.status_code}"
    except Exception as e:
        logger.exception("Unexpected error in get_bbox_for_place tool")
        logger.error(f"type={type(e).__name__}, repr={e!r}, cause={e.__cause__!r}, context={e.__context__!r}")
        return f"Error geocoding location: {type(e).__name__}"
