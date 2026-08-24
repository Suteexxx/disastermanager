"""
Loads real Indian state/UT boundary polygons (from india_states.geojson,
simplified from the Survey-of-India-aligned 2019 state dataset) and
provides:

  - get_polygon(state)     -> shapely geometry of the real state shape
  - get_bounds(state)      -> (min_lon, min_lat, max_lon, max_lat)
  - list_states()          -> every state/UT name available
  - pick_resolution(state) -> a grid resolution (degrees) sized so each
                               state gets a comparable number of cells,
                               instead of a giant state getting 50x more
                               cells than a small one
  - cells_in_state(state, resolution) -> list of (lat, lon) cell centers
                               that actually fall INSIDE the real polygon
                               (not just its rectangular bounding box) --
                               this is what makes the output zones follow
                               the real coastline/border shape instead of
                               a plain rectangle.
"""
from __future__ import annotations
import json
import math
import os
from functools import lru_cache
from shapely.geometry import shape, Point
from shapely.prepared import prep

_GEOJSON_PATH = os.path.join(os.path.dirname(__file__), "india_states.geojson")


@lru_cache(maxsize=1)
def _load_all():
    with open(_GEOJSON_PATH) as f:
        data = json.load(f)
    return {feat["properties"]["name"]: shape(feat["geometry"]) for feat in data["features"]}


def list_states():
    return sorted(_load_all().keys())


def get_polygon(state):
    polys = _load_all()
    if state not in polys:
        raise KeyError(f"Unknown state '{state}'. Available: {list_states()}")
    return polys[state]


def get_bounds(state):
    return get_polygon(state).bounds  # (min_lon, min_lat, max_lon, max_lat)


def pick_resolution(state, target_cells=220, min_res=0.05, max_res=1.0):
    """Bigger states get coarser grids, smaller states finer grids, so
    every state trains on a roughly comparable number of cells instead of
    Rajasthan generating 40x more rows than Goa."""
    min_lon, min_lat, max_lon, max_lat = get_bounds(state)
    bbox_area = (max_lon - min_lon) * (max_lat - min_lat)
    # bounding-box area overestimates the real polygon area for irregular
    # shapes -- fudge factor accounts for that so target_cells lands closer
    # to the actual in-polygon cell count after filtering.
    res = math.sqrt(bbox_area / (target_cells * 1.6))
    return max(min_res, min(res, max_res))


def cells_in_state(state, resolution=None):
    """Returns [(lat, lon), ...] for grid-cell centers that fall inside the
    real state polygon -- this is the step that makes zones hug the actual
    coastline / border instead of a rectangle.

    Falls back to progressively finer resolution (and finally the polygon
    centroid) for very small or fragmented territories -- e.g. Puducherry
    is split into several small non-contiguous enclaves, so a coarse grid
    can land zero points inside any of them."""
    if resolution is None:
        resolution = pick_resolution(state)
    poly = get_polygon(state)
    prepared = prep(poly)

    for attempt_res in [resolution, resolution / 2, resolution / 4, resolution / 8]:
        min_lon, min_lat, max_lon, max_lat = poly.bounds
        lats = _frange(min_lat, max_lat, attempt_res)
        lons = _frange(min_lon, max_lon, attempt_res)
        cells = [
            (round(lat, 4), round(lon, 4))
            for lat in lats for lon in lons
            if prepared.contains(Point(lon, lat))
        ]
        if cells:
            return cells

    # last resort: use representative points of each polygon part so
    # fragmented micro-territories still get at least one cell each
    parts = poly.geoms if poly.geom_type == "MultiPolygon" else [poly]
    return [(round(p.representative_point().y, 4), round(p.representative_point().x, 4)) for p in parts]


def _frange(start, stop, step):
    n = int((stop - start) / step) + 1
    return [start + i * step for i in range(max(n, 1))]
