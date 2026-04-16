"""
map_utils.py
地图工具：加载 GeoJSON 图层 + 点在多边形内判断（point-in-polygon）

提供:
    load_all_layers()         -> dict of GeoJSON dicts (cached)
    identify_region(lat, lon) -> {tribes, counties, cities, states}
    build_folium_map(marker_latlon=None) -> folium.Map
"""

import json
import os
from functools import lru_cache

import folium
from shapely.geometry import Point, shape

_BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
GEOJSON_DIR = os.path.join(_BASE_DIR, "GeoJson_Data")

_LAYER_FILES = {
    "state":       os.path.join(GEOJSON_DIR, "state_layer.geojson"),
    "tribes_az":   os.path.join(GEOJSON_DIR, "Tribes_in_Arizona.geojson"),
    "tribes_nm":   os.path.join(GEOJSON_DIR, "Tribes_in_New_Mexico.geojson"),
    "tribes_ok":   os.path.join(GEOJSON_DIR, "Tribes_in_Oklahoma.geojson"),
    "counties_az": os.path.join(GEOJSON_DIR, "Counties_in_Arizona.geojson"),
    "counties_nm": os.path.join(GEOJSON_DIR, "Counties_in_New_Mexico.geojson"),
    "counties_ok": os.path.join(GEOJSON_DIR, "Counties_in_Oklahoma.geojson"),
    "cities_az":   os.path.join(GEOJSON_DIR, "Cities_in_Arizona.geojson"),
    "cities_nm":   os.path.join(GEOJSON_DIR, "Cities_in_New_Mexico.geojson"),
    "cities_ok":   os.path.join(GEOJSON_DIR, "Cities_in_Oklahoma.geojson"),
}

def _load(path: str) -> dict:
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        return json.load(f)

@lru_cache(maxsize=1)
def load_all_layers() -> dict:
    return {k: _load(v) for k, v in _LAYER_FILES.items()}

# ─── City name list (for dropdown) ───────────────────────────────────────────
def get_all_cities() -> list[str]:
    layers = load_all_layers()
    cities = set()
    for key in ("cities_az", "cities_nm", "cities_ok"):
        for feat in layers[key]["features"]:
            name = feat["properties"].get("NAME", "")
            if name and len(name.strip()) > 2:
                cities.add(name.strip())
    return sorted(cities)

# ─── Point-in-polygon ─────────────────────────────────────────────────────────
def identify_region(lat: float, lon: float) -> dict:
    """
    Given a lat/lon, return which state/tribe/county/city it falls in.
    Returns {tribes: [], counties: [], cities: [], states: []}
    """
    pt = Point(lon, lat)   # shapely uses (x=lon, y=lat)
    layers = load_all_layers()

    tribes, counties, cities, states = [], [], [], []

    for feat in layers["state"]["features"]:
        try:
            if shape(feat["geometry"]).contains(pt):
                states.append(feat["properties"]["NAME"])
        except Exception:
            pass

    for key in ("tribes_az", "tribes_nm", "tribes_ok"):
        for feat in layers[key]["features"]:
            try:
                if shape(feat["geometry"]).contains(pt):
                    name = feat["properties"].get("NAME", "")
                    if name:
                        tribes.append(name)
            except Exception:
                pass

    for key in ("counties_az", "counties_nm", "counties_ok"):
        for feat in layers[key]["features"]:
            try:
                if shape(feat["geometry"]).contains(pt):
                    name = feat["properties"].get("NAME", "")
                    if name:
                        counties.append(name)
            except Exception:
                pass

    for key in ("cities_az", "cities_nm", "cities_ok"):
        for feat in layers[key]["features"]:
            try:
                if shape(feat["geometry"]).contains(pt):
                    name = feat["properties"].get("NAME", "")
                    if name:
                        cities.append(name)
            except Exception:
                pass

    return {"tribes": tribes, "counties": counties, "cities": cities, "states": states}

# ─── Conflict detection ───────────────────────────────────────────────────────
def detect_conflict(map_geo: dict, dropdown_geo: dict) -> str | None:
    level_labels = {"states": "State", "tribes": "Tribe", "counties": "County", "cities": "City"}
    for level, label in level_labels.items():
        m_vals = set(map_geo.get(level, []))
        d_vals = set(dropdown_geo.get(level, []))
        if m_vals and d_vals and not m_vals & d_vals:
            return (
                f"**{label} conflict:** Map location says **{', '.join(sorted(m_vals))}** "
                f"but dropdown says **{', '.join(sorted(d_vals))}**. "
                "Please clear the map marker or reset the dropdown."
            )
    return None

# ─── Folium map builder ───────────────────────────────────────────────────────
_TRIBE_COLORS = {"tribes_az": "#3182bd", "tribes_nm": "#31a354", "tribes_ok": "#e6550d"}
_TRIBE_LABELS = {"tribes_az": "AZ Tribes", "tribes_nm": "NM Tribes", "tribes_ok": "OK Tribes"}
_CITY_LABELS  = {"cities_az": "AZ Cities", "cities_nm": "NM Cities", "cities_ok": "OK Cities"}

def build_folium_map(marker_latlon: tuple | None = None) -> folium.Map:
    layers = load_all_layers()

    m = folium.Map(
        location=[35.5, -107.5],
        zoom_start=6,
        tiles="CartoDB positron",
    )

    # State boundaries
    folium.GeoJson(
        layers["state"],
        name="State Boundaries",
        style_function=lambda _: {
            "fillColor": "#f0f0f0", "color": "#555", "weight": 2, "fillOpacity": 0.1
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["NAME", "PolicyNum", "AllNum"],
            aliases=["State:", "State Policies:", "Total Policies:"],
            localize=True,
        ),
    ).add_to(m)

    # County boundaries
    county_group = folium.FeatureGroup(name="Counties", show=True)
    for key in ("counties_az", "counties_nm", "counties_ok"):
        folium.GeoJson(
            layers[key],
            style_function=lambda _: {
                "fillColor": "#ffffcc", "color": "#999", "weight": 1, "fillOpacity": 0.3
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["NAMELSAD", "PolicyNum"],
                aliases=["County:", "Policies:"],
                localize=True,
            ),
        ).add_to(county_group)
    county_group.add_to(m)

    # Tribal boundaries (per-state groups for toggling)
    for key, color in _TRIBE_COLORS.items():
        group = folium.FeatureGroup(name=_TRIBE_LABELS[key], show=True)
        folium.GeoJson(
            layers[key],
            style_function=lambda _, c=color: {
                "fillColor": c, "color": c, "weight": 1.5, "fillOpacity": 0.45
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["NAME", "PolicyNum"],
                aliases=["Tribe:", "Policies:"],
                localize=True,
            ),
        ).add_to(group)
        group.add_to(m)

    # City boundaries (default hidden to keep map clean)
    for key, label in _CITY_LABELS.items():
        group = folium.FeatureGroup(name=label, show=False)
        folium.GeoJson(
            layers[key],
            style_function=lambda _: {
                "fillColor": "#9e9ac8", "color": "#756bb1", "weight": 1, "fillOpacity": 0.35
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["NAME", "PolicyNum"],
                aliases=["City:", "Policies:"],
                localize=True,
            ),
        ).add_to(group)
        group.add_to(m)

    # Clicked marker
    if marker_latlon:
        folium.Marker(
            location=marker_latlon,
            icon=folium.Icon(color="red", icon="map-marker", prefix="fa"),
            popup="Clicked location",
            tooltip="Clicked location",
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m
