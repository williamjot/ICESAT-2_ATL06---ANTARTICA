"""
thwaites.grid.reproject
=======================
Conversões entre coordenadas geográficas (EPSG:4326) e polares
estereográficas (EPSG:3031). Transformers em cache (criar é caro).
"""

from __future__ import annotations

from functools import lru_cache

from pyproj import Transformer

from thwaites.config import Config


@lru_cache(maxsize=8)
def _transformer(src_epsg: int, dst_epsg: int) -> Transformer:
    return Transformer.from_crs(f"EPSG:{src_epsg}", f"EPSG:{dst_epsg}", always_xy=True)


def to_polar(lon, lat, cfg: Config):
    """lon/lat (graus) -> x/y (m, EPSG:3031)."""
    return _transformer(cfg.area.epsg_lonlat, cfg.area.epsg_polar).transform(lon, lat)


def to_lonlat(x, y, cfg: Config):
    """x/y (m, EPSG:3031) -> lon/lat (graus)."""
    return _transformer(cfg.area.epsg_polar, cfg.area.epsg_lonlat).transform(x, y)
