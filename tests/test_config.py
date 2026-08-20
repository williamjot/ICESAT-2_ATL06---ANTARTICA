"""Testes da configuração: carga, perfis, validação e caminhos."""

import pytest
from pydantic import ValidationError

from thwaites import load_config
from thwaites.config import Config
from conftest import CONFIG_DIR


def test_load_default(cfg):
    assert isinstance(cfg, Config)
    assert cfg.season.name == "jja"
    assert cfg.season.months == [6, 7, 8]
    assert cfg.product.short_name == "ATL06"
    assert cfg.product.version == "007"
    # área na ordem esperada pelo earthaccess
    assert cfg.area.bounding_box == (cfg.area.lon_min, cfg.area.lat_min,
                                     cfg.area.lon_max, cfg.area.lat_max)
    assert cfg.temporal.temporal_range == ("2019-01-01", "2025-12-31")


def test_profile_anual_overrides(cfg_anual):
    # perfil anual sobrescreve a estação e a variante de tendência
    assert cfg_anual.season.name == "anual"
    assert len(cfg_anual.season.months) == 12
    assert cfg_anual.trend.mk_variant == "seasonal"
    # ...sem quebrar o que não foi sobrescrito
    assert cfg_anual.product.short_name == "ATL06"


def test_data_driven_fields_are_null(cfg):
    # variograma/grade do mapa são derivados dos dados -> null na config base
    assert cfg.interpolation.variogram.lag_m is None
    assert cfg.interpolation.variogram.model is None


def test_paths_resolved_and_ensured(cfg, tmp_path):
    assert cfg.paths.base_dir == tmp_path
    # `raw_temp` é COMPARTILHADO de propósito: a regra do projeto exige que a
    # pasta de download temporário esteja vazia entre execuções, e um caminho
    # por estação criaria vários lugares onde um .h5 órfão pode se esconder.
    assert cfg.paths.raw_temp == tmp_path / "data" / "raw_temp"
    # `data_dir` também é compartilhado: é onde ficam REMA, BedMachine, CATS,
    # GSFC-FDM e Caron, que não dependem da janela sazonal.
    assert cfg.paths.data_dir == tmp_path / "data"
    cfg.paths.ensure()
    assert cfg.paths.raw_temp.is_dir()
    assert cfg.paths.figures.is_dir()


def test_derived_paths_isolated_by_season(tmp_path):
    """
    Derivados de estações diferentes NÃO podem compartilhar caminho.

    Antes do isolamento, `processed`/`interim`/`tiles`/`dhdt` eram fixos: com
    40 dos 44 pipelines aceitando `--profile`, rodar um perfil novo
    sobrescrevia em silêncio o produto do anterior — inclusive o
    `dhdt_nodes.parquet` que originou o resultado de massa publicado.
    """
    from thwaites.config import Paths

    a = Paths.from_base(tmp_path, season="jja")
    b = Paths.from_base(tmp_path, season="djf")
    for field in ("processed", "interim", "tiles_dir", "dhdt_dir",
                  "qc_flags", "timeseries_dir", "outputs_dir", "tables"):
        pa, pb = getattr(a, field), getattr(b, field)
        assert pa != pb, f"'{field}' colide entre estações: {pa}"
        assert "jja" in pa.parts and "djf" in pb.parts

    # o compartilhado tem de continuar compartilhado
    assert a.data_dir == b.data_dir
    assert a.raw_temp == b.raw_temp
    assert a.logs == b.logs


def test_unknown_key_rejected(tmp_path):
    # chave estranha em subseção deve falhar (extra='forbid')
    with pytest.raises(ValidationError):
        Config(
            project={"name": "x", "chave_inexistente": 1},
            area={"lon_min": -114.5, "lon_max": -69.0, "lat_min": -80.0, "lat_max": -68.5},
            temporal={"year_start": 2019, "year_end": 2025},
            season={"name": "jja", "months": [6, 7, 8]},
            product={"short_name": "ATL06", "version": "007", "beams": ["gt1l"],
                     "segments_group": "land_ice_segments",
                     "variables": {"latitude": "latitude"},
                     "atlas_epoch_year": 2018.0, "seconds_per_year": 31557600.0},
            qc={"quality_max": 1, "fill_value": 3e30, "h_min_valid": -500.0,
                "keep_mask_values": [1]},
            tiles={"tile_km": 50, "halo_km": 15, "chunk": 1000},
            dhdt={"search_radius_m": 22000, "node_spacing_m": 5000, "min_points": 30,
                  "dt_min_years": 3.0, "dt_min_years_accel": 4.0, "t_ref": 2022.0,
                  "rate_limit": 20.0, "accel_limit": 5.0, "poly_order": 2,
                  "temp_order": 2, "use_weights": True, "max_iter": 5,
                  "n_sigma": 3.0, "resid_limit": 5.0},
            timeseries={"node_spacing_m": 10000, "search_radius_m": 30000,
                        "min_points_per_epoch": 10, "min_epochs": 4},
            interpolation={"candidates": ["idw"],
                           "cv": {"strategy": "spatial_block", "block_km": 50, "n_folds": 5},
                           "variogram": {"lag_m": None, "model": None, "max_lag_m": 150000}},
            trend={"alpha": 0.05, "fdr_method": "bh", "mk_variant": "auto"},
            mass_balance={"ice_density": 917.0, "ocean_area_m2": 3.618e14},
            logging={},
            paths={"base_dir": tmp_path, "data_dir": tmp_path, "raw_temp": tmp_path,
                   "processed": tmp_path, "interim": tmp_path, "tiles_dir": tmp_path,
                   "dhdt_dir": tmp_path, "timeseries_dir": tmp_path,
                   "outputs_dir": tmp_path, "figures": tmp_path, "tables": tmp_path,
                   "qgis": tmp_path, "logs": tmp_path},
        )


def test_area_order_validator():
    from thwaites.config import AreaCfg
    with pytest.raises(ValidationError):
        AreaCfg(lon_min=-69.0, lon_max=-114.5, lat_min=-80.0, lat_max=-68.5)
