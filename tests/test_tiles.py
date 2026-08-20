"""Testes do tiling com halo (grid/tiles)."""

import numpy as np
import pandas as pd

from thwaites.grid.tiles import build_tiles, load_manifest


def _grid_df(n_side=40, extent=100_000.0):
    """Grade regular de pontos em x,y (3031) cobrindo [0, extent]²."""
    xs = np.linspace(1_000, extent - 1_000, n_side)
    ys = np.linspace(1_000, extent - 1_000, n_side)
    X, Y = np.meshgrid(xs, ys)
    n = X.size
    return pd.DataFrame({
        "x": X.ravel(), "y": Y.ravel(),
        "h_corr": np.zeros(n), "t_year": np.full(n, 2022.0),
        "s_elv": np.full(n, 0.1),
    })


def test_tiles_cover_and_core_partition(cfg, tmp_path):
    # tile 50 km, halo 15 km (defaults). Extensão 100 km -> grade 2x2 de tiles.
    df = _grid_df()
    manifest = build_tiles(df, cfg, out_dir=tmp_path / "tiles")

    assert len(manifest) == 4  # 2x2
    # cada núcleo particiona os pontos: soma dos núcleos == total
    assert sum(e["n_core"] for e in manifest) == len(df)
    # halo duplica pontos -> soma com halo é estritamente maior
    assert sum(e["n_with_halo"] for e in manifest) > len(df)
    for e in manifest:
        assert e["n_with_halo"] >= e["n_core"] > 0
        assert (tmp_path / "tiles" / e["file"]).exists()


def test_manifest_roundtrip(cfg, tmp_path):
    df = _grid_df(n_side=20)
    build_tiles(df, cfg, out_dir=tmp_path / "tiles")
    man = load_manifest(cfg, tiles_dir=tmp_path / "tiles")
    assert isinstance(man, list) and len(man) >= 1
    assert {"tile", "file", "x_min", "x_max", "n_core"} <= set(man[0].keys())
