"""
thwaites.grid.tiles
===================
Divide os pontos (já mascarados/corrigidos) em tiles espaciais em EPSG:3031,
COM HALO (sobreposição). Cada arquivo de tile contém os pontos do núcleo do
tile MAIS um anel de `halo_km` ao redor — assim o ajuste de dh/dt nos nós
próximos à borda tem vizinhos do tile adjacente, evitando descontinuidade
entre tiles por meio de uma região de halo.

Um manifesto (JSON) registra, por tile, os limites do NÚCLEO (onde os nós de
saída são gerados) e o arquivo. O dh/dt usa o halo só como vizinhança.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from thwaites.config import Config
from thwaites.grid.reproject import to_polar
from thwaites.logging import get_logger


def assign_xy(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Garante colunas x,y (EPSG:3031). Não recomputa se já existirem."""
    if "x" in df.columns and "y" in df.columns:
        return df
    x, y = to_polar(df["lon"].to_numpy(), df["lat"].to_numpy(), cfg)
    df = df.copy()
    df["x"] = x
    df["y"] = y
    return df


def build_tiles(df: pd.DataFrame, cfg: Config, out_dir: Path | None = None) -> list[dict]:
    """
    Escreve um Parquet por tile (núcleo + halo) e um manifesto JSON.

    Retorna a lista de dicts do manifesto (um por tile ativo).
    """
    logger = get_logger()
    out_dir = Path(out_dir) if out_dir else cfg.paths.tiles_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = assign_xy(df, cfg)
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    tile_m = cfg.tiles.tile_m
    halo_m = cfg.tiles.halo_m

    # Origem da grade de tiles (alinhada a múltiplos de tile_m).
    x0 = np.floor(x.min() / tile_m) * tile_m
    y0 = np.floor(y.min() / tile_m) * tile_m

    # Índice de tile (núcleo) de cada ponto.
    ti = np.floor((x - x0) / tile_m).astype(int)
    tj = np.floor((y - y0) / tile_m).astype(int)

    active = sorted(set(zip(ti.tolist(), tj.tolist())))
    logger.info(f"Tiles ativos (com núcleo): {len(active)} "
                f"(tile={cfg.tiles.tile_km} km, halo={cfg.tiles.halo_km} km)")

    manifest: list[dict] = []
    for (i, j) in active:
        xc0, xc1 = x0 + i * tile_m, x0 + (i + 1) * tile_m
        yc0, yc1 = y0 + j * tile_m, y0 + (j + 1) * tile_m

        # Seleção com halo (núcleo ± halo_m).
        sel = ((x >= xc0 - halo_m) & (x < xc1 + halo_m) &
               (y >= yc0 - halo_m) & (y < yc1 + halo_m))
        n_core = int(((x >= xc0) & (x < xc1) & (y >= yc0) & (y < yc1)).sum())
        if n_core == 0:
            continue

        tile_df = df.loc[sel]
        name = f"tile_{i:04d}_{j:04d}"
        fpath = out_dir / f"{name}.parquet"
        tile_df.to_parquet(fpath, index=False, engine="pyarrow", compression="snappy")

        manifest.append({
            "tile": name,
            "file": fpath.name,
            "x_min": float(xc0), "x_max": float(xc1),
            "y_min": float(yc0), "y_max": float(yc1),
            "n_core": n_core,
            "n_with_halo": int(len(tile_df)),
        })

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)
    logger.info(f"Manifesto -> {manifest_path} ({len(manifest)} tiles)")
    return manifest


def build_tiles_streaming(src_path, cfg: Config, out_dir: Path | None = None,
                          batch_rows: int = 2_000_000) -> list[dict]:
    """
    Versão de MEMÓRIA LIMITADA do tiling: uma passagem sobre o Parquet, com um
    writer aberto por tile.

    Por que existe: `build_tiles` carrega a tabela inteira (20 M × 17 ≈ 1,1 GB),
    `assign_xy` copia e cada seleção por tile copia de novo — pico bem acima dos
    ~3 GB livres da máquina alvo. Aqui a memória fica no tamanho do lote.

    Duas passagens leves:
      1. só x/y (streaming) para achar a extensão e os tiles ativos;
      2. lotes completos distribuídos aos writers dos tiles (núcleo + halo).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from thwaites.io.memory import iter_points, free_memory_gb

    logger = get_logger()
    out_dir = Path(out_dir) if out_dir else cfg.paths.tiles_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    src_path = Path(src_path)
    names = pq.ParquetFile(src_path).schema_arrow.names
    if "x" not in names or "y" not in names:
        raise ValueError(f"{src_path.name} sem colunas x/y — rode run_slope.py "
                         f"(ou use build_tiles, que as calcula).")

    tile_m, halo_m = cfg.tiles.tile_m, cfg.tiles.halo_m

    # --- passagem 1: extensão e contagem por tile (só x/y) ------------------
    logger.info("tiling p1/2: extensão e tiles ativos (só x/y)...")
    x_min = y_min = np.inf
    x_max = y_max = -np.inf
    for df in iter_points(src_path, ["x", "y"], batch_rows=batch_rows, do_downcast=False):
        x_min = min(x_min, float(df["x"].min())); x_max = max(x_max, float(df["x"].max()))
        y_min = min(y_min, float(df["y"].min())); y_max = max(y_max, float(df["y"].max()))
    x0 = np.floor(x_min / tile_m) * tile_m
    y0 = np.floor(y_min / tile_m) * tile_m

    core_counts: dict[tuple[int, int], int] = {}
    for df in iter_points(src_path, ["x", "y"], batch_rows=batch_rows, do_downcast=False):
        ti = np.floor((df["x"].to_numpy() - x0) / tile_m).astype(int)
        tj = np.floor((df["y"].to_numpy() - y0) / tile_m).astype(int)
        for k, c in zip(*np.unique(np.c_[ti, tj], axis=0, return_counts=True)):
            key = (int(k[0]), int(k[1]))
            core_counts[key] = core_counts.get(key, 0) + int(c)
    active = sorted(core_counts)
    logger.info(f"tiling: {len(active)} tiles ativos (tile {cfg.tiles.tile_km} km, "
                f"halo {cfg.tiles.halo_km} km) | livre {free_memory_gb():.1f} GB")

    # --- passagem 2: distribui os lotes -------------------------------------
    bounds = {}
    writers: dict[tuple[int, int], object] = {}
    halo_counts = {k: 0 for k in active}
    for (i, j) in active:
        bounds[(i, j)] = (x0 + i * tile_m, x0 + (i + 1) * tile_m,
                          y0 + j * tile_m, y0 + (j + 1) * tile_m)

    logger.info("tiling p2/2: distribuindo pontos em streaming...")
    try:
        done = 0
        for df in iter_points(src_path, list(names), batch_rows=batch_rows,
                              do_downcast=False):
            xv = df["x"].to_numpy(); yv = df["y"].to_numpy()
            for key in active:
                xc0, xc1, yc0, yc1 = bounds[key]
                sel = ((xv >= xc0 - halo_m) & (xv < xc1 + halo_m) &
                       (yv >= yc0 - halo_m) & (yv < yc1 + halo_m))
                if not sel.any():
                    continue
                sub = df.loc[sel]
                table = pa.Table.from_pandas(sub, preserve_index=False)
                w = writers.get(key)
                if w is None:
                    p = out_dir / f"tile_{key[0]:04d}_{key[1]:04d}.parquet"
                    w = pq.ParquetWriter(p, table.schema, compression="snappy")
                    writers[key] = w
                w.write_table(table)
                halo_counts[key] += len(sub)
                del sub, table
            done += len(df)
            logger.info(f"  {done:,} linhas distribuídas "
                        f"(livre {free_memory_gb():.1f} GB)")
    finally:
        for w in writers.values():
            try:
                w.close()
            except Exception:
                pass

    manifest = []
    for key in active:
        if key not in writers:
            continue
        xc0, xc1, yc0, yc1 = bounds[key]
        manifest.append({
            "tile": f"tile_{key[0]:04d}_{key[1]:04d}",
            "file": f"tile_{key[0]:04d}_{key[1]:04d}.parquet",
            "x_min": float(xc0), "x_max": float(xc1),
            "y_min": float(yc0), "y_max": float(yc1),
            "n_core": int(core_counts[key]),
            "n_with_halo": int(halo_counts[key]),
        })
    mpath = out_dir / "manifest.json"
    with open(mpath, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)
    logger.info(f"Manifesto -> {mpath} ({len(manifest)} tiles)")
    return manifest


def load_manifest(cfg: Config, tiles_dir: Path | None = None) -> list[dict]:
    tiles_dir = Path(tiles_dir) if tiles_dir else cfg.paths.tiles_dir
    with open(tiles_dir / "manifest.json", "r", encoding="utf-8") as fp:
        return json.load(fp)
