"""
thwaites.viz.basemap
====================
Base cartográfica: sombreamento do REMA e contornos (linha de costa, linha de
aterramento, frentes de calving datadas).

Escolhas
--------
* O relevo sombreado vem do **REMA** já em disco, lido DECIMADO. O mosaico tem
  22.729 x 21.173 px a 32 m (640 MB); lê-lo inteiro esgotaria a memória da
  máquina. Para um mapa de 20 x 20 cm, ~500 m por pixel já satura a resolução
  de impressão.
* Os contornos são derivados do **BedMachine**, a MESMA fonte da máscara
  científica. Usar um shapefile externo de costa introduziria uma geometria
  inconsistente com a que define os produtos — a linha desenhada passaria a
  discordar da linha usada no cálculo.
* As frentes de calving vêm do IceLines, com época declarada, porque a frente
  migra e desenhar uma única linha sem data seria enganoso.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from thwaites.config import Config
from thwaites.logging import get_logger


def load_hillshade(cfg: Config, x0, x1, y0, y1, target_px: int = 1400,
                   azdeg: float = 315.0, altdeg: float = 45.0):
    """
    Relevo sombreado do REMA na janela pedida.

    Devolve (extent, hillshade, elev). A decimação é escolhida para que o
    resultado tenha ~`target_px` de lado — leitura por janela + `out_shape`, de
    modo que o mosaico completo nunca é materializado.
    """
    import rasterio
    from rasterio.windows import from_bounds
    from matplotlib.colors import LightSource

    logger = get_logger()
    path = cfg.paths.data_dir / Path(cfg.slope.rema_path).name
    if not path.exists():
        raise FileNotFoundError(f"REMA não encontrado: {path}")

    with rasterio.open(path) as src:
        win = from_bounds(x0, y0, x1, y1, transform=src.transform)
        win = win.intersection(
            rasterio.windows.Window(0, 0, src.width, src.height))
        h = int(min(target_px, max(win.height, 1)))
        w = int(min(target_px, max(win.width, 1)))
        elev = src.read(1, window=win, out_shape=(h, w),
                        resampling=rasterio.enums.Resampling.average).astype(float)
        nodata = src.nodata
        bounds = rasterio.windows.bounds(win, src.transform)
    if nodata is not None:
        elev[elev == nodata] = np.nan
    elev[elev < -100] = np.nan

    logger.info(f"REMA para basemap: {elev.shape} "
                f"(~{(bounds[2]-bounds[0])/w:.0f} m/px)")

    # Preenche NaN apenas para o cálculo do gradiente (evita buracos no
    # sombreamento), mas RESTAURA o NaN depois: sem isso o oceano — 17% do
    # mosaico — é renderizado como terreno cinza, o que inventa relevo onde há
    # água e engana o leitor sobre a extensão do gelo.
    valid = np.isfinite(elev)
    filled = np.where(valid, elev, np.nanmedian(elev))
    ls = LightSource(azdeg=azdeg, altdeg=altdeg)
    hs = ls.hillshade(filled, vert_exag=3.0,
                      dx=(bounds[2] - bounds[0]) / w,
                      dy=(bounds[3] - bounds[1]) / h)
    hs = np.where(valid, hs, np.nan)
    extent = (bounds[0] / 1000, bounds[2] / 1000,
              bounds[1] / 1000, bounds[3] / 1000)   # km
    return extent, hs, elev


def mask_contours(cfg: Config, x0, x1, y0, y1, decimate: int = 2):
    """
    Campos binários do BedMachine para desenhar contornos.

    Devolve (x_km, y_km, dict) com máscaras de `ice` (aterrado+flutuante),
    `grounded` e `floating` — cujos limites dão, respectivamente, a linha de
    costa, a linha de aterramento e a extensão das plataformas.
    """
    import xarray as xr

    cands = sorted(cfg.paths.data_dir.glob("*BedMachineAntarctica*.nc"))
    if not cands:
        raise FileNotFoundError("BedMachine .nc não encontrado em data/.")
    ds = xr.open_dataset(cands[0])
    bx = np.asarray(ds["x"].values, float)
    by = np.asarray(ds["y"].values, float)
    ix = np.where((bx >= x0) & (bx <= x1))[0][::decimate]
    iy = np.where((by >= y0) & (by <= y1))[0][::decimate]
    m = np.asarray(ds["mask"].isel(x=ix, y=iy).values)
    sx, sy = bx[ix], by[iy]
    ds.close()
    if sy[0] > sy[-1]:
        sy, m = sy[::-1], m[::-1, :]

    return sx / 1000, sy / 1000, {
        "ice": np.isin(m, (1, 2, 3, 4)).astype(float),
        "grounded": (m == 2).astype(float),
        "floating": (m == 3).astype(float),
    }


def draw_basemap(ax, cfg: Config, x0, x1, y0, y1, *, hillshade: bool = True,
                 coastline: bool = True, grounding_line: bool = True,
                 shelf_fill: bool = True, target_px: int = 1400):
    """
    Desenha a base num eixo: sombreado, plataformas em tom frio, linha de costa
    e linha de aterramento. Devolve o `extent` em km.
    """
    ext = (x0 / 1000, x1 / 1000, y0 / 1000, y1 / 1000)
    # fundo de oceano — o que não for coberto pelo relevo fica água, não vazio
    ax.set_facecolor("#dfeaf4")
    # A cobertura REMA recortada pode ter lacunas retangulares de tiles. Sem
    # uma base categórica por baixo, essas lacunas herdavam a cor do oceano e
    # pareciam plataformas/água. O BedMachine define primeiro toda a área de
    # gelo em cinza; depois hillshade e floating refinam a aparência.
    cartography = coastline or grounding_line or shelf_fill
    gx = gy = M = None
    if cartography:
        gx, gy, M = mask_contours(cfg, x0, x1, y0, y1)
        ax.contourf(gx, gy, M["ice"], levels=[0.5, 1.5],
                    colors=["#b8b8b8"], alpha=1.0, zorder=0.5)
    if hillshade:
        try:
            ext, hs, _ = load_hillshade(cfg, x0, x1, y0, y1, target_px=target_px)
            ax.imshow(hs, extent=ext, cmap="gray", vmin=0.05, vmax=1.30,
                      origin="upper", interpolation="bilinear", zorder=1)
        except Exception as e:                                # pragma: no cover
            get_logger().warning(f"sem hillshade: {type(e).__name__}: {e}")

    if cartography:
        # O REMA contém valores sobre partes do oceano onde existem tiles e
        # NaN onde não existem, criando retângulos cinza/azul artificiais. A
        # classe oceano do BedMachine é recolocada por cima do hillshade para
        # que toda a água tenha a mesma aparência, independente do tile REMA.
        ax.contourf(gx, gy, M["ice"], levels=[-0.5, 0.5],
                    colors=["#dfeaf4"], alpha=1.0, zorder=1.5)
        if shelf_fill:
            ax.contourf(gx, gy, M["floating"], levels=[0.5, 1.5],
                        colors=["#b9d6ec"], alpha=0.75, zorder=2)
        if coastline:
            ax.contour(gx, gy, M["ice"], levels=[0.5], colors="#1a1a1a",
                       linewidths=0.9, zorder=4)
        if grounding_line:
            ax.contour(gx, gy, M["grounded"], levels=[0.5], colors="#8b0000",
                       linewidths=0.8, linestyles="--", zorder=5)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    return ext


def draw_calving_fronts(ax, cfg: Config, epoch: float, tol: float = 0.6,
                        color: str = "#00509e", lw: float = 1.0):
    """
    Frentes de calving do IceLines mais próximas de `epoch`.

    Desenha apenas frentes dentro de `tol` anos da época pedida e devolve
    quantas foram usadas — a época NUNCA fica implícita, porque a frente migra.
    """
    import pandas as pd
    from shapely import wkt
    from shapely.geometry import MultiLineString

    p = cfg.paths.data_dir / "calving_fronts.parquet"
    if not p.exists():
        return 0
    d = pd.read_parquet(p)
    # por plataforma, a época publicada mais próxima da pedida
    keep = []
    for sh, g in d.groupby("shelf"):
        k = (g["epoch_year"] - epoch).abs().idxmin()
        if abs(g.loc[k, "epoch_year"] - epoch) <= tol:
            e = g.loc[k, "epoch_year"]
            keep.append(g[g["epoch_year"] == e])
    if not keep:
        return 0
    sel = pd.concat(keep)
    n = 0
    for w in sel["wkt"]:
        try:
            geom = wkt.loads(w)
        except Exception:
            continue
        parts = list(geom.geoms) if isinstance(geom, MultiLineString) else [geom]
        for ln in parts:
            xs, ys = ln.xy
            ax.plot(np.asarray(xs) / 1000, np.asarray(ys) / 1000,
                    color=color, lw=lw, zorder=6, solid_capstyle="round")
            n += 1
    return n


def add_scale_bar(ax, length_km: float = 100, loc=(0.06, 0.06), color="k"):
    """Barra de escala em coordenadas de eixo (dados em km)."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xs = x0 + loc[0] * (x1 - x0)
    ys = y0 + loc[1] * (y1 - y0)
    ax.plot([xs, xs + length_km], [ys, ys], color=color, lw=2.6,
            solid_capstyle="butt", zorder=10)
    ax.text(xs + length_km / 2, ys + 0.012 * (y1 - y0), f"{length_km:.0f} km",
            ha="center", va="bottom", fontsize=8, color=color, zorder=10)
