"""
thwaites.corrections.datum
==========================
Harmonização de datum vertical: converte altura elipsoidal (WGS84) do ATL06 em
altura acima do geoide.

Por que existe
--------------
O ATL06 fornece `h_li` sobre o ELIPSOIDE. O geoide varia espacialmente — na ROI
do Amundsen, de −34,4 a −19,1 m. Isso é irrelevante para uma tendência num nó
FIXO (o offset é constante e cancela na derivada temporal), mas NÃO cancela ao
seguir uma parcela por dezenas de quilômetros: ali a altura é comparada em
posições diferentes, e o gradiente do geoide aparece como DH/Dt falso.

Fonte
-----
O BedMachine Antarctica v4 já traz a variável `geoid` (EIGEN-6C4, Förste et al.
2014), na mesma grade de 500 m usada pela máscara. Isso evita re-baixar os
grânulos ATL06 apenas para obter `dem/geoid_h`.

Magnitude medida nesta ROI (trajetórias de 2019,5 a 2025,5):

    deslocamento     |ΔN|        taxa espúria
    < 5 km          0,011 m      0,0019 m/ano
    5–10 km         0,065 m      0,0108 m/ano
    > 20 km         0,381 m      0,0636 m/ano

É uma correção pequena — não explica o RMSE elevado das trajetórias longas
(medido: ~1% do sinal). Aplicá-la mesmo assim é correto: é erro sistemático
conhecido, com correção disponível e custo desprezível.

Compatibilidade com produtos processados
----------------------------------------
Alguns Parquet disponíveis têm a coluna `geoid` inteiramente nula porque o ATL06
armazena o geoide em `land_ice_segments/dem/geoid_h`, fora do grupo
`land_ice_segments/geophysical/`. A configuração usa a sintaxe
`"grupo/variavel"`; este módulo também permite harmonizar produtos já processados
sem exigir novo download.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from thwaites.config import Config
from thwaites.logging import get_logger


class GeoidField:
    """Geoide EIGEN-6C4 do BedMachine, recortado à área pedida."""

    def __init__(self, cfg: Config, x_min, x_max, y_min, y_max,
                 buffer_m: float = 60_000.0):
        import xarray as xr

        logger = get_logger()
        cands = sorted(cfg.paths.data_dir.glob("*BedMachineAntarctica*.nc"))
        if not cands:
            raise FileNotFoundError(
                "BedMachine .nc não encontrado em data/ — é a fonte do geoide.")
        ds = xr.open_dataset(cands[0])
        if "geoid" not in ds:
            ds.close()
            raise KeyError(f"'geoid' ausente. Disponíveis: {list(ds.data_vars)}")

        bx = np.asarray(ds["x"].values, dtype=np.float64)
        by = np.asarray(ds["y"].values, dtype=np.float64)
        ix = np.where((bx >= x_min - buffer_m) & (bx <= x_max + buffer_m))[0]
        iy = np.where((by >= y_min - buffer_m) & (by <= y_max + buffer_m))[0]
        if ix.size == 0 or iy.size == 0:
            ds.close()
            raise ValueError("a área pedida não intersecta o BedMachine.")

        self.N = np.asarray(ds["geoid"].isel(x=ix, y=iy).values, dtype=np.float64)
        self.x = bx[ix]
        self.y = by[iy]
        self.attrs = dict(ds["geoid"].attrs)
        ds.close()

        if self.y[0] > self.y[-1]:
            self.y = self.y[::-1]
            self.N = self.N[::-1, :]
        self.dx = float(self.x[1] - self.x[0])
        self.dy = float(self.y[1] - self.y[0])
        logger.info(f"geoide (BedMachine/EIGEN-6C4): {self.N.shape} @ "
                    f"{abs(self.dx):.0f} m | {np.nanmin(self.N):.1f} a "
                    f"{np.nanmax(self.N):.1f} m")

    def at(self, px, py):
        """
        Altura do geoide (m) nas posições dadas, por interpolação BILINEAR.

        Bilinear e não vizinho mais próximo: o geoide é um campo contínuo e
        suave, e ao longo de uma trajetória o que interessa é justamente a
        VARIAÇÃO — degraus de 500 m introduziriam saltos artificiais na série
        da parcela.
        """
        px = np.asarray(px, dtype=np.float64)
        py = np.asarray(py, dtype=np.float64)
        fx = (px - self.x[0]) / self.dx
        fy = (py - self.y[0]) / self.dy
        j0 = np.floor(fx).astype(np.int64)
        i0 = np.floor(fy).astype(np.int64)
        ok = ((j0 >= 0) & (j0 < len(self.x) - 1) &
              (i0 >= 0) & (i0 < len(self.y) - 1))
        out = np.full(px.shape, np.nan)
        if not ok.any():
            return out
        j, i = j0[ok], i0[ok]
        wx, wy = fx[ok] - j, fy[ok] - i
        out[ok] = ((1 - wx) * (1 - wy) * self.N[i, j] +
                   wx * (1 - wy) * self.N[i, j + 1] +
                   (1 - wx) * wy * self.N[i + 1, j] +
                   wx * wy * self.N[i + 1, j + 1])
        return out


def to_orthometric(h_ellipsoid, geoid_n):
    """
    Altura ortométrica: h* = h_elipsoidal − N.

    Convenção verificada no atributo do produto: `geoid` do BedMachine é
    "EIGEN-6C4 Geoid − WGS84 Ellipsoid difference", ou seja, N é a altura do
    geoide ACIMA do elipsoide — logo se SUBTRAI de h para obter altura sobre o
    geoide. Errar este sinal dobraria o erro em vez de removê-lo.
    """
    return np.asarray(h_ellipsoid, dtype=float) - np.asarray(geoid_n, dtype=float)


def geoid_gradient_error(geoid: GeoidField, x0, y0, x1, y1, dt_years: float):
    """
    Taxa espúria de dh/dt que o gradiente do geoide produziria entre duas
    posições, se o datum NÃO fosse harmonizado.

    Serve para quantificar o erro evitado — número que deve ir para o
    orçamento de incerteza, não ficar implícito.
    """
    dN = geoid.at(x1, y1) - geoid.at(x0, y0)
    return dN / float(dt_years)
