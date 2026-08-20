"""
thwaites.qc.grounded_mask
=========================
Máscara espacial para análise de **gelo aterrado** (grounded ice) e os campos de
distância que a sustentam.

Motivação
---------
Manter `keep_values = [1, 2, 3, 4]` do BedMachine, isto é, tudo menos oceano,
admite três contaminações numa análise de gelo continental apoiado no substrato:

* classe 1 = `ice_free_land` — rocha exposta. Não é gelo; sua elevação não muda
  por processos glaciológicos e seu dh/dt é ruído instrumental puro.
* classe 3 = `floating_ice` — plataformas. Respondem à maré e ao oceano, com
  amplitude de metros; misturar com gelo aterrado contamina o dh/dt.
* proximidade da linha de aterramento e da costa — mesmo dentro da classe 2, o
  gelo a poucos km da linha sofre flexão de maré, e o footprint do ATLAS
  (~11 m, mas com segmentos ATL06 de 40 m e geolocalização imperfeita) pode
  misturar sinal de superfícies vizinhas.

Campos calculados
-----------------
Tudo derivado do próprio BedMachine (mesma fonte da classificação, portanto sem
inconsistência entre produtos):

* `dist_to_nongrounded` — distância de cada pixel aterrado à borda mais próxima
  de qualquer coisa que não seja gelo aterrado. É o critério de buffer geral.
* `dist_to_floating` — distância à plataforma mais próxima; proxy direto da
  linha de aterramento (a fronteira grounded/floating É a grounding line na
  definição do BedMachine).
* `dist_to_ocean` — distância ao oceano aberto; controla contaminação costeira
  e por gelo marinho.

O gelo marinho não tem classe própria no BedMachine: ele ocupa a classe 0
(ocean). Logo, remover a classe 0 remove também o gelo marinho — mas apenas
enquanto a geolocalização estiver correta, o que motiva o buffer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from thwaites.config import Config
from thwaites.logging import get_logger

# Codificação oficial do BedMachine Antarctica v4 (atributo flag_meanings da
# variável 'mask'): 0 ocean, 1 ice_free_land, 2 grounded_ice, 3 floating_ice,
# 4 lake_vostok. Constantes nomeadas tornam explícita e auditável a intenção.
BM_OCEAN = 0
BM_ICE_FREE_LAND = 1
BM_GROUNDED_ICE = 2
BM_FLOATING_ICE = 3
BM_LAKE_VOSTOK = 4

BM_NAMES = {
    BM_OCEAN: "ocean",
    BM_ICE_FREE_LAND: "ice_free_land",
    BM_GROUNDED_ICE: "grounded_ice",
    BM_FLOATING_ICE: "floating_ice",
    BM_LAKE_VOSTOK: "lake_vostok",
}


def _resolve_bedmachine_nc(cfg: Config) -> Path:
    cands = sorted(cfg.paths.data_dir.glob("*BedMachineAntarctica*.nc"))
    if not cands:
        raise FileNotFoundError(
            "NetCDF do BedMachine não encontrado em data/. É a fonte da máscara "
            "e dos campos de distância (rode pipelines/fetch_bedmachine.py).")
    return cands[0]


def load_bedmachine_roi(cfg: Config, buffer_m: float = 60_000.0):
    """
    Lê a variável `mask` do BedMachine recortada à ROI (+buffer) e devolve
    (x, y, mask) com eixos crescentes.

    O buffer existe porque as distâncias são calculadas por transformada de
    distância: sem margem, a borda do recorte seria tratada como se fosse borda
    física, inventando "proximidade de costa" onde há apenas fim do recorte.
    """
    import xarray as xr
    from thwaites.grid.reproject import to_polar

    logger = get_logger()
    roi = cfg.roi or cfg.area
    clon = np.array([roi.lon_min, roi.lon_max, roi.lon_min, roi.lon_max])
    clat = np.array([roi.lat_min, roi.lat_min, roi.lat_max, roi.lat_max])
    cx, cy = to_polar(clon, clat, cfg)
    x0, x1 = float(cx.min()) - buffer_m, float(cx.max()) + buffer_m
    y0, y1 = float(cy.min()) - buffer_m, float(cy.max()) + buffer_m

    ds = xr.open_dataset(_resolve_bedmachine_nc(cfg))
    bx = np.asarray(ds["x"].values, dtype=np.float64)
    by = np.asarray(ds["y"].values, dtype=np.float64)
    ix = np.where((bx >= x0) & (bx <= x1))[0]
    iy = np.where((by >= y0) & (by <= y1))[0]
    if ix.size == 0 or iy.size == 0:
        ds.close()
        raise ValueError("a ROI não intersecta o BedMachine.")
    m = np.asarray(ds["mask"].isel(x=ix, y=iy).values)
    sx, sy = bx[ix], by[iy]
    ds.close()

    if sy[0] > sy[-1]:            # eixos crescentes para indexação previsível
        sy = sy[::-1]
        m = m[::-1, :]
    logger.info(f"BedMachine recortado: {m.shape} @ "
                f"{abs(sx[1]-sx[0]):.0f} m (+buffer {buffer_m/1000:.0f} km)")
    return sx, sy, m.astype(np.int8)


def distance_fields(mask: np.ndarray, pixel_m: float) -> dict:
    """
    Transformadas de distância euclidiana, em METROS, a partir da máscara.

    Devolve dict com:
      dist_to_nongrounded : distância ao não-aterrado mais próximo
      dist_to_floating    : distância à plataforma mais próxima (~grounding line)
      dist_to_ocean       : distância ao oceano mais próximo

    `distance_transform_edt` mede distância aos ZEROS do array de entrada, então
    cada campo passa a máscara booleana negada do alvo.
    """
    from scipy.ndimage import distance_transform_edt

    grounded = mask == BM_GROUNDED_ICE
    floating = mask == BM_FLOATING_ICE
    ocean = mask == BM_OCEAN

    out = {}
    out["dist_to_nongrounded"] = distance_transform_edt(grounded) * pixel_m
    # se não há alvo na janela, a distância é infinita — não zero, que seria
    # interpretado como "colado na feição"
    out["dist_to_floating"] = (distance_transform_edt(~floating) * pixel_m
                              if floating.any() else
                              np.full(mask.shape, np.inf, dtype=np.float64))
    out["dist_to_ocean"] = (distance_transform_edt(~ocean) * pixel_m
                           if ocean.any() else
                           np.full(mask.shape, np.inf, dtype=np.float64))
    return out


def sample_fields_at(x_pts, y_pts, sx, sy, fields: dict) -> dict:
    """
    Amostra campos de grade nas posições dos pontos por VIZINHO MAIS PRÓXIMO.

    Vizinho mais próximo, e não bilinear: os campos derivam de uma máscara
    categórica, e interpolar suaviza justamente a descontinuidade da borda que
    se quer detectar.
    """
    x_pts = np.asarray(x_pts, dtype=np.float64)
    y_pts = np.asarray(y_pts, dtype=np.float64)
    dx = sx[1] - sx[0]
    dy = sy[1] - sy[0]
    j = np.clip(np.rint((x_pts - sx[0]) / dx).astype(np.int64), 0, len(sx) - 1)
    i = np.clip(np.rint((y_pts - sy[0]) / dy).astype(np.int64), 0, len(sy) - 1)
    return {k: v[i, j] for k, v in fields.items()}


def grounded_keep_mask(mask_class, dist_nongrounded, dist_floating,
                       buffer_coast_m: float, buffer_gl_m: float) -> np.ndarray:
    """
    Critério de retenção para análise de gelo aterrado.

    Mantém o ponto se, e somente se:
      1. classe == grounded_ice (2), e
      2. distância a qualquer não-aterrado >= buffer_coast_m, e
      3. distância à plataforma        >= buffer_gl_m.

    Os dois buffers são separados de propósito: o de costa protege de mistura
    geométrica de sinal (borda de rocha, oceano) e o de linha de aterramento
    protege de um efeito FÍSICO distinto — a flexão de maré, que se propaga
    continente adentro por vários km e produz sinal periódico real, não erro.
    """
    keep = np.asarray(mask_class) == BM_GROUNDED_ICE
    keep &= np.asarray(dist_nongrounded) >= buffer_coast_m
    keep &= np.asarray(dist_floating) >= buffer_gl_m
    return keep


# ---------------------------------------------------------------------------
# Domínio FLUTUANTE (plataforma) — produto separado, nunca misturado
# ---------------------------------------------------------------------------
# Construído como máscara PRÓPRIA, não como negação da máscara aterrada. A
# diferença não é cosmética: a negação incluiria oceano aberto, mélange,
# icebergs e pinning points, e herdaria buffers escolhidos por um critério
# físico (flexão de maré vista do lado aterrado) que não é o relevante aqui.
#
# LIMITAÇÃO ESTRUTURAL (declarar em qualquer produto derivado):
# a máscara é ESTÁTICA. O BedMachine v4 tem `nominal_year = 2015` e
# `time_coverage` de 1970 a 2019-10 — não é "a geometria de 2019". A frente de
# calving recua e os limites de flutuação migram, então observações de 2024-25
# classificadas aqui como plataforma podem já ser oceano ou iceberg. Uma
# máscara contemporânea exige frentes datadas (IceLines/Sentinel-1) e ainda
# não está implementada; por isso os produtos gerados com esta função são
# EXPLORATÓRIOS e não devem ser rotulados como derretimento basal.

def shelf_keep_mask(mask_class, dist_open_water, dist_grounded,
                    buffer_grounding_zone_m: float,
                    buffer_front_m: float) -> np.ndarray:
    """
    Critério de retenção para o domínio de plataforma.

    Mantém o ponto se, e somente se:
      1. classe == floating_ice (3), e
      2. distância ao gelo ATERRADO >= buffer_grounding_zone_m — afasta da
         zona de aterramento, onde a flutuação não é plena e a hipótese
         hidrostática falha; e
      3. distância ao OCEANO >= buffer_front_m — afasta da frente de calving e
         das margens laterais. Mede distância ao oceano (classe 0), não a
         qualquer não-flutuante: incluir o gelo aterrado tornaria este critério
         redundante com o buffer da zona de aterramento.

    Os dois buffers são independentes dos usados no domínio aterrado e devem
    ser avaliados por sensibilidade própria (o documento de avaliação sugere
    2, 5 e 10 km), não herdados.
    """
    keep = np.asarray(mask_class) == BM_FLOATING_ICE
    keep &= np.asarray(dist_grounded) >= buffer_grounding_zone_m
    keep &= np.asarray(dist_open_water) >= buffer_front_m
    return keep


def distance_to_grounded(mask: np.ndarray, pixel_m: float) -> np.ndarray:
    """Distância de cada pixel ao gelo aterrado mais próximo, em metros."""
    from scipy.ndimage import distance_transform_edt
    grounded = mask == BM_GROUNDED_ICE
    if not grounded.any():
        return np.full(mask.shape, np.inf, dtype=np.float64)
    return distance_transform_edt(~grounded) * pixel_m


def distance_to_open_water(mask: np.ndarray, pixel_m: float) -> np.ndarray:
    """
    Distância ao OCEANO mais próximo — o critério certo para o buffer de frente
    de calving e margens laterais.

    Medir a distância a qualquer classe não flutuante incluiria gelo aterrado e
    tornaria o buffer de frente redundante com `buffer_grounding_zone_m`.
    A distância é, portanto, medida apenas ao oceano (classe 0), de modo que os dois
    buffers passam a controlar fenômenos distintos: um a flexão junto ao
    aterramento, outro a borda livre.
    """
    from scipy.ndimage import distance_transform_edt
    ocean = mask == BM_OCEAN
    if not ocean.any():
        # sem oceano na janela, nenhum ponto está perto da borda livre
        return np.full(mask.shape, np.inf, dtype=np.float64)
    return distance_transform_edt(~ocean) * pixel_m
