"""
thwaites.uncertainty.geolocation
================================
Erro de altura induzido pela geolocalização horizontal sobre superfície
inclinada — o termo que o ATBD ATL14/ATL15 r005 (§3.4.4) chama de fonte
DOMINANTE de erro correlacionado, e que este projeto não tinha.

O problema
----------
O peso do ajuste de dh/dt usava só `h_li_sigma` (coluna `s_elv`), cuja mediana
nos tiles é 0,034 m. Mas `h_li_sigma` é o erro de AJUSTE do segmento (ruído de
fóton e correção de first-photon bias) — não inclui o erro de apontamento. E o
erro de apontamento, projetado na declividade, é maior: o piso vertical puro já
é 0,133 m, quatro vezes o `s_elv` mediano.

Pior que a magnitude é a ESTRUTURA. O ATBD:

    "We expect these geolocation errors to be consistent over time scales of
    several seconds or more, equivalent to spatial scales of tens of km or
    more."

Isto é, o termo dominante é ESPACIALMENTE CORRELACIONADO em dezenas de km,
enquanto o ruído de ajuste é praticamente branco. Tratar tudo como um único
componente com um único comprimento de correlação — como fazíamos — é o que
deixava a barra de erro de massa indeterminada por um fator 4,5 (σ entre 3,25 e
14,74 Gt/ano conforme L fosse 34,1 ou 154,4 km). Separar os dois componentes
substitui essa escolha arbitrária por uma atribuição física.

Modelo calibrado, não suposto
-----------------------------
`sigma_geo_h` do ATL06 é o erro vertical total de geolocalização, e está em
disco para os 1.106 grânulos (`data/qc_flags/`), junto de `dh_fit_dx` (a
declividade ao longo da trilha). Isso permite CALIBRAR

    σ_geo²(s) = σ_v0² + (σ_horiz · s)²

contra o próprio produto, em vez de adotar um valor de literatura. O ajuste
robusto por medianas em bins de declividade dá σ_v0 = 0,133 m e
σ_horiz = 6,19 m (R² = 0,998) — e os 6,19 m concordam com o requisito de
conhecimento de geolocalização horizontal do ICESat-2, de 6,5 m, o que é uma
verificação independente de que a calibração está medindo o que se pretende.

Propagação para dh/dt
---------------------
Um viés de altura CONSTANTE num nó não afeta dh/dt — some no intercepto. O que
afeta é o viés que MUDA entre passagens: cada sobrevoo tem seu próprio erro de
apontamento, aproximadamente constante ao longo dos segundos em que cruza o nó.

Modelando o deslocamento da passagem k como δ_k independente entre passagens e
perfeitamente correlacionado dentro dela, o ajuste de mínimos quadrados de h
contra t dá

    Var(dh/dt) = σ_geo² / Σ_k (t_k − t̄)²

com k percorrendo PASSAGENS, não observações. É essa contagem — dezenas de
passagens, não centenas de milhares de segmentos — que define a amostra efetiva
deste termo, e é por isso que o erro formal atual o subestima tanto.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from thwaites.logging import get_logger

# Requisito de conhecimento de geolocalização horizontal do ICESat-2, usado só
# como VERIFICAÇÃO da calibração — nunca como valor adotado.
ICESAT2_GEOLOC_REQUIREMENT_M = 6.5


@dataclass
class GeoErrorModel:
    """σ_geo²(s) = σ_v0² + (σ_horiz·s)², calibrado contra o próprio ATL06."""

    sigma_v0_m: float
    sigma_horiz_m: float
    r2: float
    n_granules: int
    n_segments: int

    def sigma_geo(self, slope_mag) -> np.ndarray:
        """Erro vertical de geolocalização (m) para declividade `slope_mag`."""
        s = np.abs(np.asarray(slope_mag, float))
        return np.sqrt(self.sigma_v0_m ** 2 + (self.sigma_horiz_m * s) ** 2)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["requisito_icesat2_m"] = ICESAT2_GEOLOC_REQUIREMENT_M
        d["concordancia_com_requisito"] = (
            f"{self.sigma_horiz_m:.2f} m calibrado vs "
            f"{ICESAT2_GEOLOC_REQUIREMENT_M} m de requisito")
        return d


def calibrate_from_qc(qc_dir: str | Path, n_granules: int = 40,
                      n_bins: int = 24, min_per_bin: int = 500,
                      quality_col: str = "atl06_quality_summary") -> GeoErrorModel:
    """
    Calibra σ_v0 e σ_horiz nos arquivos de flags de QC já em disco.

    Usa MEDIANAS por bin de declividade, e não regressão direta sobre todos os
    segmentos, porque `sigma_geo_h` tem cauda longa (máximo de ~43 m contra
    mediana de 0,14 m): a regressão global é dominada pelos outliers e devolve
    σ_v0 = 1,24 m, um valor quase dez vezes o real. O ajuste por medianas dá
    0,133 m com R² de 0,998 — a diferença entre os dois é inteiramente
    robustez, não escolha de modelo.
    """
    import pandas as pd

    qc_dir = Path(qc_dir)
    files = sorted(qc_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"sem arquivos de flags em {qc_dir}")
    step = max(len(files) // n_granules, 1)
    sel = files[::step][:n_granules]

    S, D = [], []
    for f in sel:
        d = pd.read_parquet(f, columns=["sigma_geo_h", "dh_fit_dx", quality_col])
        keep = d[quality_col].to_numpy() == 0
        S.append(d["sigma_geo_h"].to_numpy(float)[keep])
        D.append(np.abs(d["dh_fit_dx"].to_numpy(float))[keep])
    s = np.concatenate(S)
    g = np.concatenate(D)
    ok = np.isfinite(s) & np.isfinite(g) & (s > 0)
    s, g = s[ok], g[ok]
    if s.size < 10_000:
        raise ValueError(f"amostra insuficiente para calibrar: {s.size} segmentos")

    edges = np.percentile(g, np.linspace(0.0, 100.0, n_bins + 1))
    xs, ys = [], []
    for i in range(n_bins):
        m = (g >= edges[i]) & (g < edges[i + 1])
        if m.sum() < min_per_bin:
            continue
        xs.append(np.median(g[m]) ** 2)
        ys.append(np.median(s[m]) ** 2)
    xs, ys = np.asarray(xs), np.asarray(ys)
    if xs.size < 4:
        raise ValueError(f"apenas {xs.size} bins úteis — calibração não confiável")

    A = np.column_stack([np.ones(xs.size), xs])
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    r2 = 1.0 - np.sum((ys - A @ coef) ** 2) / np.sum((ys - ys.mean()) ** 2)

    model = GeoErrorModel(
        sigma_v0_m=float(np.sqrt(max(coef[0], 0.0))),
        sigma_horiz_m=float(np.sqrt(max(coef[1], 0.0))),
        r2=float(r2), n_granules=len(sel), n_segments=int(s.size))

    log = get_logger()
    log.info(f"erro de geolocalização calibrado em {len(sel)} grânulos "
             f"({s.size:,} segmentos): σ_v0 = {model.sigma_v0_m:.4f} m, "
             f"σ_horiz = {model.sigma_horiz_m:.2f} m (R² = {r2:.4f})")
    ratio = model.sigma_horiz_m / ICESAT2_GEOLOC_REQUIREMENT_M
    if not 0.5 <= ratio <= 1.5:
        log.warning(f"σ_horiz calibrado ({model.sigma_horiz_m:.2f} m) destoa do "
                    f"requisito do ICESat-2 ({ICESAT2_GEOLOC_REQUIREMENT_M} m) "
                    f"por fator {ratio:.2f} — verificar antes de usar.")
    return model


def overpass_leverage(t: np.ndarray, tol_years: float = 1.0 / 365.25) -> tuple:
    """
    Agrupa épocas em PASSAGENS e devolve (n_passagens, Σ(t_k − t̄)²).

    Duas observações separadas por menos de `tol_years` pertencem à mesma
    passagem e compartilham o mesmo erro de apontamento — contá-las como
    independentes é exatamente o que infla artificialmente a amostra efetiva.
    O padrão de um dia é folgado o bastante para juntar as seis trilhas de um
    mesmo sobrevoo e estreito o bastante para separar ciclos de 91 dias.

    A `Σ(t_k − t̄)²` é a alavancagem temporal: é ela, e não o número de
    segmentos, que determina quanto um erro por passagem contamina a tendência.
    """
    t = np.asarray(t, float)
    t = np.sort(t[np.isfinite(t)])
    if t.size == 0:
        return 0, 0.0
    # início de nova passagem sempre que o salto excede a tolerância
    new = np.concatenate([[True], np.diff(t) > tol_years])
    gid = np.cumsum(new) - 1
    n_pass = int(gid[-1]) + 1
    centers = np.bincount(gid, weights=t) / np.bincount(gid)
    lev = float(np.sum((centers - centers.mean()) ** 2))
    return n_pass, lev


def sigma_dhdt_from_geolocation(sigma_geo, leverage: float) -> np.ndarray:
    """
    Erro em dh/dt (m/ano) devido ao erro de apontamento por passagem.

        σ_dhdt = σ_geo / sqrt(Σ_k (t_k − t̄)²)

    Alavancagem nula (uma só passagem, ou todas simultâneas) devolve NaN, e não
    zero: sem alavancagem temporal a tendência não é estimável, e devolver zero
    afirmaria certeza onde não há informação alguma.
    """
    sg = np.asarray(sigma_geo, float)
    if not np.isfinite(leverage) or leverage <= 0:
        return np.full(sg.shape, np.nan)
    return sg / np.sqrt(leverage)


def slope_magnitude_grid(rema_path, x0, x1, y0, y1, decimate: int = 1):
    """
    |∇h| do REMA na janela pedida. Devolve (x_centros, y_centros, |∇h|).

    A declividade é calculada na resolução NATIVA por padrão (32 m). Decimar
    suavizaria o relevo e subestimaria a declividade — e subestimar a
    declividade subestima σ_geo, o que empurraria a incerteza para baixo. O erro
    seria, portanto, na direção anti-conservadora.
    """
    import rasterio
    from rasterio.windows import from_bounds, Window

    with rasterio.open(rema_path) as src:
        win = from_bounds(x0, y0, x1, y1, transform=src.transform)
        win = win.intersection(Window(0, 0, src.width, src.height))
        h = max(int(win.height) // decimate, 1)
        w = max(int(win.width) // decimate, 1)
        z = src.read(1, window=win, out_shape=(h, w),
                     resampling=rasterio.enums.Resampling.average).astype(np.float32)
        nod = src.nodata
        bx0, by0, bx1, by1 = rasterio.windows.bounds(win, src.transform)
    if nod is not None:
        z[z == nod] = np.nan
    z[z < -100] = np.nan

    dx = (bx1 - bx0) / w
    dy = (by1 - by0) / h
    gy, gx = np.gradient(z, dy, dx)
    mag = np.hypot(gx, gy)

    xs = bx0 + (np.arange(w) + 0.5) * dx
    ys = by1 - (np.arange(h) + 0.5) * dy      # raster do topo para baixo
    return xs, ys, mag
