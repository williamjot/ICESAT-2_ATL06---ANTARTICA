"""
thwaites.corrections.gia
========================
Movimento vertical do embasamento (VLM) — ajuste isostático glacial.

Por que isto é obrigatório
--------------------------
O altímetro mede a superfície num referencial fixo à Terra, não a espessura do
gelo:

    dh/dt medido = dh/dt do gelo + dB/dt do embasamento

Sem subtrair dB/dt, soerguimento positivo faz a superfície subir e o dh/dt
medido fica MENOS negativo que o adelgaçamento real — subestima-se a perda.
É viés ASSINADO: aumentar a amostra não o reduz.

No Amundsen Sea Embayment isso não é detalhe. Barletta et al. (2018, Science,
doi:10.1126/science.aao1447) mediram +41 mm/ano de soerguimento por GPS e
viscosidade de manto de 4e18 Pa·s — resposta GIA em DÉCADAS, não milênios.

Fonte usada
-----------
Caron et al. (2018), GRL 45, doi:10.1002/2017GL076644 — estatística bayesiana
sobre ~1e5 modelos GIA, distribuída em https://vesl.jpl.nasa.gov/solid-earth/gia/
como tabela ASCII com colatitude, longitude, e expectativa e desvio-padrão de
VLM, taxa de geoide e taxa de gravidade.

Escolhida pelo DESVIO-PADRÃO: é a única das opções correntes que traz incerteza
formal por posição, o que permite propagar o erro do GIA para as Gt/ano em vez
de aplicar um escalar sem σ.

LIMITAÇÃO QUE NÃO PODE SER OMITIDA
----------------------------------
O Caron é harmônico esférico truncado no grau 89, isto é, resolução de 1°
(~28 km em longitude e ~111 km em latitude a 75°S), e é vinculado globalmente
por dados de nível relativo do mar. Ele NÃO resolve a resposta local de baixa
viscosidade que Barletta mediu no ASE. A expectativa que ele fornece para a
nossa região é, quase certamente, um LIMITE INFERIOR do soerguimento real —
o que significa que a correção aplicada aqui é, ela própria, conservadora, e a
perda corrigida deve ser lida como limite inferior da perda verdadeira.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from thwaites.logging import get_logger

MM_TO_M = 1e-3


@dataclass
class GIAField:
    """VLM (m/ano) e seu desvio-padrão, interpolados da grade de 1° do Caron."""

    colat: np.ndarray          # graus, crescente
    lon: np.ndarray            # graus em [0, 360), crescente
    vlm: np.ndarray            # (n_colat, n_lon), m/ano
    vlm_sigma: np.ndarray      # (n_colat, n_lon), m/ano
    source: str

    @classmethod
    def from_caron_table(cls, path: str | Path) -> "GIAField":
        """
        Lê `GIA_maps_Caron_et_al_2018` (ASCII, linhas de comentário com '%').

        Colunas: colatitude, longitude, VLM exp, VLM std, geoide exp, geoide
        std, gravidade exp, gravidade std. Só as duas de VLM são usadas — as de
        geoide e gravidade servem a GRACE, não a altimetria.
        """
        path = Path(path)
        raw = np.loadtxt(path, comments="%")
        if raw.shape[1] < 4:
            raise ValueError(
                f"{path.name}: esperadas >=4 colunas, achadas {raw.shape[1]}")

        colat = np.unique(raw[:, 0])
        lon = np.unique(raw[:, 1])
        if colat.size * lon.size != raw.shape[0]:
            raise ValueError(
                f"{path.name}: {raw.shape[0]} linhas não formam grade "
                f"{colat.size}x{lon.size} — o arquivo não é regular.")

        ic = np.searchsorted(colat, raw[:, 0])
        il = np.searchsorted(lon, raw[:, 1])
        vlm = np.full((colat.size, lon.size), np.nan)
        sig = np.full((colat.size, lon.size), np.nan)
        vlm[ic, il] = raw[:, 2] * MM_TO_M
        sig[ic, il] = raw[:, 3] * MM_TO_M

        get_logger().info(
            f"GIA Caron et al. (2018): grade {colat.size}x{lon.size} "
            f"(colat {colat.min():.0f}-{colat.max():.0f}°, passo "
            f"{np.diff(colat).min():.2f}°) | VLM {1e3*np.nanmin(vlm):+.2f} a "
            f"{1e3*np.nanmax(vlm):+.2f} mm/ano")
        return cls(colat=colat, lon=lon, vlm=vlm, vlm_sigma=sig,
                   source=f"Caron et al. (2018), {path.name}")

    # ------------------------------------------------------------------ uso
    def sample(self, lon_deg: np.ndarray, lat_deg: np.ndarray) -> tuple:
        """
        VLM e σ (m/ano) nas posições dadas, por interpolação BILINEAR.

        Bilinear e não vizinho-mais-próximo porque o campo é suave em 1° e a
        nossa grade tem 5 km: o vizinho mais próximo criaria degraus de ~1° na
        correção, que apareceriam como artefato retangular no mapa de dh/dt
        corrigido e seriam confundidos com estrutura glaciológica.
        """
        lon_deg = np.asarray(lon_deg, float) % 360.0
        colat_q = 90.0 - np.asarray(lat_deg, float)

        def _interp(F):
            # índices e pesos em colatitude (sem envolvimento nos polos)
            ci = np.clip(np.searchsorted(self.colat, colat_q) - 1,
                         0, self.colat.size - 2)
            c0, c1 = self.colat[ci], self.colat[ci + 1]
            wc = np.clip((colat_q - c0) / (c1 - c0), 0.0, 1.0)

            # longitude é CÍCLICA: o setor entre o último e o primeiro nó
            # (359°->0°) tem de fechar, ou toda a faixa cairia no valor de borda
            li = np.searchsorted(self.lon, lon_deg) - 1
            li = np.where(li < 0, self.lon.size - 1, li)
            lj = (li + 1) % self.lon.size
            l0 = self.lon[li]
            step = (self.lon[lj] - l0) % 360.0
            step = np.where(step == 0, 360.0, step)
            wl = np.clip(((lon_deg - l0) % 360.0) / step, 0.0, 1.0)

            return ((1 - wc) * (1 - wl) * F[ci, li]
                    + (1 - wc) * wl * F[ci, lj]
                    + wc * (1 - wl) * F[ci + 1, li]
                    + wc * wl * F[ci + 1, lj])

        return _interp(self.vlm), _interp(self.vlm_sigma)


def correct_elevation_rate(dhdt: np.ndarray, vlm: np.ndarray) -> np.ndarray:
    """
    Remove o movimento do embasamento da taxa de elevação medida.

        dh/dt_gelo = dh/dt_medido − dB/dt

    Sinal: `vlm` POSITIVO é soerguimento. Subtrair torna o resultado MAIS
    negativo, isto é, aumenta a perda estimada — que é o efeito esperado, já
    que o soerguimento estava mascarando parte do adelgaçamento.
    """
    return np.asarray(dhdt, float) - np.asarray(vlm, float)


def systematic_mass_uncertainty(sigma_vlm: np.ndarray, area_m2: float,
                                ice_density: float) -> float:
    """
    Incerteza de massa (Gt/ano) devida ao erro do modelo GIA.

    TRATADA COMO TOTALMENTE CORRELACIONADA sobre a região, de propósito. O σ do
    Caron vem da dispersão de um ensemble de modelos que diferem em reologia e
    história de degelo — parâmetros GLOBAIS. Dois pontos vizinhos não erram de
    forma independente: erram juntos, porque é o mesmo modelo errado nos dois.

    Dividir por sqrt(N) aqui, como se cada célula trouxesse informação nova,
    reduziria essa incerteza a quase zero e seria um erro grosseiro. Por isso a
    média de σ multiplica a área INTEIRA, sem atenuação.
    """
    s = np.asarray(sigma_vlm, float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return float("nan")
    return float(np.mean(s) * area_m2 * ice_density / 1e12)
