"""
thwaites.io.extract
===================
Extração das variáveis de interesse de um grânulo ATL06 (.h5).

Função pura e testável: recebe o caminho de um .h5 já baixado e a config,
devolve um DataFrame leve com:
  - variáveis nucleares (lon, lat, h_elv, s_elv, t_year, beam), já filtradas
    pelo controle de qualidade;
  - variáveis de correção geofísica (tide_ocean, dac, geoid) do grupo
    `land_ice_segments/geophysical/`, guardadas como colunas SEPARADAS
    (não aplicadas aqui — isso é papel de thwaites.corrections). Como o .h5
    bruto é deletado logo após, tudo que pode ser útil depois sai agora.

Não baixa nem deleta nada — isso é responsabilidade de thwaites.io.download.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import h5py

from thwaites.config import Config

_CORE_COLS = ("lon", "lat", "h_elv", "s_elv", "t_year", "beam")


def _beam_index_map(beams: list[str]) -> dict[str, int]:
    """gt1l->1, gt1r->2, ... (ordem da config)."""
    return {b: i + 1 for i, b in enumerate(beams)}


def _out_columns(cfg: Config) -> list[str]:
    return list(_CORE_COLS) + list(cfg.product.correction_vars.keys())


def extract_atl06(h5_path: str | Path, cfg: Config) -> pd.DataFrame:
    """
    Extrai e filtra os pontos válidos de um grânulo ATL06.

    Retorna DataFrame com colunas nucleares + colunas de correção
    (tide_ocean, dac, geoid). Correções vêm com NaN onde forem fill/inválidas.
    Pode ter 0 linhas se nenhum ponto passar no QC.
    """
    h5_path = Path(h5_path)
    p = cfg.product
    v = p.variables
    corr_vars = p.correction_vars           # {saida: nome_no_h5}
    beam_map = _beam_index_map(p.beams)
    fill_thr = cfg.corrections.fill_threshold

    out_cols = _out_columns(cfg)
    parts: dict[str, list[np.ndarray]] = {c: [] for c in out_cols}
    # variáveis de correção que nenhum feixe conseguiu localizar
    missing_vars: set[str] = set(corr_vars)

    with h5py.File(h5_path, "r") as f:
        for beam in p.beams:
            if beam not in f or p.segments_group not in f[beam]:
                continue
            seg = f[f"{beam}/{p.segments_group}"]

            if not all(name in seg for name in v.values()):
                continue

            lat  = seg[v["latitude"]][:]
            lon  = seg[v["longitude"]][:]
            h    = seg[v["height"]][:]
            s    = seg[v["sigma"]][:]
            dt   = seg[v["delta_t"]][:]
            qual = seg[v["quality"]][:]

            valid = (
                (qual <= cfg.qc.quality_max) &
                (h < cfg.qc.fill_value) &
                (h > cfg.qc.h_min_valid) &
                (~np.isnan(h)) &
                (~np.isnan(lat))
            )
            if not valid.any():
                continue

            t_year = p.atlas_epoch_year + (dt[valid] / p.seconds_per_year)

            parts["lon"].append(lon[valid])
            parts["lat"].append(lat[valid])
            parts["h_elv"].append(h[valid])
            parts["s_elv"].append(s[valid])
            parts["t_year"].append(t_year)
            parts["beam"].append(
                np.full(int(valid.sum()), beam_map[beam], dtype=np.int8)
            )

            # Correções geofísicas (grupo geophysical), alinhadas por segmento.
            n_valid = int(valid.sum())
            for out_name, h5name in corr_vars.items():
                # "grupo/variavel" busca em subgrupo explicito; nome simples
                # continua sendo procurado em `geophysical_group`.
                if "/" in h5name:
                    grp_name, var_name = h5name.split("/", 1)
                    grp = seg[grp_name] if grp_name in seg else None
                else:
                    grp_name, var_name = p.geophysical_group, h5name
                    grp = seg[grp_name] if grp_name in seg else None

                if grp is not None and var_name in grp:
                    arr = grp[var_name][:][valid].astype(np.float64)
                    # fill/inválido -> NaN (sinaliza "sem correção" ao passo seguinte)
                    arr = np.where(np.abs(arr) > fill_thr, np.nan, arr)
                    missing_vars.discard(out_name)
                else:
                    # NÃO silencioso: a variável ausente é registrada e o
                    # chamador avisa no fim. Preencher NaN sem rastro foi
                    # exatamente o que deixou a coluna `geoid` vazia em todo o
                    # dataset sem ninguém perceber.
                    arr = np.full(n_valid, np.nan)
                parts[out_name].append(arr)

    if missing_vars:
        import warnings
        warnings.warn(
            f"variáveis de correção NÃO encontradas no grânulo e preenchidas "
            f"com NaN: {sorted(missing_vars)}. Verifique o caminho declarado em "
            f"product.correction_vars (use 'grupo/variavel' se não estiver em "
            f"'{p.geophysical_group}').", RuntimeWarning, stacklevel=2)

    if not parts["lat"]:
        empty = {c: np.array([], dtype="float64") for c in out_cols}
        return pd.DataFrame(empty)

    df = pd.DataFrame({
        "lon":    np.concatenate(parts["lon"]).astype(np.float64),
        "lat":    np.concatenate(parts["lat"]).astype(np.float64),
        "h_elv":  np.concatenate(parts["h_elv"]).astype(np.float32),
        "s_elv":  np.concatenate(parts["s_elv"]).astype(np.float32),
        "t_year": np.concatenate(parts["t_year"]).astype(np.float64),
        "beam":   np.concatenate(parts["beam"]).astype(np.int8),
    })
    for out_name in corr_vars:
        df[out_name] = np.concatenate(parts[out_name]).astype(np.float32)
    return df
