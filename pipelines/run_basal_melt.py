"""
pipelines/run_basal_melt.py
===========================
Derretimento basal ṁ_b por parcela, na formulação LAGRANGIANA.

    data/dhdt/shelf_windows_divergence.parquet + SMB (GSFC-FDM)
        -> data/dhdt/shelf_basal_melt.parquet
        -> outputs/tables/basal_melt_report.json

Formulação
----------
Segue a estrutura de Adusumilli et al. (2018, GRL, 10.1002/2017GL076652), que
para um referencial EULERIANO escrevem

    dh/dt = (ρw−ρi)/ρw · [ Ms/ρi − ∇·(v̄H) − w_b ] + dh_s/dt

Aqui o dh/dt é LAGRANGIANO (parcela seguida no tempo), então o termo de
transporte é H·∇·v e NÃO a divergência completa ∇·(H·v): como
∇·(H·v) = v·∇H + H·∇·v, usar a forma completa junto de DH/Dt contaria a
advecção duas vezes — uma no seguimento da parcela, outra em v·∇H.

Resolvendo para o derretimento basal:

    ṁ_b = a_s − DH/Dt − H·∇·v

com DH/Dt = amplificação hidrostática aplicada ao DH_freeboard/Dt medido, e
a_s o SMB em metros de gelo equivalente por ano.

Sinal: ṁ_b POSITIVO = derretimento (perda de massa na base).

O que NÃO está incluído
-----------------------
Adusumilli separam ainda `dh_s/dt`, a variação de altura por mudança de
densidade na coluna de firn (modelo IMAU-FDM). Aqui usa-se o GSFC-FDM, e a
taxa de FAC é aplicada ao dh/dt antes da amplificação. A cobertura do FDM
termina em jun/2022 enquanto o período vai a 2025 — extrapolação declarada.

Uso: python pipelines/run_basal_melt.py [--smooth-km 3]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config
from thwaites.logging import setup_logging


def main():
    ap = argparse.ArgumentParser(description="Derretimento basal lagrangiano.")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--input", default="shelf_windows_divergence.parquet")
    ap.add_argument("--reliable-only", action="store_true", default=True)
    args = ap.parse_args()

    cfg = load_config(args.profile)
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level,
                        run_name="basal_melt")

    from netCDF4 import Dataset

    src = cfg.paths.dhdt_dir / args.input
    if not src.exists():
        raise FileNotFoundError(f"{src} (rode run_shelf_divergence.py).")
    d = pd.read_parquet(src)
    log.info(f"{len(d):,} registros de parcela")

    rho_i = cfg.mass_balance.ice_density
    rho_w = cfg.flux.water_density
    amp = rho_w / (rho_w - rho_i)

    # ---- SMB (a_s), em m de gelo equivalente por ano -----------------------
    smb_p = cfg.paths.data_dir / "smb_thwaites.nc"
    if not smb_p.exists():
        raise FileNotFoundError(f"{smb_p} (rode fetch_firn.py --which smb).")
    with Dataset(smb_p) as f:
        sx = np.asarray(f["x"][:], float).ravel()
        sy = np.asarray(f["y"][:], float).ravel()
        # SMB do GSFC-FDM já vem como taxa em m de gelo equivalente por ano;
        # média temporal sobre o período disponível
        S = np.nanmean(np.asarray(f["SMB"][:], float), axis=0)
        n_t = f["SMB"].shape[0]
    if sy[0] > sy[-1]:
        sy, S = sy[::-1], S[::-1, :]

    def samp(F, ax, ay, px, py):
        j = np.clip(np.rint((px - ax[0]) / (ax[1] - ax[0])).astype(np.int64),
                    0, len(ax) - 1)
        i = np.clip(np.rint((py - ay[0]) / (ay[1] - ay[0])).astype(np.int64),
                    0, len(ay) - 1)
        return F[i, j]

    px = d["x_ref"].to_numpy()
    py = d["y_ref"].to_numpy()
    d["smb"] = samp(S, sx, sy, px, py)
    log.info(f"SMB: {n_t} épocas | mediana {np.nanmedian(d['smb']):+.3f} m gelo/ano")

    # ---- FAC: separa mudança de altura por densificação do firn ------------
    fac_rate = np.zeros(len(d))
    fp = cfg.paths.data_dir / cfg.firn.path
    if fp.exists():
        with Dataset(fp) as f:
            fx = np.asarray(f["x"][:], float).ravel()
            fy = np.asarray(f["y"][:], float).ravel()
            t = np.asarray(f["time"][:], float)
            t_units = str(getattr(f["time"], "units", "")).lower()
            FAC = np.asarray(f["FAC"][:], float)
        if fy[0] > fy[-1]:
            fy, FAC = fy[::-1], FAC[:, ::-1, :]

        # Converte o eixo temporal para ANOS lendo as UNIDADES declaradas.
        if "year" in t_units:
            t_years = t
        elif "day" in t_units:
            t_years = t / 365.25
        else:
            raise ValueError(
                f"unidade de tempo do FAC não reconhecida: {t_units!r}. "
                f"Declare explicitamente em vez de adivinhar pela magnitude.")
        tt = t_years - t_years.mean()
        denom = float(np.sum(tt ** 2))
        RATE = (np.tensordot(tt, FAC - FAC.mean(axis=0), axes=(0, 0)) / denom
                if denom > 0 else np.zeros(FAC.shape[1:]))
        fac_rate = samp(RATE, fx, fy, px, py)
        log.info(f"dFAC/dt mediano: {np.nanmedian(fac_rate):+.4f} m/ano "
                 f"(cobertura do FDM termina em jun/2022 — EXTRAPOLADO depois)")
    else:
        log.warning("FAC indisponível — dh_s/dt NÃO separado do sinal.")

    # ---- balanço -----------------------------------------------------------
    # altura de gelo (removida a parte de firn) -> espessura por hidrostática
    dh_ice = d["dhdt_lagrangian"].to_numpy() - fac_rate
    d["dhdt_ice"] = dh_ice
    d["DHDt"] = amp * dh_ice                       # ∂H/∂t lagrangiano
    d["basal_melt"] = d["smb"] - d["DHDt"] - d["H_divv"]
    # Limite inferior estatístico: propaga apenas o erro do ajuste de DH/Dt.
    # Não rotular como incerteza total; faltam covariâncias/erros de SMB, FAC,
    # maré, trajetória, velocidade, divergência e espessura.
    if "dhdt_sigma_stat" in d.columns:
        d["basal_melt_sigma_stat_lower_bound"] = (
            amp * d["dhdt_sigma_stat"].to_numpy(dtype=float))

    sel = np.isfinite(d["basal_melt"])
    if args.reliable_only and "reliable" in d.columns:
        sel &= d["reliable"].to_numpy(dtype=bool)
    g = d[sel]
    v = g["basal_melt"].to_numpy()

    dst = cfg.paths.dhdt_dir / "shelf_basal_melt.parquet"
    d.to_parquet(dst, index=False)

    per_win = {}
    for (a, b), gw in g.groupby(["window_start", "window_end"]):
        per_win[f"{a}-{b}"] = {
            "n": int(len(gw)),
            "basal_melt_mediana": float(gw["basal_melt"].median()),
            "DHDt_mediana": float(gw["DHDt"].median()),
            "H_divv_mediana": float(gw["H_divv"].median()),
            "smb_mediana": float(gw["smb"].median()),
        }

    def summarize_group(group):
        values = group["basal_melt"].to_numpy(dtype=float)
        return {
            "n": int(len(group)),
            "basal_melt_mediana_m_ano": float(np.median(values)),
            "basal_melt_media_m_ano": float(np.mean(values)),
            "basal_melt_p10_m_ano": float(np.percentile(values, 10)),
            "basal_melt_p90_m_ano": float(np.percentile(values, 90)),
            "DHDt_mediana_m_ano": float(group["DHDt"].median()),
            "H_divv_mediana_m_ano": float(group["H_divv"].median()),
            "smb_mediana_m_ano": float(group["smb"].median()),
        }

    per_shelf = {}
    per_shelf_window = {}
    if "shelf" in g.columns:
        for shelf, group in g.groupby("shelf", dropna=False):
            shelf_name = str(shelf)
            per_shelf[shelf_name] = summarize_group(group)
            per_shelf_window[shelf_name] = {
                f"{a}-{b}": summarize_group(window)
                for (a, b), window in group.groupby(["window_start", "window_end"])
            }

    rep = {
        "formulacao": ("m_b = a_s - DH/Dt - H*div(v)  [LAGRANGIANO]. Estrutura "
                       "de Adusumilli et al. (2018, GRL); o termo de transporte "
                       "e H*div(v), NAO div(H*v), porque DH/Dt ja segue a "
                       "parcela — usar a forma completa contaria a adveccao "
                       "duas vezes."),
        "sinal": "m_b POSITIVO = derretimento basal (perda de massa na base)",
        "n_parcelas": int(len(g)),
        "usou_apenas_confiaveis": bool(args.reliable_only),
        "hydrostatic_amplification": float(amp),
        "basal_melt_mediana_m_ano": float(np.median(v)),
        "basal_melt_media_m_ano": float(np.mean(v)),
        "basal_melt_p10": float(np.percentile(v, 10)),
        "basal_melt_p90": float(np.percentile(v, 90)),
        "sigma_stat_lower_bound_m_yr_median": (
            float(g["basal_melt_sigma_stat_lower_bound"].median())
            if "basal_melt_sigma_stat_lower_bound" in g.columns else None),
        "uncertainty_status": (
            "LIMITE_INFERIOR: inclui apenas sigma estatistico de DH/Dt; nao "
            "inclui SMB, FAC, mare, trajetoria, velocidade, divergencia ou H"),
        "termos_medianos": {
            "a_s (SMB)": float(g["smb"].median()),
            "DH/Dt": float(g["DHDt"].median()),
            "H*div(v)": float(g["H_divv"].median()),
        },
        "por_janela": per_win,
        "por_plataforma": per_shelf,
        "por_plataforma_e_janela": per_shelf_window,
        "limitacoes": [
            "H do BedMachine tem epoca nominal 2015; periodo 2019-2025",
            "FAC do GSFC-FDM termina jun/2022 — extrapolado ate 2025",
            "H*div(v) e sensivel a suavizacao: muda de sinal entre 5 e 8 km "
            "(ver shelf_divergence_report.json)",
            "frentes de gelo datadas sao aplicadas por epoca; a linha de "
            "aterramento e a geometria basal continuam dependentes dos produtos "
            "estaticos usados na mascara e no BedMachine",
            "residuo de mare ~+/-17 cm amplificado ~9,3x = ~1,6 m por observacao",
            "buffer de 3 km na zona de aterramento SUBESTIMA m_b, pois e ali "
            "que o derretimento e maior (ressalva do proprio Adusumilli)",
        ],
        "referencia": ("Adusumilli, Fricker, Siegfried, Padman, Paolo, "
                       "Ligtenberg (2018), GRL 45, 4086-4095, "
                       "doi:10.1002/2017GL076652"),
    }
    rp = cfg.paths.tables / "basal_melt_report.json"
    rp.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info(f"Derretimento basal -> {dst} ({len(g):,} parcelas)")
    log.info(f"  termos medianos: a_s {g['smb'].median():+.3f} | "
             f"DH/Dt {g['DHDt'].median():+.3f} | "
             f"H*div(v) {g['H_divv'].median():+.3f} m/ano")
    log.info(f"  m_b mediano: {np.median(v):+.3f} m/ano "
             f"(p10 {np.percentile(v,10):+.2f}, p90 {np.percentile(v,90):+.2f})")
    for k, val in per_win.items():
        log.info(f"    {k}: m_b {val['basal_melt_mediana']:+.3f} m/ano "
                 f"(n={val['n']:,})")
    log.info(f"Relatório -> {rp}")


if __name__ == "__main__":
    main()
