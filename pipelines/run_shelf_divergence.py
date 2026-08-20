"""
pipelines/run_shelf_divergence.py
=================================
Termo H·∇·v nas parcelas da plataforma — o segundo termo do balanço Lagrangiano.

    data/dhdt/shelf_lagrangian_windows.parquet
  + data/velocity_itslive_annual.nc + BedMachine + FAC
        -> data/dhdt/shelf_windows_divergence.parquet
        -> outputs/tables/shelf_divergence_report.json

Qual termo, e por quê
---------------------
Com DH/Dt LAGRANGIANO a equação de conservação é

    ṁ_b = a_s − DH/Dt − H·∇·v

usando **H·∇·v**, NÃO a divergência completa ∇·(H·v). Como
∇·(H·v) = v·∇H + H·∇·v, empregar a forma completa junto de DH/Dt contaria a
advecção DUAS vezes — uma no seguimento da parcela, outra no termo v·∇H. Este
script calcula deliberadamente apenas H·∇·v.

Espessura
---------
H vem do BedMachine (`thickness`). Duas ressalvas registradas na saída:

* a época nominal do BedMachine é 2015 e o período é 2019-2025, então H é um
  estado inicial defasado, não a espessura contemporânea;
* sobre gelo flutuante, a alternativa seria derivar H do freeboard por
  hidrostática, `H = ρw/(ρw−ρi)·(h − FAC)`. Isso é calculado aqui em paralelo,
  como VERIFICAÇÃO — a diferença entre as duas estimativas mede o quanto a
  defasagem do BedMachine e o FAC importam, em vez de deixar a escolha
  implícita.

Suavização antes de derivar
---------------------------
Derivar um campo ruidoso amplifica ruído: sem suavizar, textura da velocidade
vira divergência aparente. A escala de suavização é parâmetro explícito e a
sensibilidade a ela é reportada — não fixada em silêncio.

Uso:
    python pipelines/run_shelf_divergence.py [--smooth-km 3]
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
from thwaites.io.memory import free_memory_gb


def _smooth_nan(a, sigma_px):
    """Suavização gaussiana preservando NaN (média ponderada por validade)."""
    from scipy.ndimage import gaussian_filter
    if sigma_px <= 0:
        return a
    v = np.isfinite(a)
    num = gaussian_filter(np.where(v, a, 0.0), sigma_px)
    den = gaussian_filter(v.astype(float), sigma_px)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 1e-6, num / den, np.nan)
    return out


def divergence_field(vx, vy, dx, dy, smooth_px):
    """
    ∇·v = ∂vx/∂x + ∂vy/∂y, em 1/ano.

    Suaviza ANTES de derivar. `np.gradient` usa diferenças centrais, então o
    espaçamento entra com o sinal correto (dy pode ser negativo se o eixo for
    decrescente — aqui os eixos já vêm crescentes).
    """
    vxs = _smooth_nan(vx, smooth_px)
    vys = _smooth_nan(vy, smooth_px)
    dvx_dx = np.gradient(vxs, dx, axis=1)
    dvy_dy = np.gradient(vys, dy, axis=0)
    return dvx_dx + dvy_dy


def main():
    ap = argparse.ArgumentParser(description="Termo H·∇·v nas parcelas.")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--parcels", default="shelf_lagrangian_windows.parquet")
    ap.add_argument("--velocity", default="velocity_itslive_annual.nc")
    ap.add_argument("--smooth-km", type=float, default=3.0)
    ap.add_argument("--decimate", type=int, default=4)
    args = ap.parse_args()

    cfg = load_config(args.profile)
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level,
                        run_name="shelf_divergence")

    import xarray as xr
    from netCDF4 import Dataset

    par_p = cfg.paths.dhdt_dir / args.parcels
    if not par_p.exists():
        raise FileNotFoundError(f"{par_p} não existe (rode run_shelf_windows.py).")
    par = pd.read_parquet(par_p)
    log.info(f"{len(par):,} registros de parcela | "
             f"{par.groupby(['x_ref','y_ref']).ngroups:,} parcelas distintas")

    # ---- velocidade (decimada; derivar não precisa da resolução nativa) ----
    D = max(1, args.decimate)
    with Dataset(cfg.paths.data_dir / args.velocity) as d:
        vx_ = np.ma.filled(np.asarray(d["vx"][:, ::D, ::D], float), np.nan)
        vy_ = np.ma.filled(np.asarray(d["vy"][:, ::D, ::D], float), np.nan)
        gx = np.asarray(d["x"][::D], float)
        gy = np.asarray(d["y"][::D], float)
        t = d["time"]
        try:
            import cftime
            dts = cftime.num2date(t[:], t.units, only_use_cftime_datetimes=False)
            tv = np.array([x.year + (x.timetuple().tm_yday - 1) / 365.25
                           for x in np.atleast_1d(dts)])
        except Exception:
            tv = np.asarray(t[:], float)
    if gy[0] > gy[-1]:
        gy, vx_, vy_ = gy[::-1], vx_[:, ::-1, :], vy_[:, ::-1, :]
    dx = float(gx[1] - gx[0])
    dy = float(gy[1] - gy[0])
    smooth_px = (args.smooth_km * 1000.0) / abs(dx)
    log.info(f"velocidade {vx_.shape} @ {abs(dx):.0f} m | suavização "
             f"{args.smooth_km:.1f} km ({smooth_px:.1f} px) | "
             f"livre {free_memory_gb():.1f} GB")

    # Reusa a primeira época na análise de sensibilidade.
    vx_reference = vx_[0].copy()
    vy_reference = vy_[0].copy()
    DIV = np.stack([divergence_field(vx_[i], vy_[i], dx, dy, smooth_px)
                    for i in range(vx_.shape[0])])
    del vx_, vy_

    # ---- espessura e FAC ---------------------------------------------------
    cands = sorted(cfg.paths.data_dir.glob("*BedMachineAntarctica*.nc"))
    ds = xr.open_dataset(cands[0])
    bx = np.asarray(ds["x"].values, float)
    by = np.asarray(ds["y"].values, float)
    x0, x1 = par["x_ref"].min() - 20e3, par["x_ref"].max() + 20e3
    y0, y1 = par["y_ref"].min() - 20e3, par["y_ref"].max() + 20e3
    ix = np.where((bx >= x0) & (bx <= x1))[0]
    iy = np.where((by >= y0) & (by <= y1))[0]
    H = np.asarray(ds["thickness"].isel(x=ix, y=iy).values, float)
    hx, hy = bx[ix], by[iy]
    ds.close()
    if hy[0] > hy[-1]:
        hy, H = hy[::-1], H[::-1, :]

    def samp(F, ax, ay, px, py):
        j = np.clip(np.rint((px - ax[0]) / (ax[1] - ax[0])).astype(np.int64),
                    0, len(ax) - 1)
        i = np.clip(np.rint((py - ay[0]) / (ay[1] - ay[0])).astype(np.int64),
                    0, len(ay) - 1)
        return F[i, j]

    px = par["x_ref"].to_numpy()
    py = par["y_ref"].to_numpy()
    tc = par["t_center"].to_numpy()

    # ∇·v na época central de cada janela (interpolação linear entre anos)
    div = np.full(len(par), np.nan)
    for k in range(len(tv) - 1):
        m = (tc >= tv[k]) & (tc <= tv[k + 1])
        if not m.any():
            continue
        w = (tc[m] - tv[k]) / max(tv[k + 1] - tv[k], 1e-9)
        d0 = samp(DIV[k], gx, gy, px[m], py[m])
        d1 = samp(DIV[k + 1], gx, gy, px[m], py[m])
        div[m] = (1 - w) * d0 + w * d1
    # janelas fora do intervalo coberto: usa a época extrema (extrapolação)
    out_lo = tc < tv[0]
    out_hi = tc > tv[-1]
    if out_lo.any():
        div[out_lo] = samp(DIV[0], gx, gy, px[out_lo], py[out_lo])
    if out_hi.any():
        div[out_hi] = samp(DIV[-1], gx, gy, px[out_hi], py[out_hi])
    n_extrap = int(out_lo.sum() + out_hi.sum())

    h_bm = samp(H, hx, hy, px, py)
    par["div_v"] = div
    par["H_bedmachine"] = h_bm
    par["H_divv"] = h_bm * div

    # ---- amplificação hidrostática -----------------------------------------
    rho_i = cfg.mass_balance.ice_density
    rho_w = cfg.flux.water_density
    amp = rho_w / (rho_w - rho_i)
    # Não reconstruir um "H hidrostático" a partir do próprio H do BedMachine:
    # isso era circular e sempre devolvia exatamente a entrada. Um H(t)
    # independente exige freeboard observado + altura instantânea do mar/FAC ou
    # radar contemporâneo. Até lá, a hipótese de época 2015 fica explícita.
    par["H_source"] = "BedMachine_v4_nominal_2015"

    dst = cfg.paths.dhdt_dir / "shelf_windows_divergence.parquet"
    par.to_parquet(dst, index=False)

    ok = np.isfinite(par["H_divv"])
    rel = ok & par["reliable"] if "reliable" in par else ok
    rep = {
        "termo": "H*div(v) — NAO div(H*v): com DH/Dt lagrangiano a forma "
                 "completa contaria a adveccao duas vezes",
        "smooth_km": args.smooth_km,
        "n_registros": int(len(par)),
        "n_com_termo": int(ok.sum()),
        "n_extrapolados_no_tempo": n_extrap,
        "div_v_mediano_por_ano": float(np.nanmedian(par.loc[ok, "div_v"])),
        "H_mediano_m": float(np.nanmedian(par.loc[ok, "H_bedmachine"])),
        "H_divv_mediano_m_ano": float(np.nanmedian(par.loc[ok, "H_divv"])),
        "H_divv_p10": float(np.nanpercentile(par.loc[ok, "H_divv"], 10)),
        "H_divv_p90": float(np.nanpercentile(par.loc[ok, "H_divv"], 90)),
        "H_divv_mediano_confiaveis": (float(np.nanmedian(par.loc[rel, "H_divv"]))
                                      if rel.any() else None),
        "hydrostatic_amplification": float(amp),
        "thickness_validation": (
            "NAO_REALIZADA: a verificacao anterior era circular, pois inferia "
            "freeboard do proprio H_BedMachine e reconstruia o mesmo H"),
        "ressalvas": [
            "H do BedMachine tem epoca nominal 2015; o periodo e 2019-2025 — "
            "e estado inicial defasado, nao espessura contemporanea",
            "derivar velocidade amplifica ruido: suavizacao aplicada ANTES de "
            f"derivar ({args.smooth_km:.1f} km); ver sensibilidade abaixo",
            "faltam a_s (SMB) e H(t) para fechar m_b",
        ],
    }

    # sensibilidade à suavização — o parâmetro mais arbitrário deste passo
    sens = {}
    for s_km in (1.0, 2.0, 3.0, 5.0, 8.0):
        spx = (s_km * 1000.0) / abs(dx)
        dv = divergence_field(vx_reference, vy_reference, dx, dy, spx)
        v = samp(dv, gx, gy, px, py)
        sens[f"{s_km:.0f}km"] = {
            "div_v_mediano": float(np.nanmedian(v)),
            "H_divv_mediano": float(np.nanmedian(v * h_bm)),
        }
    rep["sensibilidade_suavizacao"] = sens

    rp = cfg.paths.tables / "shelf_divergence_report.json"
    rp.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info(f"Divergência -> {dst}")
    log.info(f"  div(v) mediano: {rep['div_v_mediano_por_ano']:+.5f} /ano")
    log.info(f"  H mediano: {rep['H_mediano_m']:,.0f} m")
    log.info(f"  H*div(v) mediano: {rep['H_divv_mediano_m_ano']:+.3f} m/ano "
             f"(p10 {rep['H_divv_p10']:+.2f}, p90 {rep['H_divv_p90']:+.2f})")
    log.info(f"  sensibilidade à suavização: "
             f"{ {k: round(v['H_divv_mediano'], 3) for k, v in sens.items()} }")
    if n_extrap:
        log.warning(f"{n_extrap:,} registros com ∇·v extrapolado no tempo")
    log.info(f"Relatório -> {rp}")


if __name__ == "__main__":
    main()
