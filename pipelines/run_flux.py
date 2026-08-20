"""
pipelines/run_flux.py
=====================
Divergência de fluxo e derretimento basal (cubediv/cubemelt do captoolkit).

    data/velocity_thwaites.nc  (fetch_velocity.py)
  + BedMachine .nc            (fetch_bedmachine.py — variável 'thickness')
  + data/interim/dhdt_grid.parquet  (run_interpolation.py)
        -> data/interim/flux_divergence.nc
        -> outputs/tables/flux_summary.json

Só é fisicamente válido sobre gelo FLUTUANTE (usa equilíbrio hidrostático).
Leia as premissas declaradas em thwaites/glaciology/flux.py antes de publicar.

Uso: python pipelines/run_flux.py [--profile anual]
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
from thwaites.glaciology.flux import (
    load_velocity, sample_bedmachine, flux_divergence, basal_melt_rate,
    hydrostatic_thickness_rate, hydrostatic_amplification,
)


def main():
    import xarray as xr
    from scipy.interpolate import griddata

    ap = argparse.ArgumentParser(description="Divergência de fluxo e derretimento basal.")
    ap.add_argument("--profile", default=None)
    args = ap.parse_args()

    cfg = load_config(args.profile)
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level, run_name="flux")
    if not cfg.flux.enabled:
        log.warning("flux.enabled=false — ative na config após baixar a velocidade.")
        return

    # --- grade de cálculo: a da velocidade, reamostrada à resolução pedida ---
    log.info("Carregando velocidade MEaSUREs...")
    vx_x, vx_y, vx, vy = load_velocity(cfg)
    res = cfg.flux.grid_res_m
    gx = np.arange(vx_x.min(), vx_x.max() + res, res)
    gy = np.arange(vx_y.min(), vx_y.max() + res, res)
    log.info(f"Grade de cálculo: {len(gy)} × {len(gx)} @ {res:.0f} m")

    # reamostra a velocidade para a grade de cálculo
    from scipy.interpolate import RegularGridInterpolator
    GX, GY = np.meshgrid(gx, gy)
    pts = np.c_[GY.ravel(), GX.ravel()]
    VX = RegularGridInterpolator((vx_y, vx_x), vx, method="linear",
                                 bounds_error=False, fill_value=np.nan)(pts).reshape(GY.shape)
    VY = RegularGridInterpolator((vx_y, vx_x), vy, method="linear",
                                 bounds_error=False, fill_value=np.nan)(pts).reshape(GY.shape)

    log.info("Amostrando espessura do BedMachine...")
    H = sample_bedmachine(cfg, "thickness", gx, gy)

    # --- máscara de gelo flutuante ------------------------------------------
    # Amplificação hidrostática e derretimento basal só se aplicam a gelo
    # flutuante; a hipótese de flutuação livre é inválida sobre gelo aterrado.
    # 'mask' é categórica: amostrada por vizinho mais próximo.
    log.info("Amostrando máscara de flutuação do BedMachine (nearest)...")
    MASK = sample_bedmachine(cfg, "mask", gx, gy, method="nearest")
    floating = (np.rint(MASK) == cfg.mask.floating_class)
    n_float = int(floating.sum())
    log.info(f"gelo flutuante: {n_float:,}/{floating.size:,} células "
             f"({100*n_float/floating.size:.1f}%) = {n_float*(res/1000)**2:,.0f} km²")
    if n_float == 0:
        raise SystemExit(
            "Nenhuma célula de gelo flutuante na grade — o derretimento basal "
            "não é definido aqui. Verifique mask.floating_class na config "
            f"(atual: {cfg.mask.floating_class}) e a cobertura da ROI.")

    log.info("Calculando ∇·(H·v)...")
    div = flux_divergence(gx, gy, H, VX, VY, cfg)

    # --- dh/dt interpolado na mesma grade -----------------------------------
    grid_p = cfg.paths.interim / "dhdt_grid.parquet"
    if not grid_p.exists():
        raise FileNotFoundError(f"{grid_p} não existe (rode run_interpolation.py).")
    g = pd.read_parquet(grid_p, columns=["x", "y", "pred"])
    log.info(f"Interpolando dh/dt ({len(g):,} células) para a grade de fluxo...")
    DHDT = griddata((g["x"].to_numpy(), g["y"].to_numpy()), g["pred"].to_numpy(),
                    (GX, GY), method="linear")

    # --- espessura e derretimento basal (só gelo flutuante) -----------------
    # A amplificação hidrostática vale só onde o gelo flutua; sobre gelo
    # aterrado ∂H/∂t = dh/dt (fator 1).
    dHdt = hydrostatic_thickness_rate(DHDT, cfg, floating=floating)

    # ---- SMB: o que o dado disponível REALMENTE permite ---------------------
    # A variável SMB é selecionada pelo nome (não por posição) e o dado
    # cumulativo é derivado no tempo para obter taxa.
    # Limitação de FÍSICA que nenhum código conserta: `SMB_a` é uma
    # ANOMALIA relativa à climatologia 1980-2019, não o SMB total. Sem a
    # climatologia de referência, a conservação de massa fecha em
    # (ṁ_b − SMB_ref), não em ṁ_b.
    smb_rate = None
    smb_info = None
    smb_is_total = False
    if cfg.firn.enabled:
        # 1ª escolha: SMB TOTAL (arquivo de componentes) -> permite ṁ_b ABSOLUTO
        try:
            from thwaites.corrections.firn import smb_total_at
            smb_rate, smb_info = smb_total_at(GX.ravel(), GY.ravel(), cfg)
            smb_rate = smb_rate.reshape(GY.shape)
            smb_is_total = True
            log.info(f"SMB TOTAL aplicado (mediana "
                     f"{np.nanmedian(smb_rate):+.4f} m gelo/ano) -> ṁ_b ABSOLUTO")
        except Exception as e:
            log.warning(f"SMB total indisponível ({type(e).__name__}: {e})")
            # 2ª escolha: só a anomalia -> fecha em (ṁ_b − SMB_ref)
            try:
                from thwaites.corrections.firn import smb_anomaly_rate_at
                smb_rate, smb_info = smb_anomaly_rate_at(GX.ravel(), GY.ravel(), cfg)
                smb_rate = smb_rate.reshape(GY.shape)
                log.warning("usando ANOMALIA de SMB — a saída é (ṁ_b − SMB_ref)")
            except Exception as e2:
                log.warning(f"SMB indisponível ({type(e2).__name__}: {e2})")
    if smb_rate is None:
        log.warning("Sem termo de SMB — o resultado é (ṁ_b − SMB), não ṁ_b puro.")

    # NaN fora do gelo flutuante: o resíduo do balanço só é interpretável como
    # derretimento basal sob a plataforma.
    melt = basal_melt_rate(dHdt, div, smb_rate, floating=floating)
    melt_name = "basal_melt" if smb_is_total else "basal_melt_minus_smb_ref"

    # --- salva ---------------------------------------------------------------
    out = xr.Dataset(
        {
            "thickness": (("y", "x"), H),
            "vx": (("y", "x"), VX), "vy": (("y", "x"), VY),
            "flux_divergence": (("y", "x"), div),
            "dhdt": (("y", "x"), DHDT),
            "dHdt_hydrostatic": (("y", "x"), dHdt),
            # gravada junto para que qualquer análise a jusante possa refazer o
            # gating sem ter de reabrir o BedMachine
            "floating": (("y", "x"), floating.astype("int8")),
            # nome depende do que o dado permite: com SMB TOTAL é
            # derretimento basal absoluto; só com anomalia é (ṁ_b − SMB_ref).
            melt_name: (("y", "x"), melt),
        },
        coords={"x": gx, "y": gy},
        attrs={
            "crs": "EPSG:3031",
            "velocity_epoch": cfg.velocity.epoch_note,
            # NetCDF não aceita bool em atributo -> inteiro 0/1
            "smb_applied": int(smb_rate is not None),
            "smb_is_total": int(smb_is_total),
            "output_variable": melt_name,
            "note": ("Válido apenas sobre gelo FLUTUANTE (hipótese hidrostática) "
                     "e assumindo v_coluna ≈ v_superfície. Velocidade é mosaico "
                     "1996-2018 vs dh/dt 2019-2025."),
        },
    )
    dst = cfg.paths.interim / "flux_divergence.nc"
    out.to_netcdf(dst)
    log.info(f"Saída -> {dst}")

    fin = np.isfinite(melt)
    cell_km2 = (res / 1000.0) ** 2
    # Estatísticas de ṁ_b restritas ao gelo flutuante (é onde a grandeza existe).
    # As de ∇·(H·v) também: sobre gelo aterrado a divergência é um diagnóstico
    # legítimo, mas misturá-la na mesma mediana torna o número ininterpretável.
    div_float = div[floating & np.isfinite(div)]
    summary = {
        "n_cells_valid": int(fin.sum()),
        "n_cells_floating": int(floating.sum()),
        "area_floating_km2": float(floating.sum() * cell_km2),
        "area_melt_valid_km2": float(fin.sum() * cell_km2),
        "grid_res_m": res,
        "hydrostatic_amplification": float(hydrostatic_amplification(cfg)),
        "flux_div_median_m_yr": (float(np.median(div_float))
                                 if div_float.size else None),
        "flux_div_median_all_m_yr": float(np.nanmedian(div)),
        "melt_median_m_yr": (float(np.nanmedian(melt[fin])) if fin.any() else None),
        "melt_p25_m_yr": (float(np.percentile(melt[fin], 25)) if fin.any() else None),
        "melt_p75_m_yr": (float(np.percentile(melt[fin], 75)) if fin.any() else None),
        "output_variable": melt_name,
        "smb_applied": bool(smb_rate is not None),
        "smb_is_total": bool(smb_is_total),
        "smb_median_m_ice_yr": (float(np.nanmedian(smb_rate[floating]))
                                if smb_rate is not None else None),
        "output_meaning": ("derretimento basal ABSOLUTO" if smb_is_total
                           else "(m_b - SMB_ref)"),
        "gating": (f"estatísticas restritas a mask=={cfg.mask.floating_class} "
                   f"(gelo flutuante do BedMachine, vizinho mais próximo)"),
        "velocity_epoch": cfg.velocity.epoch_note,
        "caveats": [
            "válido só sobre gelo flutuante (hipótese hidrostática) — AGORA "
            "imposto no código, não apenas declarado",
            "v_coluna ≈ v_superfície",
            "SMB do GSFC-FDM cobre 2019-06/2022; extrapolado ate 2025",
            "descasamento temporal velocidade × dh/dt",
        ],
    }
    cfg.paths.tables.mkdir(parents=True, exist_ok=True)
    (cfg.paths.tables / "flux_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info(f"Resumo -> {cfg.paths.tables / 'flux_summary.json'}")


if __name__ == "__main__":
    main()
