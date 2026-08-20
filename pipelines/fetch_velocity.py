"""
pipelines/fetch_velocity.py
===========================
Baixa o mapa de velocidade do gelo MEaSUREs e recorta à ROI da Thwaites.

Produto padrão: **NSIDC-0754** (MEaSUREs Phase-Based Antarctica Ice Velocity
Map, v1) — 450 m, baseado em fase InSAR, precisão ~10× melhor que os mapas por
feature/speckle tracking. Alternativa: NSIDC-0484 v2 (InSAR-based, com desvio
padrão de vx/vy por pixel).

Saída: data/velocity_thwaites.nc  (vx, vy [m/ano] em EPSG:3031, recortado)

LIMITAÇÃO A DECLARAR: o NSIDC-0754 é um mosaico de 1996–2018, enquanto o nosso
dh/dt é de 2019–2025. A Thwaites ACELEROU nesse intervalo, então a divergência
de fluxo calculada com essa velocidade subestima o fluxo atual. Ao publicar,
declare o descasamento temporal (ou use NSIDC-0484 anual, que cobre 2011–2016
com incerteza por pixel, ou uma versão mais recente se disponível).

Uso: python pipelines/fetch_velocity.py [--short-name NSIDC-0754]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config
from thwaites.grid.reproject import to_polar
from thwaites.io.gridded import crop_gridded_to_roi


def main():
    import earthaccess
    import xarray as xr

    ap = argparse.ArgumentParser(description="Baixa velocidade MEaSUREs e recorta à ROI.")
    ap.add_argument("--short-name", default=None,
                    help="produto NSIDC (default: da config velocity.short_name)")
    ap.add_argument("--buffer-km", type=float, default=50.0)
    args = ap.parse_args()

    cfg = load_config()
    short_name = args.short_name or cfg.velocity.short_name
    data_dir = cfg.paths.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Autenticando (Earthdata)...")
    earthaccess.login()

    print(f"[2/4] Buscando {short_name}...")
    results = earthaccess.search_data(short_name=short_name,
                                      version=cfg.velocity.version, count=5)
    if not results:
        raise SystemExit(f"Nenhum grânulo encontrado para {short_name}.")
    links = [l for l in results[0].data_links() if l.endswith(".nc")]
    if not links:
        raise SystemExit(f"Nenhum .nc no grânulo: {results[0].data_links()[:3]}")
    fname = links[0].split("/")[-1]
    nc_path = data_dir / fname
    if nc_path.exists() and nc_path.stat().st_size > 1_000_000:
        print(f"      já existe ({nc_path.name}), pulando download.")
    else:
        print(f"      baixando {fname} (vários GB, pode demorar)...")
        got = earthaccess.download([results[0]], str(data_dir))
        nc_path = Path(got[0])
    print(f"      NetCDF: {nc_path} ({nc_path.stat().st_size/1024**3:.2f} GB)")

    print("[3/4] Recortando à ROI da Thwaites...")
    # inspeciona o que existe ANTES de recortar (não assume nomes de variável)
    ds0 = xr.open_dataset(nc_path, chunks={}, decode_times=False)
    avail = list(ds0.data_vars)
    print(f"      variáveis no arquivo: {avail}")
    print(f"      coords: {list(ds0.coords)} | dims: {dict(ds0.sizes)}")
    ds0.close()

    wanted = [cfg.velocity.vx_var, cfg.velocity.vy_var]
    missing = [v for v in wanted if v not in avail]
    if missing:
        raise SystemExit(
            f"variáveis {missing} ausentes. Disponíveis: {avail}. "
            f"Ajuste `velocity.vx_var`/`velocity.vy_var` na config.")
    # leva também as incertezas, se o produto as trouxer (§7.3 pede preservá-las)
    extras = [v for v in avail
              if any(k in v.upper() for k in ("ERR", "STD", "SIGMA", "COUNT"))]
    if extras:
        print(f"      incertezas/qualidade preservadas: {extras}")

    # recorte robusto: aceita coords 1D ou 2D e devolve (y, x) com coords 1D
    # (o helper foi validado contra o formato 2D do GSFC-FDM e o 1D típico
    #  do MEaSUREs, justamente para não perder um download por suposição)
    sub = crop_gridded_to_roi(nc_path, cfg, variables=wanted + extras,
                              buffer_km=args.buffer_km)
    if sub is None:
        raise SystemExit("recorte falhou — ver mensagem acima.")

    out = data_dir / "velocity_thwaites.nc"
    sub.to_netcdf(out)
    sub.close()

    print(f"[4/4] Salvo: {out} ({out.stat().st_size/1024**2:.0f} MB)")
    print("\nATENÇÃO: declare o descasamento temporal (velocidade 1996–2018 vs "
          "dh/dt 2019–2025) — a Thwaites acelerou no período.")
    print("Próximo: python pipelines/run_flux.py")


if __name__ == "__main__":
    main()
