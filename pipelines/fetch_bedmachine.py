"""
pipelines/fetch_bedmachine.py
=============================
Baixa o BedMachine Antarctica (NSIDC-0756, NetCDF) via earthaccess e converte
a variável `mask` para GeoTIFF em EPSG:3031 — o formato que thwaites.qc.mask
espera (banda 1, polar estereográfica).

O BedMachine já está em EPSG:3031, então a conversão é só extrair a grade e
gravar com o transform correto. Os códigos de `mask` são LIDOS dos metadados
(flag_values/flag_meanings) e comparados com a config — não assumidos.

Saída: data/BedMachineAntarctica.tif  (bate com config mask.bedmachine_path)
Mantém o .nc de origem (tem bed/thickness úteis para trabalhos futuros).

Uso:
    python pipelines/fetch_bedmachine.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config

SHORT_NAME = "NSIDC-0756"   # BedMachine Antarctica


def main():
    import earthaccess
    import xarray as xr
    import rasterio
    from rasterio.transform import from_origin

    cfg = load_config()
    data_dir = cfg.paths.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Autenticando (Earthdata)...")
    earthaccess.login()

    print("[2/4] Buscando o grânulo BedMachine Antarctica...")
    results = earthaccess.search_data(short_name=SHORT_NAME, count=1)
    if not results:
        raise SystemExit("Nenhum grânulo NSIDC-0756 encontrado.")
    fname = results[0].data_links()[0].split("/")[-1]
    nc_path = data_dir / fname
    if nc_path.exists() and nc_path.stat().st_size > 1_000_000:
        print(f"      NetCDF já existe ({nc_path.name}), pulando download.")
    else:
        print(f"      Baixando {fname} (~3-4 GB, pode demorar)...")
        got = earthaccess.download([results[0]], str(data_dir))
        nc_path = Path(got[0])
    print(f"      NetCDF: {nc_path}  ({nc_path.stat().st_size/1024**3:.2f} GB)")

    print("[3/4] Convertendo `mask` -> GeoTIFF EPSG:3031...")
    ds = xr.open_dataset(nc_path)
    if "mask" not in ds:
        raise SystemExit(f"variável 'mask' ausente. Variáveis: {list(ds.data_vars)}")

    mvar = ds["mask"]
    x = ds["x"].values.astype("float64")
    y = ds["y"].values.astype("float64")
    mask = np.asarray(mvar.values)

    # Reporta os códigos declarados nos metadados (não assume).
    fv = mvar.attrs.get("flag_values")
    fm = mvar.attrs.get("flag_meanings")
    print(f"      flag_values  : {fv}")
    print(f"      flag_meanings: {fm}")
    print(f"      valores únicos presentes: {np.unique(mask)}")

    dx, dy = abs(x[1] - x[0]), abs(y[1] - y[0])
    # Garante orientação north-up (linha 0 = maior y) e west->east.
    if y[0] < y[-1]:
        y = y[::-1]; mask = mask[::-1, :]
    if x[0] > x[-1]:
        x = x[::-1]; mask = mask[:, ::-1]
    west = x.min() - dx / 2
    north = y.max() + dy / 2
    transform = from_origin(west, north, dx, dy)

    mask = mask.astype("uint8")   # códigos 0..4 cabem em uint8
    tif_path = data_dir / "BedMachineAntarctica.tif"
    with rasterio.open(
        tif_path, "w", driver="GTiff",
        height=mask.shape[0], width=mask.shape[1], count=1,
        dtype="uint8", crs="EPSG:3031", transform=transform,
        compress="deflate", predictor=2, tiled=True,
    ) as dst:
        dst.write(mask, 1)
        dst.update_tags(1, source="BedMachine Antarctica NSIDC-0756",
                        flag_meanings=str(fm))
    ds.close()
    print(f"      GeoTIFF: {tif_path}  ({tif_path.stat().st_size/1024**2:.0f} MB)")

    print("[4/4] Conferindo contra a config...")
    keep = set(cfg.mask.keep_values)
    floating = cfg.mask.floating_class
    print(f"      config keep_values = {sorted(keep)} | floating_class = {floating}")
    print(f"      config mask.bedmachine_path = {cfg.mask.bedmachine_path}")
    print("      >>> Confira acima se os códigos (flag_meanings) batem: "
          "0=oceano descartado, 3=gelo flutuante p/ gating de maré.")
    print("\nPronto. Próximo passo: python pipelines/run_mask.py")


if __name__ == "__main__":
    main()
