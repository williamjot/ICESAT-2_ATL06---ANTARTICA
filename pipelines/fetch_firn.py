"""
pipelines/fetch_firn.py
=======================
Baixa o GSFC-FDM v1.2.1 (Zenodo, acesso aberto) e recorta à ROI da Thwaites.

O QUE É: modelo de densificação de firn da NASA/GSFC, forçado por MERRA-2,
grade de 12,5 km, resolução temporal de 5 dias, 1980–jun/2022. Fornece o
**firn air content (FAC)** e o SMB. Foi produzido pelo escritório científico do
ICESat-2 exatamente para separar mudança de volume atmosférica da dinâmica.

POR QUE PRECISAMOS: sem ele, converter dh/dt em massa com ρ=917 uniforme trata
compactação de firn (mudança de ALTURA sem mudança de MASSA) como perda de gelo.

DISCIPLINA DE DISCO/MEMÓRIA (mesma do ATL06): o .zip tem ~2,8 GB e o conteúdo
descomprimido é bem maior. Aqui cada membro do zip é extraído para uma pasta
temporária, IMEDIATAMENTE recortado à ROI e ao período de interesse, e o bruto
é apagado num `try/finally`. O produto final tem poucos MB.

Saída: data/firn_thwaites.nc  (FAC e afins recortados)

Uso:
    python pipelines/fetch_firn.py                # FAC
    python pipelines/fetch_firn.py --which smb    # SMB
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config
from thwaites.grid.reproject import to_polar

ZENODO_RECORD = "7221954"
FILES = {
    "fac": "gsfc_fdm_v1_2_1_ais_June22.zip",
    "smb": "gsfc_fdm_smb_v1_2_1_ais_June22.zip",
}
BASE_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files"


def download(url: str, dest: Path, chunk: int = 8 * 1024 * 1024):
    """Download em streaming, com progresso (não carrega o arquivo na RAM)."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        got = 0
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            f.write(buf)
            got += len(buf)
            if total:
                print(f"\r      {got/1024**3:.2f}/{total/1024**3:.2f} GB "
                      f"({100*got/total:.0f}%)", end="", flush=True)
    print()
    return dest


def roi_bounds_polar(cfg, buffer_km: float = 100.0):
    """bbox da ROI em EPSG:3031 (a grade do GSFC-FDM é polar estereográfica)."""
    roi = cfg.roi or cfg.area
    clon = np.array([roi.lon_min, roi.lon_max, roi.lon_min, roi.lon_max])
    clat = np.array([roi.lat_min, roi.lat_min, roi.lat_max, roi.lat_max])
    cx, cy = to_polar(clon, clat, cfg)
    b = buffer_km * 1000.0
    return (float(np.min(cx)) - b, float(np.max(cx)) + b,
            float(np.min(cy)) - b, float(np.max(cy)) + b)


def crop_member(nc_path: Path, cfg, year_start: int) -> "xr.Dataset | None":
    """
    Abre o NetCDF do FDM de forma LAZY, recorta à ROI e ao período.

    ESTRUTURA REAL do GSFC-FDMv1.2.1 (verificada, não presumida):
      - dims: (time, x, y) — nesta ordem;
      - `x` e `y` são coordenadas **2D** (441, 401) em EPSG:3031, passo 12,5 km,
        mas a grade É regular: x varia só ao longo do dim x, y só ao longo do
        dim y. Por isso extraímos eixos 1D e recortamos por índice;
      - `time` já vem em **anos decimais** (1980,007 → 2022,484), então não há
        decodificação de calendário;
      - variáveis: FAC (m de ar), SMB_a (m de gelo, cumulativo), h_a (m).

    A saída é transposta para (time, y, x) com coordenadas 1D — a convenção que
    thwaites.corrections.firn espera.
    """
    import numpy as np
    import xarray as xr

    x0, x1, y0, y1 = roi_bounds_polar(cfg)
    ds = xr.open_dataset(nc_path, chunks={}, decode_times=False)

    if "x" not in ds.coords or "y" not in ds.coords:
        print(f"      (sem coords x/y: {list(ds.coords)}) — pulado")
        ds.close()
        return None

    X = np.asarray(ds["x"].values)
    Y = np.asarray(ds["y"].values)
    if X.ndim == 2:
        # confirma a regularidade antes de reduzir a 1D
        if not (np.allclose(X[:, 0], X[:, -1]) and np.allclose(Y[0, :], Y[-1, :])):
            print("      (grade 2D NÃO regular — recorte por índice inválido) — pulado")
            ds.close()
            return None
        ax = X[:, 0].astype(float)      # eixo do dim x
        ay = Y[0, :].astype(float)      # eixo do dim y
    else:
        ax = X.astype(float)
        ay = Y.astype(float)

    ix = np.flatnonzero((ax >= x0) & (ax <= x1))
    iy = np.flatnonzero((ay >= y0) & (ay <= y1))
    if ix.size == 0 or iy.size == 0:
        print(f"      (ROI fora da grade do FDM) — pulado")
        ds.close()
        return None

    sel = {"x": slice(int(ix[0]), int(ix[-1]) + 1),
           "y": slice(int(iy[0]), int(iy[-1]) + 1)}
    if "time" in ds.dims:
        t = np.asarray(ds["time"].values, dtype=float)   # anos decimais
        it = np.flatnonzero(t >= year_start)
        if it.size:
            sel["time"] = slice(int(it[0]), int(it[-1]) + 1)

    sub = ds.isel(sel)
    # substitui as coords 2D por eixos 1D e materializa (já pequeno)
    sub = sub.drop_vars([c for c in ("x", "y") if c in sub.coords])
    sub = sub.assign_coords(x=("x", ax[ix]), y=("y", ay[iy]))
    out = sub.load()
    ds.close()

    # convenção downstream: (time, y, x)
    dims = [d for d in ("time", "y", "x") if d in out.dims]
    out = out.transpose(*dims)
    print(f"        recortado: {dict(out.sizes)} | vars {list(out.data_vars)}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Baixa e recorta o GSFC-FDM (firn).")
    ap.add_argument("--which", choices=["fac", "smb"], default="fac")
    ap.add_argument("--keep-zip", action="store_true",
                    help="mantém o .zip (default: apaga após o recorte)")
    args = ap.parse_args()

    import xarray as xr

    cfg = load_config()
    data_dir = cfg.paths.data_dir
    tmp = cfg.paths.raw_temp / "firn"
    tmp.mkdir(parents=True, exist_ok=True)

    fname = FILES[args.which]
    zip_path = data_dir / fname
    url = f"{BASE_URL}/{fname}/content"

    print(f"[1/4] GSFC-FDM v1.2.1 ({args.which.upper()}) — Zenodo {ZENODO_RECORD}")
    if zip_path.exists() and zip_path.stat().st_size > 1_000_000:
        print(f"      zip já existe ({zip_path.stat().st_size/1024**3:.2f} GB)")
    else:
        print(f"      baixando {fname}...")
        download(url, zip_path)

    print("[2/4] inspecionando o zip...")
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if m.endswith(".nc")]
    print(f"      {len(members)} NetCDF(s) no arquivo")
    if not members:
        raise SystemExit("nenhum .nc dentro do zip.")

    print("[3/4] extraindo, recortando à ROI e apagando o bruto (um por vez)...")
    crops = []
    for i, m in enumerate(members, start=1):
        extracted = None
        try:
            with zipfile.ZipFile(zip_path) as z:
                extracted = Path(z.extract(m, path=tmp))
            print(f"      [{i}/{len(members)}] {Path(m).name} "
                  f"({extracted.stat().st_size/1024**2:.0f} MB)")
            c = crop_member(extracted, cfg, cfg.temporal.year_start)
            if c is not None and len(c.data_vars):
                crops.append(c)
        except Exception as e:
            print(f"        erro em {m}: {type(e).__name__}: {e}")
        finally:
            # apaga o bruto SEMPRE (mesma regra do ATL06)
            if extracted and extracted.exists():
                try:
                    extracted.unlink()
                except OSError:
                    pass

    if not crops:
        raise SystemExit("nada foi recortado — verifique os nomes de coordenadas.")

    print("[4/4] consolidando o recorte...")
    merged = xr.merge(crops, compat="override", join="outer") if len(crops) > 1 else crops[0]
    out = data_dir / (f"firn_thwaites.nc" if args.which == "fac"
                      else "smb_thwaites.nc")
    merged.to_netcdf(out)
    for c in crops:
        c.close()
    shutil.rmtree(tmp, ignore_errors=True)
    if not args.keep_zip:
        zip_path.unlink(missing_ok=True)
        print(f"      zip apagado (use --keep-zip para preservar)")

    print(f"\nSaída: {out} ({out.stat().st_size/1024**2:.1f} MB)")
    print(f"  variáveis: {list(merged.data_vars)}")
    print(f"  dims: {dict(merged.sizes)}")
    print("\nATENÇÃO: o GSFC-FDM termina em 30/06/2022 e o dh/dt vai a 2025 — "
          "a correção de firn cobre ~3,5 dos 7 invernos. Declare a extrapolação.")
    print("Próximo: python pipelines/run_firn.py")


if __name__ == "__main__":
    main()
