"""
pipelines/fetch_itslive.py
==========================
Velocidade ANUAL do ITS_LIVE v2, recortada à ROI.

    (AWS S3 público, zarr) -> data/velocity_itslive_annual.nc

Por que este produto e não o MEaSUREs NSIDC-0754
------------------------------------------------
O NSIDC-0754 é um MOSAICO de 1996–2018, enquanto o dh/dt é de 2019–2025. A
Thwaites acelerou no período, então usar aquele mosaico subestima ∇·(H·v) — e o
erro entra INTEGRALMENTE no resíduo interpretado como derretimento basal. Não há
tratamento posterior que corrija isso: é preciso velocidade contemporânea.

O ITS_LIVE fornece composites ANUAIS. Verificado empiricamente neste projeto:
cobrem 2012–2025 (14 épocas), em EPSG:3031 nativo, grade de 120 m, com
`v_error` por época — o que permite propagar incerteza de velocidade em vez de
ignorá-la.

Acesso
------
Os composites são zarr num bucket S3 público, um por tile de 100 km. A leitura é
REMOTA e recortada: só os chunks que cobrem a ROI trafegam. Baixar os mosaicos
anuais equivalentes custaria ~8 GB POR ANO (48 GB para a série), contra alguns
MB por tile aqui.

Nota de acesso: o protocolo `s3://` falha neste ambiente (incompatibilidade de
versão do botocore); a URL HTTPS do mesmo bucket funciona e é usada.

Uso:
    python pipelines/fetch_itslive.py
    python pipelines/fetch_itslive.py --year-min 2019 --year-max 2025
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config
from thwaites.logging import setup_logging

CATALOG = "https://its-live-data.s3.amazonaws.com/datacubes/catalog_v02.json"


def roi_composites(cfg, log) -> list[str]:
    """URLs dos composites anuais que intersectam a ROI (via catálogo oficial)."""
    log.info("baixando catálogo de datacubes do ITS_LIVE...")
    with urllib.request.urlopen(CATALOG, timeout=300) as r:
        cat = json.load(r)

    roi = cfg.roi or cfg.area
    urls = []
    for f in cat["features"]:
        g = f.get("geometry") or {}
        if g.get("type") != "Polygon":
            continue
        cs = np.asarray(g["coordinates"][0], dtype=float)
        if cs[:, 0].max() < roi.lon_min or cs[:, 0].min() > roi.lon_max:
            continue
        if cs[:, 1].max() < roi.lat_min or cs[:, 1].min() > roi.lat_max:
            continue
        p = f.get("properties", {})
        # só EPSG:3031 — misturar projeções exigiria reamostrar e não é o caso
        if int(p.get("epsg", 0)) != cfg.area.epsg_polar:
            continue
        u = p.get("composite_zarr_url")
        if u:
            urls.append(u)
    log.info(f"{len(urls)} composites intersectam a ROI")
    return sorted(set(urls))


def main():
    ap = argparse.ArgumentParser(description="Velocidade anual ITS_LIVE recortada à ROI.")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--year-min", type=int, default=None)
    ap.add_argument("--year-max", type=int, default=None)
    ap.add_argument("--output", default="velocity_itslive_annual.nc")
    ap.add_argument("--buffer-km", type=float, default=50.0)
    args = ap.parse_args()

    cfg = load_config(args.profile)
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level, run_name="itslive")

    import xarray as xr
    from thwaites.grid.reproject import to_polar

    y0 = args.year_min or cfg.temporal.year_start
    y1 = args.year_max or cfg.temporal.year_end

    roi = cfg.roi or cfg.area
    clon = np.array([roi.lon_min, roi.lon_max, roi.lon_min, roi.lon_max])
    clat = np.array([roi.lat_min, roi.lat_min, roi.lat_max, roi.lat_max])
    cx, cy = to_polar(clon, clat, cfg)
    b = args.buffer_km * 1000.0
    x0, x1 = float(cx.min()) - b, float(cx.max()) + b
    yy0, yy1 = float(cy.min()) - b, float(cy.max()) + b
    log.info(f"ROI (+{args.buffer_km:.0f} km): x [{x0/1e3:.0f},{x1/1e3:.0f}] "
             f"y [{yy0/1e3:.0f},{yy1/1e3:.0f}] km | anos {y0}–{y1}")

    urls = roi_composites(cfg, log)
    if not urls:
        raise SystemExit("nenhum composite do ITS_LIVE intersecta a ROI.")

    # Variáveis mantidas. `vx`/`vy` são os COMPONENTES anuais (confirmado no
    # produto: ambos têm dimensão `time`) — só a magnitude `v` não permitiria
    # calcular ∇·(H·v) nem integrar trajetórias. Os `*_error` entram porque a
    # incerteza de velocidade propaga direto para a divergência de fluxo.
    WANT = ["vx", "vy", "v", "vx_error", "vy_error", "v_error", "count"]

    # MEMÓRIA: montar a grade de destino UMA vez e preencher tile a tile.
    # Materializar as 37 peças e chamar combine_by_coords no fim consome ~5 GB
    # (37 x 7 épocas x 833² x 7 variáveis) — numa máquina de 8 GB isso derruba o
    # processo. Aqui só uma peça fica viva por vez.
    log.info("passo 1/2: descobrindo a grade de destino...")
    xs, ys, times = set(), set(), None
    metas = []
    for i, u in enumerate(urls, 1):
        try:
            ds = xr.open_dataset(u, engine="zarr", chunks={}, decode_timedelta=False)
        except Exception as e:
            log.warning(f"[{i}/{len(urls)}] {u.split('/')[-1]}: {type(e).__name__} — pulado")
            continue
        ysel = (slice(yy1, yy0) if float(ds.y[0]) > float(ds.y[-1])
                else slice(yy0, yy1))
        sub = ds[[v for v in WANT if v in ds.data_vars]].sel(x=slice(x0, x1), y=ysel)
        if sub.sizes.get("x", 0) and sub.sizes.get("y", 0):
            yrs = sub["time"].dt.year
            sub = sub.sel(time=(yrs >= y0) & (yrs <= y1))
            if sub.sizes.get("time", 0):
                xs.update(np.asarray(sub.x.values).tolist())
                ys.update(np.asarray(sub.y.values).tolist())
                if times is None:
                    times = np.asarray(sub["time"].values)
                metas.append((u, float(sub.x.min()), float(sub.x.max()),
                              float(sub.y.min()), float(sub.y.max())))
        ds.close()

    if not metas:
        raise SystemExit("nenhuma peça válida após o recorte.")
    gx = np.array(sorted(xs)); gy = np.array(sorted(ys))
    log.info(f"grade de destino: {len(times)} épocas x {len(gy)} y x {len(gx)} x "
             f"(~{len(times)*len(gy)*len(gx)*4/1024**2:.0f} MB por variável)")

    log.info("passo 2/2: preenchendo tile a tile...")
    fields = {v: np.full((len(times), len(gy), len(gx)), np.nan, dtype=np.float32)
              for v in WANT}
    filled = set()
    for i, (u, *_ ) in enumerate(metas, 1):
        ds = xr.open_dataset(u, engine="zarr", chunks={}, decode_timedelta=False)
        have = [v for v in WANT if v in ds.data_vars]
        ysel = (slice(yy1, yy0) if float(ds.y[0]) > float(ds.y[-1])
                else slice(yy0, yy1))
        sub = ds[have].sel(x=slice(x0, x1), y=ysel)
        yrs = sub["time"].dt.year
        sub = sub.sel(time=(yrs >= y0) & (yrs <= y1)).load()
        jx = np.searchsorted(gx, np.asarray(sub.x.values))
        iy = np.searchsorted(gy, np.asarray(sub.y.values))
        for v in have:
            arr = np.asarray(sub[v].values, dtype=np.float32)
            if arr.ndim == 3:
                fields[v][np.ix_(np.arange(len(times)), iy, jx)] = arr
                filled.add(v)
        del sub
        ds.close()
        if i % 5 == 0 or i == len(metas):
            log.info(f"  [{i}/{len(metas)}] tiles preenchidos")

    merged = xr.Dataset(
        {v: (("time", "y", "x"), fields[v]) for v in sorted(filled)},
        coords={"time": times, "y": gy, "x": gx},
    )

    out = cfg.paths.data_dir / args.output
    merged.attrs.update({
        "source": "ITS_LIVE v2 annual composites (NASA JPL), EPSG:3031, 120 m",
        "catalog": CATALOG,
        "roi": f"{roi.lon_min}..{roi.lon_max} / {roi.lat_min}..{roi.lat_max}",
        "years": f"{y0}-{y1}",
        "note": ("velocidade ANUAL — substitui o mosaico MEaSUREs NSIDC-0754 "
                 "(1996-2018), temporalmente incompatível com dh/dt 2019-2025"),
    })
    enc = {v: {"zlib": True, "complevel": 4} for v in merged.data_vars}
    merged.to_netcdf(out, encoding=enc)
    log.info(f"ITS_LIVE anual -> {out} ({out.stat().st_size/1024**2:.0f} MB)")
    log.info(f"  dims: {dict(merged.sizes)} | variáveis: {sorted(filled)}")
    t = merged["time"].values
    log.info(f"  épocas: {len(t)} | {str(t[0])[:10]} .. {str(t[-1])[:10]}")


if __name__ == "__main__":
    main()
