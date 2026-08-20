"""
pipelines/fetch_icelines.py
===========================
Frentes de calving DATADAS (IceLines / Sentinel-1, DLR) para as plataformas do
Amundsen Sea Embayment.

    (DLR Geoservice) -> data/calving_fronts.parquet

Por que isto é necessário
-------------------------
A máscara do BedMachine é ESTÁTICA (`nominal_year = 2015`, `time_coverage` até
2019-10). A frente de calving recua e avança: observações de 2024-25
classificadas como plataforma podem já ser oceano, mélange ou iceberg. Sem
frente contemporânea, o domínio flutuante é classificado com a geometria errada
e o erro entra no ṁ_b sem deixar rastro.

Cobertura real (verificada, não presumida)
------------------------------------------
O IceLines publica em três frequências, com convenções de nome distintas e
lacunas — a cobertura NÃO é uniforme:

    annual     `2020noQ1_mean-Thwaites1.gpkg`
    quarterly  `2023Q1__mean-Thwaites1.gpkg`
    monthly    `1SSH_202104-Thwaites1.gpkg`   (prefixo = sensor)

Medido na Thwaites1: anual tem 2015, 2016, 2020, 2021, 2022 (faltam 2017-2019 e
2023+); trimestral tem 2015-2017, 2023, 2024. As três frequências são coletadas
juntas justamente para maximizar a cobertura temporal, e as lacunas remanescentes
ficam registradas no relatório — nunca preenchidas silenciosamente.

Uso:
    python pipelines/fetch_icelines.py
    python pipelines/fetch_icelines.py --shelves Thwaites1 PineIsland
"""

import argparse
import json
import re
import ssl
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config
from thwaites.logging import setup_logging

BASE = "https://download.geoservice.dlr.de/icelines/files/"

# Plataformas do Amundsen Sea Embayment. Thwaites e Getz são publicadas em
# partes numeradas pelo próprio IceLines.
ASE_SHELVES = ["Thwaites1", "Thwaites2", "PineIsland", "Crosson", "Dotson",
               "Cosgrove", "Getz1", "Getz2", "Getz3", "Abbot1", "Abbot2"]

_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE
_UA = {"User-Agent": "Mozilla/5.0"}


def _get(url: str, timeout: int = 180) -> bytes:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout, context=_CTX) as r:
        return r.read()


def parse_epoch(fname: str) -> float | None:
    """
    Ano decimal a partir do nome do arquivo, cobrindo as três convenções.

    Devolve None se nenhuma casar — o arquivo é então registrado como não
    datável em vez de receber uma data inventada.
    """
    # mensal: 1SSH_202104-Shelf.gpkg  -> AAAAMM em qualquer posição
    m = re.search(r"(?:^|[_-])(\d{4})(\d{2})(?:[_-])", fname)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        if 2000 <= y <= 2100 and 1 <= mo <= 12:
            return y + (mo - 0.5) / 12.0
    # trimestral: 2023Q1__mean-Shelf.gpkg
    m = re.match(r"(\d{4})Q([1-4])", fname)
    if m:
        y, q = int(m.group(1)), int(m.group(2))
        return y + (q - 0.5) / 4.0
    # anual: 2020noQ1_mean-Shelf.gpkg
    m = re.match(r"(\d{4})", fname)
    if m:
        y = int(m.group(1))
        if 2000 <= y <= 2100:
            return y + 0.5
    return None


def list_fronts(shelf: str, freq: str, log) -> list[tuple[str, float]]:
    """(url, época decimal) dos GeoPackages de frente de uma plataforma."""
    url = f"{BASE}{shelf}/{freq}/fronts/"
    try:
        txt = _get(url).decode(errors="replace")
    except Exception as e:
        log.warning(f"  {shelf}/{freq}: {type(e).__name__}")
        return []
    out = []
    for f in re.findall(r'href="([^"?][^"]*\.gpkg)"', txt):
        ep = parse_epoch(f)
        if ep is None:
            log.warning(f"  {shelf}/{freq}: '{f}' sem data reconhecível — ignorado")
            continue
        out.append((url + f, ep))
    return out


def main():
    ap = argparse.ArgumentParser(description="Frentes de calving datadas (IceLines).")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--shelves", nargs="+", default=None)
    ap.add_argument("--freqs", nargs="+", default=["annual", "quarterly", "monthly"])
    ap.add_argument("--output", default="calving_fronts.parquet")
    args = ap.parse_args()

    cfg = load_config(args.profile)
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level, run_name="icelines")

    import geopandas as gpd

    shelves = args.shelves or ASE_SHELVES
    tmp = cfg.paths.data_dir / "raw_temp"
    tmp.mkdir(parents=True, exist_ok=True)

    rows = []
    per_shelf = {}
    for sh in shelves:
        found = []
        for fr in args.freqs:
            found += [(u, e, fr) for u, e in list_fronts(sh, fr, log)]
        if not found:
            log.warning(f"{sh}: nenhuma frente encontrada")
            per_shelf[sh] = {"n": 0, "epochs": []}
            continue

        got = 0
        for url, epoch, freq in sorted(found, key=lambda t: t[1]):
            fname = url.split("/")[-1]
            local = tmp / fname
            try:
                # baixa para arquivo temporário, lê e apaga — o .gpkg bruto não
                # precisa ficar em disco (mesma disciplina dos grânulos)
                local.write_bytes(_get(url))
                g = gpd.read_file(local)
                if g.empty:
                    continue
                g = g.to_crs(epsg=cfg.area.epsg_polar)
                for geom in g.geometry:
                    if geom is None or geom.is_empty:
                        continue
                    rows.append({"shelf": sh, "epoch_year": epoch, "freq": freq,
                                 "source_file": fname, "wkt": geom.wkt})
                got += 1
            except Exception as e:
                log.warning(f"  {fname}: {type(e).__name__}: {str(e)[:70]}")
            finally:
                if local.exists():
                    local.unlink()

        eps = sorted({e for _, e, _ in found})
        per_shelf[sh] = {"n": got, "epochs": [round(e, 3) for e in eps]}
        log.info(f"{sh}: {got} frentes | épocas "
                 f"{min(eps):.2f}–{max(eps):.2f}")

    if not rows:
        raise SystemExit("nenhuma frente obtida.")

    df = pd.DataFrame(rows)
    out = cfg.paths.data_dir / args.output
    df.to_parquet(out, index=False)

    # cobertura por ano do período do projeto — as LACUNAS são o dado crítico
    yrs = np.arange(cfg.temporal.year_start, cfg.temporal.year_end + 1)
    cover = {}
    for y in yrs:
        s = df[(df.epoch_year >= y) & (df.epoch_year < y + 1)]
        cover[int(y)] = {"n_frentes": int(len(s)),
                         "plataformas": sorted(s.shelf.unique().tolist())}

    report = {
        "output": out.name,
        "n_geometrias": int(len(df)),
        "plataformas": per_shelf,
        "cobertura_por_ano": cover,
        "anos_sem_frente": [int(y) for y in yrs if cover[int(y)]["n_frentes"] == 0],
        "fonte": "IceLines (DLR Geoservice), Sentinel-1",
        "crs": f"EPSG:{cfg.area.epsg_polar}",
        "limitacao": ("cobertura NÃO é uniforme: há anos e plataformas sem "
                      "frente publicada. Interpolar entre épocas é uma escolha "
                      "que deve ser declarada, não um preenchimento automático."),
    }
    rp = cfg.paths.tables / "calving_fronts_report.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info(f"Frentes -> {out} ({len(df):,} geometrias)")
    log.info(f"cobertura por ano: "
             f"{ {y: c['n_frentes'] for y, c in cover.items()} }")
    if report["anos_sem_frente"]:
        log.warning(f"anos SEM nenhuma frente: {report['anos_sem_frente']} — "
                    f"a máscara temporal terá de extrapolar nesses anos")
    log.info(f"Relatório -> {rp}")


if __name__ == "__main__":
    main()
