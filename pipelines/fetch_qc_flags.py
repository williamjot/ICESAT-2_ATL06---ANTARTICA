"""
pipelines/fetch_qc_flags.py
===========================
Re-extrai APENAS as variáveis de qualidade do ATL06, para os grânulos que de
fato contribuem com pontos para a ROI atual.

    (NASA Earthdata) -> data/qc_flags/<granulo>.parquet

Por que este script existe
--------------------------
A extração original guardou só seis variáveis (lon, lat, h_li, h_li_sigma,
delta_time, correções geofísicas). O flag `atl06_quality_summary` foi usado
como filtro mas NÃO salvo — e o filtro em si nunca filtrou nada, porque a
config usava `quality_max: 1` e o flag é BINÁRIO (0 = best_quality,
1 = potential_problem; verificado no atributo `flag_values` do produto). Com
`qual <= 1`, todos os pontos passavam.

Sem os flags não é possível fazer o controle de qualidade que a literatura de
ATL06 considera padrão (rugosidade/crevasses, neve soprada, nuvem, blunder de
detecção de superfície). Como os `.h5` são deletados por regra do projeto, a
única saída é re-baixar — mas só os grânulos que importam.

Correspondência com os dados já processados
-------------------------------------------
NÃO há coluna de chave nos Parquet já processados, então o pareamento é
POSICIONAL. Ele é válido porque a máscara de validade em `thwaites.io.extract`
é determinística e depende apenas do conteúdo do grânulo:

    valid = (qual <= quality_max) & (h < fill_value) & (h > h_min_valid)
            & ~isnan(h) & ~isnan(lat)

replicada aqui EXATAMENTE, na mesma ordem de feixes (`cfg.product.beams`).
Cada grânulo é verificado: se a contagem de linhas não bater com o Parquet
original, o grânulo é marcado como divergente e NÃO é gravado — falha alta,
nunca pareamento silenciosamente errado.

Disciplina de download: um grânulo por vez, `.h5` deletado dentro
de `finally`. Retomável: grânulos já extraídos são pulados.

Uso:
    python pipelines/fetch_qc_flags.py                 # todos os relevantes
    python pipelines/fetch_qc_flags.py --limit 20      # teste
    python pipelines/fetch_qc_flags.py --list-only     # só conta o trabalho
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config
from thwaites.logging import setup_logging

# Variáveis de qualidade, por grupo dentro de land_ice_segments.
# Confirmadas contra um grânulo real (não assumidas) — ver o docstring de
# thwaites/qc/atl06_flags.py para o significado de cada uma.
QC_ROOT = ["atl06_quality_summary", "segment_id", "sigma_geo_h"]
QC_FIT = ["h_robust_sprd", "n_fit_photons", "dh_fit_dx", "h_expected_rms",
          "snr", "snr_significance", "w_surface_window_final"]
# solar_elevation entra porque a janela do estudo é JJA (noite polar) e
# `cloud_flg_asr` deriva de refletância SOLAR aparente: sem sol, esse flag
# pode não ser informativo. Guardar a elevação solar permite testar isso em
# vez de aplicar o filtro às cegas.
QC_GEO = ["bsnow_conf", "bsnow_od", "bsnow_h", "cloud_flg_asr", "cloud_flg_atm",
          "msw_flag", "r_eff", "solar_elevation"]
QC_DEM = ["dem_h", "dem_flag"]

# tipos compactos — o volume total importa (centenas de milhões de linhas)
DTYPES = {
    "atl06_quality_summary": "int8", "segment_id": "int32",
    "bsnow_conf": "int8", "cloud_flg_asr": "int8", "cloud_flg_atm": "int8",
    "msw_flag": "int8", "dem_flag": "int8", "n_fit_photons": "int32",
}


def granule_datetime(name: str):
    """Extrai o instante do nome do grânulo (ATL06_YYYYMMDDHHMMSS_...)."""
    m = re.match(r"ATL06_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})_", name)
    if not m:
        return None
    y, mo, d, h, mi, s = (int(x) for x in m.groups())
    # O nome do grânulo pode trazer segundo 60 (leap second) — datetime rejeita.
    # Truncar para 59 é seguro: a busca usa uma janela de alguns segundos.
    s = min(s, 59)
    return datetime(y, mo, d, h, mi, s)


def relevant_granules(cfg, log) -> list[str]:
    """
    Grânulos que contribuem com pontos para a ROI atual.

    Fonte: o log de recorte gerado por run_consolidate, que lista os grânulos
    SEM nenhum ponto na ROI. Usá-lo evita re-baixar milhares de grânulos que
    seriam descartados de qualquer forma (4.681 processados -> 1.106 úteis).
    """
    processed = {p.stem for p in cfg.paths.processed.glob("ATL06_*.parquet")}
    log.info(f"grânulos processados em disco: {len(processed):,}")

    crop_log = cfg.paths.logs / f"recorte_roi_{cfg.season.name}.md"
    if not crop_log.exists():
        log.warning(f"{crop_log.name} ausente — sem lista de descartados, "
                    f"todos os {len(processed):,} grânulos entram.")
        return sorted(processed)

    text = crop_log.read_text(encoding="utf-8", errors="replace")
    discarded = {m.replace(".parquet", "")
                 for m in re.findall(r"ATL06_\S+?\.parquet", text)}
    keep = sorted(processed - discarded)
    log.info(f"descartados pelo recorte de ROI: {len(discarded):,} | "
             f"relevantes: {len(keep):,}")
    return keep


def extract_qc(h5_path: Path, cfg) -> pd.DataFrame:
    """
    Extrai as variáveis de qualidade aplicando a MESMA máscara de validade da
    extração original, na mesma ordem de feixes.
    """
    import h5py

    p = cfg.product
    v = p.variables
    cols = QC_ROOT + QC_FIT + QC_GEO + QC_DEM
    parts = {c: [] for c in cols}

    with h5py.File(h5_path, "r") as f:
        for beam in p.beams:
            if beam not in f or p.segments_group not in f[beam]:
                continue
            seg = f[f"{beam}/{p.segments_group}"]
            if not all(name in seg for name in v.values()):
                continue

            lat = seg[v["latitude"]][:]
            h = seg[v["height"]][:]
            qual = seg[v["quality"]][:]
            # réplica exata de thwaites/io/extract.py — qualquer divergência
            # aqui quebra o pareamento posicional
            valid = ((qual <= cfg.qc.quality_max) &
                     (h < cfg.qc.fill_value) &
                     (h > cfg.qc.h_min_valid) &
                     (~np.isnan(h)) &
                     (~np.isnan(lat)))
            if not valid.any():
                continue
            n = int(valid.sum())

            for group_name, names in (("", QC_ROOT), ("fit_statistics", QC_FIT),
                                      (p.geophysical_group, QC_GEO), ("dem", QC_DEM)):
                grp = seg if group_name == "" else (
                    seg[group_name] if group_name in seg else None)
                for c in names:
                    if grp is not None and c in grp:
                        parts[c].append(np.asarray(grp[c][:])[valid])
                    else:
                        parts[c].append(np.full(n, np.nan))

    if not parts["atl06_quality_summary"]:
        return pd.DataFrame({c: np.array([]) for c in cols})

    df = pd.DataFrame({c: np.concatenate(parts[c]) for c in cols})
    for c, dt in DTYPES.items():
        if c in df.columns and df[c].notna().all():
            df[c] = df[c].astype(dt)
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df


def main():
    ap = argparse.ArgumentParser(description="Re-extrai flags de qualidade do ATL06.")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="processa no máximo N grânulos (teste)")
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--out-dir", default="qc_flags")
    args = ap.parse_args()

    cfg = load_config(args.profile)
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level, run_name="qc_flags")

    import earthaccess
    import h5py  # noqa: F401  (falha cedo se ausente)

    out_dir = (cfg.paths.qc_flags if args.out_dir == cfg.atl06_qc.flags_dir
               else cfg.paths.data_dir / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = cfg.paths.data_dir / "raw_temp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    names = relevant_granules(cfg, log)
    todo = [n for n in names if not (out_dir / f"{n}.parquet").exists()]
    log.info(f"já extraídos: {len(names)-len(todo):,} | a fazer: {len(todo):,}")
    if args.list_only:
        return
    if args.limit:
        todo = todo[:args.limit]
        log.info(f"--limit {args.limit}: processando {len(todo)} grânulos")
    if not todo:
        log.info("nada a fazer.")
        return

    log.info("autenticando no NASA Earthdata...")
    earthaccess.login()

    stats = {"ok": 0, "mismatch": 0, "not_found": 0, "error": 0, "rows": 0}
    mismatches = []
    t_start = time.time()

    for i, name in enumerate(todo, 1):
        dt = granule_datetime(name)
        if dt is None:
            log.warning(f"[{i}/{len(todo)}] {name}: nome fora do padrão — pulado")
            stats["error"] += 1
            continue

        h5 = None
        try:
            # Janela LARGA (±15 min): o CMR indexa pelo tempo real de cobertura,
            # que se afasta do timestamp do nome de forma dependente da região
            # da órbita. Medido: grânulos de região "12" eram achados com ±30 s,
            # mas os de região "10" só a partir de ±600 s — e região 10 é 537
            # dos 1.106 grânulos relevantes. Com janela estreita, quase metade
            # do conjunto seria perdida SILENCIOSAMENTE (apenas um aviso por
            # grânulo, fácil de ignorar num log de mil linhas).
            # A janela larga é segura: o resultado é filtrado pelo nome exato.
            res = earthaccess.search_data(
                short_name=cfg.product.short_name, version=cfg.product.version,
                temporal=((dt - timedelta(seconds=900)).strftime("%Y-%m-%dT%H:%M:%S"),
                          (dt + timedelta(seconds=900)).strftime("%Y-%m-%dT%H:%M:%S")))
            target = [g for g in res if name in g["umm"]["GranuleUR"]]
            if not target:
                log.warning(f"[{i}/{len(todo)}] {name}: não encontrado no CMR")
                stats["not_found"] += 1
                continue

            h5 = Path(earthaccess.download(target, str(tmp_dir))[0])
            qc = extract_qc(h5, cfg)

            # pareamento posicional só é válido se o nº de linhas bater
            orig = cfg.paths.processed / f"{name}.parquet"
            import pyarrow.parquet as pq
            n_orig = pq.ParquetFile(orig).metadata.num_rows
            if len(qc) != n_orig:
                log.error(f"[{i}/{len(todo)}] {name}: DIVERGÊNCIA "
                          f"{len(qc):,} (QC) vs {n_orig:,} (original) — NÃO gravado")
                stats["mismatch"] += 1
                mismatches.append({"granule": name, "n_qc": int(len(qc)),
                                   "n_original": int(n_orig)})
                continue

            qc.to_parquet(out_dir / f"{name}.parquet", index=False,
                          compression="zstd")
            stats["ok"] += 1
            stats["rows"] += len(qc)

            if i % 10 == 0 or i == len(todo):
                el = time.time() - t_start
                rate = el / i
                eta = (len(todo) - i) * rate
                log.info(f"[{i}/{len(todo)}] {stats['ok']:,} ok | "
                         f"{stats['rows']:,} linhas | {rate:.1f} s/grânulo | "
                         f"ETA {eta/3600:.1f} h")
        except Exception as e:
            log.error(f"[{i}/{len(todo)}] {name}: {type(e).__name__}: {e}")
            stats["error"] += 1
        finally:
            # regra inegociável: o .h5 nunca permanece em disco
            if h5 is not None and h5.exists():
                h5.unlink()

    el = time.time() - t_start
    report = {"n_relevant": len(names), "n_attempted": len(todo), **stats,
              "elapsed_h": el / 3600,
              "mismatches": mismatches,
              "note": ("pareamento POSICIONAL com data/processed/<granulo>.parquet; "
                       "grânulos divergentes não foram gravados")}
    rp = cfg.paths.tables / "qc_flags_report.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info(f"CONCLUÍDO em {el/3600:.2f} h | ok {stats['ok']:,} | "
             f"divergentes {stats['mismatch']:,} | não encontrados "
             f"{stats['not_found']:,} | erros {stats['error']:,}")
    log.info(f"relatório -> {rp}")
    if stats["mismatch"]:
        log.warning(f"{stats['mismatch']} grânulos divergiram — investigue antes "
                    f"de usar os flags (podem indicar reprocessamento do produto).")


if __name__ == "__main__":
    main()
