"""
pipelines/run_dhdt_janelas.py
=============================
dh/dt em JANELAS MÓVEIS — a evolução espacial da taxa ao longo do registro.

    data/<estação>/tiles/*.parquet
        -> data/<estação>/dhdt/janelas/dhdt_<ini>_<fim>.parquet
        -> outputs/<estação>/tables/dhdt_janelas.json

Por que janelas e não a aceleração
-----------------------------------
`run_acceleration` responde "há evidência estatística de curvatura?" e, no
nosso registro, responde NÃO em 99,6% dos nós — o resíduo é autocorrelado e o
critério recusa reportar. Isso é correto, mas não é a mesma pergunta que
"a taxa mudou de LUGAR?". Uma janela móvel não testa hipótese nenhuma: ela
mostra o campo em cada trecho do período e deixa a comparação para os olhos.
São produtos complementares, e a janela é honesta justamente por não afirmar
significância.

Desenho da janela
-----------------
Quatro anos de largura, passo de um ano. A largura não é livre: `dhdt.min_years`
exige 3 anos de vão, e com JJA todas as observações de um ano caem numa faixa de
89 dias — uma janela de 3 anos daria vão de exatamente 3,0 no melhor caso e
falharia na maioria dos nós. Quatro anos dá folga sem achatar o registro de sete.

Leitura obrigatória do resultado
--------------------------------
Janelas vizinhas COMPARTILHAM três dos quatro anos. As diferenças entre mapas
consecutivos são, portanto, fortemente correlacionadas — não são amostras
independentes, e ler "tendência da tendência" a partir delas superestima
qualquer mudança. A janela mostra onde o campo se move; quantificar a mudança
exige o teste de aceleração.

Uso: python pipelines/run_dhdt_janelas.py --profile djf
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from thwaites import load_config
from thwaites.logging import setup_logging
from thwaites.grid.tiles import load_manifest
from thwaites.timeseries.dhdt import compute_tile_dhdt_windows
from thwaites.io.memory import free_memory_gb


COLS = ["x", "y", "lon", "lat", "t_year", "h_res", "h_corr", "h_elv", "s_elv"]
_WORKER_CFG = None
_WORKER_WINDOWS = None
_WORKER_ACCEPTED = None
_WORKER_THREAD_LIMIT = None


def _init_worker(profile, windows):
    """Inicializa a configuração uma única vez em cada processo."""
    global _WORKER_CFG, _WORKER_WINDOWS, _WORKER_ACCEPTED, _WORKER_THREAD_LIMIT
    from threadpoolctl import threadpool_limits

    _WORKER_THREAD_LIMIT = threadpool_limits(limits=1)
    _WORKER_CFG = load_config(profile)
    _WORKER_WINDOWS = windows
    qc = pd.read_parquet(
        _WORKER_CFG.paths.dhdt_dir / "dhdt_nodes_qc.parquet", columns=["x", "y"]
    )
    _WORKER_ACCEPTED = set(
        zip(
            np.rint(qc["x"]).astype(np.int64),
            np.rint(qc["y"]).astype(np.int64),
        )
    )
    setup_logging(
        _WORKER_CFG.paths.logs,
        level="ERROR",
        run_name=f"dhdt_janelas_worker_{os.getpid()}",
    )


def _process_tile(entry):
    """Calcula todas as janelas de um bloco após uma única leitura."""
    import pyarrow.parquet as pq

    cfg = _WORKER_CFG
    windows = _WORKER_WINDOWS
    path = cfg.paths.tiles_dir / entry["file"]
    available = pq.ParquetFile(path).schema_arrow.names
    tile = pd.read_parquet(
        path, columns=[column for column in COLS if column in available], engine="pyarrow"
    )
    calculated = compute_tile_dhdt_windows(
        tile,
        cfg,
        entry["x_min"],
        entry["x_max"],
        entry["y_min"],
        entry["y_max"],
        windows,
        accepted_nodes=_WORKER_ACCEPTED,
    )
    results = {window: nodes for window, nodes in calculated.items() if len(nodes)}
    return entry["tile"], results


def janelas(largura: float, passo: float, t0: float, t1: float):
    """Janelas [ini, ini+largura) que cabem inteiras no registro."""
    out = []
    ini = np.floor(t0)
    while ini + largura <= np.ceil(t1) + 1e-9:
        out.append((float(ini), float(ini + largura)))
        ini += passo
    return out


def janelas_inicio_fixo(inicio: float, largura_minima: float, t1: float):
    """Janelas expansivas [inicio, fim), acrescentando um ano por quadro."""
    fim = float(inicio + largura_minima)
    limite = float(np.ceil(t1))
    out = []
    while fim <= limite + 1e-9:
        out.append((float(inicio), fim))
        fim += 1.0
    return out


def main():
    ap = argparse.ArgumentParser(description="dh/dt em janelas móveis.")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--largura", type=float, default=4.0)
    ap.add_argument("--passo", type=float, default=1.0)
    ap.add_argument(
        "--inicio-fixo",
        type=float,
        default=None,
        help=("mantém o início fixo e expande o fim em um ano por quadro; "
              "ex.: --inicio-fixo 2019"),
    )
    ap.add_argument("--max-tiles", type=int, default=None)
    ap.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="processos paralelos por bloco espacial (padrão: até 4)",
    )
    args = ap.parse_args()

    cfg = load_config(args.profile)
    est = cfg.season.name
    modo = "inicio_fixo" if args.inicio_fixo is not None else "moveis"
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level,
                        run_name=f"dhdt_janelas_{modo}_{est}")

    subdir = "janelas_inicio_fixo" if args.inicio_fixo is not None else "janelas"
    dst = cfg.paths.dhdt_dir / subdir
    dst.mkdir(parents=True, exist_ok=True)

    entradas = load_manifest(cfg)
    if args.max_tiles:
        entradas = entradas[:args.max_tiles]

    # extensão temporal real, de uma amostra de tiles
    tt = []
    for e in entradas[::max(len(entradas) // 8, 1)]:
        v = pd.read_parquet(cfg.paths.tiles_dir / e["file"],
                            columns=["t_year"])["t_year"].to_numpy(float)
        tt.append((np.nanmin(v), np.nanmax(v)))
    t0 = min(a for a, _ in tt)
    t1 = max(b for _, b in tt)
    if args.inicio_fixo is None:
        js = janelas(args.largura, args.passo, t0, t1)
        desenho = (f"{len(js)} janelas móveis de {args.largura:.0f} anos, "
                   f"passo {args.passo:.0f}")
    else:
        js = janelas_inicio_fixo(args.inicio_fixo, args.largura, t1)
        desenho = (f"{len(js)} janelas expansivas, início fixo em "
                   f"{args.inicio_fixo:.0f}, vão mínimo {args.largura:.0f} anos")
    reusable = {}
    js_compute = list(js)
    if args.inicio_fixo == 2019.0 and args.largura == 4.0:
        first = cfg.paths.dhdt_dir / "janelas" / "dhdt_2019_2023.parquet"
        full = cfg.paths.dhdt_dir / "dhdt_nodes.parquet"
        if first.exists() and (2019.0, 2023.0) in js_compute:
            reusable[(2019.0, 2023.0)] = first
            js_compute.remove((2019.0, 2023.0))
        if full.exists() and (2019.0, 2026.0) in js_compute:
            reusable[(2019.0, 2026.0)] = full
            js_compute.remove((2019.0, 2026.0))
    log.info(f"[{est}] registro {t0:.2f}–{t1:.2f} | {desenho}")
    for a, b in js:
        log.info(f"    {a:.0f}–{b:.0f}")
    if reusable:
        log.info(
            "Reutilização exata de produtos já calculados: "
            + ", ".join(f"{a:.0f}–{b:.0f}" for a, b in reusable)
        )

    cache_tag = (
        f"l{args.largura:g}_p{args.passo:g}_"
        f"i{args.inicio_fixo:g}" if args.inicio_fixo is not None
        else f"l{args.largura:g}_p{args.passo:g}_moveis"
    ).replace(".", "p")
    parts_root = dst / f"_partes_{cache_tag}"
    markers_dir = parts_root / "_concluidos"
    markers_dir.mkdir(parents=True, exist_ok=True)
    for start, end in js:
        (parts_root / f"{start:.0f}_{end:.0f}").mkdir(parents=True, exist_ok=True)

    completed = {path.stem for path in markers_dir.glob("*.ok")}
    pending = [entry for entry in entradas if entry["tile"] not in completed]
    workers = max(1, min(args.workers, len(entradas)))
    log.info(
        f"Processamento por bloco com {workers} processo(s) | "
        f"{len(completed)} já concluídos, {len(pending)} pendentes"
    )

    def persist(tile_name, results):
        for (start, end), nodes in results.items():
            part = parts_root / f"{start:.0f}_{end:.0f}" / f"{tile_name}.parquet"
            nodes.to_parquet(part, index=False, engine="pyarrow", compression="snappy")
        (markers_dir / f"{tile_name}.ok").write_text("ok\n", encoding="ascii")

    if pending and workers == 1:
        _init_worker(args.profile, js_compute)
        for i, entry in enumerate(pending, 1):
            tile_name, results = _process_tile(entry)
            persist(tile_name, results)
            if i % 10 == 0 or i == len(pending):
                log.info(f"  [{i}/{len(pending)} novos] {tile_name}")
    elif pending:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(args.profile, js_compute),
        ) as executor:
            futures = {executor.submit(_process_tile, entry): entry for entry in pending}
            for i, future in enumerate(as_completed(futures), 1):
                tile_name, results = future.result()
                persist(tile_name, results)
                if i % 10 == 0 or i == len(pending):
                    log.info(
                        f"  [{i}/{len(pending)} novos] {tile_name} | "
                        f"livre {free_memory_gb():.1f} GB"
                    )

    rel = {"estacao": est, "modo": modo,
           "largura_minima_anos": args.largura, "passo_anos": args.passo,
           "inicio_fixo": args.inicio_fixo,
           "registro": [t0, t1], "janelas": {}}
    qc = pd.read_parquet(
        cfg.paths.dhdt_dir / "dhdt_nodes_qc.parquet", columns=["x", "y"]
    )
    accepted_final = set(
        zip(
            np.rint(qc["x"]).astype(np.int64),
            np.rint(qc["y"]).astype(np.int64),
        )
    )
    for a, b in js:
        if (a, b) in reusable:
            nodes = pd.read_parquet(reusable[(a, b)])
        else:
            part_paths = sorted(
                (parts_root / f"{a:.0f}_{b:.0f}").glob("*.parquet")
            )
            if not part_paths:
                log.warning(f"janela {a:.0f}–{b:.0f} sem nós")
                continue
            nodes = pd.concat(
                (pd.read_parquet(path) for path in part_paths), ignore_index=True
            )
        keep = np.fromiter(
            (
                (x, y) in accepted_final
                for x, y in zip(
                    np.rint(nodes["x"]).astype(np.int64),
                    np.rint(nodes["y"]).astype(np.int64),
                )
            ),
            dtype=bool,
            count=len(nodes),
        )
        nodes = nodes.loc[keep].copy()
        f = dst / f"dhdt_{a:.0f}_{b:.0f}.parquet"
        nodes.to_parquet(f, index=False, engine="pyarrow", compression="snappy")
        v = nodes["dhdt"].to_numpy(float)
        v = v[np.isfinite(v)]
        rel["janelas"][f"{a:.0f}-{b:.0f}"] = {
            "n_nos": int(len(nodes)),
            "dhdt_mediana": float(np.median(v)),
            "dhdt_p10": float(np.percentile(v, 10)),
            "dhdt_p90": float(np.percentile(v, 90)),
            "arquivo": f.name,
        }
        log.info(f"  {a:.0f}–{b:.0f}: {len(nodes):,} nós | "
                 f"mediana {np.median(v):+.4f} m/ano")

    if args.inicio_fixo is None:
        rel["nota"] = ("Janelas vizinhas compartilham 3 dos 4 anos: as diferenças "
                       "entre mapas consecutivos são fortemente correlacionadas e "
                       "NÃO são amostras independentes. Para afirmar mudança de "
                       "taxa, use run_acceleration.")
        report_name = "dhdt_janelas.json"
    else:
        rel["nota"] = ("Janelas expansivas mantêm o início fixo e reutilizam todas "
                       "as observações anteriores; quadros consecutivos são "
                       "fortemente dependentes e não quantificam aceleração.")
        report_name = "dhdt_janelas_inicio_fixo.json"
    rp = cfg.paths.tables / report_name
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(rel, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Relatório -> {rp}")


if __name__ == "__main__":
    main()
