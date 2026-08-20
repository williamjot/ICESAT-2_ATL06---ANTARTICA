"""
pipelines/run_chain.py
======================
Orquestrador: roda a cadeia de processamento em SEQUÊNCIA, um script por vez,
num subprocesso próprio.

Por que subprocessos separados e não imports: cada etapa devolve TODA a memória
ao sistema quando o processo morre. Numa máquina de 8 GB, rodar as etapas no
mesmo processo Python acumula fragmentação de heap e caches (rasterio, pyTMD,
pyarrow) que não são liberados — foi uma das causas dos travamentos.

Regra de parada: se uma etapa falha, a cadeia PARA. Seguir adiante produziria
resultado a partir de entrada incompleta/corrompida, sem erro visível — o modo
de falha mais perigoso deste pipeline.

Uso:
    python pipelines/run_chain.py                    # cadeia completa
    python pipelines/run_chain.py --from run_slope   # retoma de uma etapa
    python pipelines/run_chain.py --only run_dhdt run_figures
    python pipelines/run_chain.py --skip run_cats_tide
    python pipelines/run_chain.py --list
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# O console do Windows usa cp1252, que não codifica símbolos como ∇ — sem isto o
# orquestrador morre com UnicodeEncodeError ao imprimir a descrição de uma etapa.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# (script, descrição, minutos estimados, args próprios da etapa)
# As estimativas vêm de medições de referência (24,2 M) escaladas por volume;
# servem só para relatar progresso, não controlam nada.
#
CHAIN = [
    ("run_consolidate",   "consolida grânulos + filtro de qualidade ATL06",       35, []),
    ("run_mask",          "máscara BedMachine larga (remove oceano)",              5, []),
    ("run_cats_tide",     "maré CATS2008 (regional; substitui GOT4.8)",          180, []),
    ("run_corrections",   "correções geofísicas (maré + DAC) -> h_corr",          12, []),
    ("run_slope",         "correção de slope por diferenciação REMA -> h_res",    20, []),
    ("run_filttrack",     "filtro de blunders ao longo do traço (MAD local)",     12, []),
    ("run_grounded_mask", "recorte científico: gelo aterrado + buffers",           5, []),
    ("run_tiles",         "particionamento espacial em tiles",                    25, []),
    ("run_dhdt",          "ajuste espaço-temporal fitsec -> dh/dt por nó",       240, []),
    # `run_qc_report` produz `dhdt_nodes_qc.parquet`, aplicando classe do
    # BedMachine, buffers de linha de aterramento e costa e fração aterrada
    # mínima. A interpolação deve consumir esse produto validado para não incluir
    # nós sobre oceano ou gelo flutuante.
    # run_uncertainty PRECISA vir antes do run_qc_report. Ele grava o jackknife
    # de volta em `dhdt_nodes.parquet`, e é desse arquivo que o qc_report deriva
    # o `_qc`. Na ordem inversa o `_qc` sai com o erro FORMAL, que o próprio
    # projeto mediu ser ~92x otimista, propagando subestimação para a grade e
    # para as Gt/ano.
    ("run_uncertainty",   "incerteza jackknife por nó -> dhdt_nodes",              20,
     ["--name", "{run_name}_uncertainty", "--overwrite"]),
    ("run_qc_report",     "QC espacial + filtro de NÓS -> dhdt_nodes_qc",          6, []),
    ("run_interpolation", "seleção de interpolador por CV em blocos + grade",     10,
     ["--nodes", "dhdt_nodes_qc.parquet"]),
    ("run_firn",          "correção de ar no firn (FAC) -> dh de gelo",            5,
     ["--name", "{run_name}_firn", "--overwrite"]),
    ("run_mass_balance",  "dh de gelo -> Gt/ano -> nível do mar",                  3,
     ["--grid", "{outputs}/experiments/{run_name}_firn/firn_corrected_grid.parquet",
      "--value-col", "dhdt_ice", "--name", "{run_name}_mass_firn", "--overwrite"]),
    ("run_xover",         "checagem de consistência em cruzamentos",              15, []),
    ("run_flux",          "divergência de fluxo ∇·(H·v) -> derretimento basal",   10, []),
    ("run_dynamics",      "razão de partição / diagnóstico dinâmico",              5,
     ["--name", "{run_name}_dynamics", "--overwrite"]),
    ("run_advection",     "dh/dt lagrangiano (termo de advecção v·∇h)",            8,
     ["--name", "{run_name}_advection", "--overwrite"]),
    ("run_figures",       "mapas, histogramas e diagramas",                        4,
     ["--experiment", "{run_name}_firn", "--mass-experiment", "{run_name}_mass_firn"]),

    # ---------------------------------------------------------------- PLATAFORMA
    # Linha de produto SEPARADA (gelo flutuante), com máscara própria e
    # referencial lagrangiano. Estava inteiramente FORA da cadeia e foi rodada à
    # mão só para o JJA — quando o DJF terminou, não tinha nenhum produto de
    # plataforma, e a comparação sazonal ficaria impossível justamente onde o
    # derretimento basal acontece. Mesmo padrão do run_qc_report: etapa
    # essencial que só existia na memória de quem rodou.
    ("run_shelf_mask",    "máscara de plataforma (flutuante) -> atl06_shelf",       6, []),
    ("run_shelf_lagrangian", "trajetórias de parcelas (RK4, ITS_LIVE)",            15, []),
    ("run_shelf_windows", "janelas móveis lagrangianas por parcela",               15, []),
    ("run_shelf_divergence", "H·∇·v por parcela",                                   8, []),
    ("run_basal_melt",    "ṁ_b = a_s − DH/Dt − H·∇·v (Adusumilli)",                 3, []),
]


def run_step(script: str, extra: list[str], log_dir: Path) -> tuple[bool, float]:
    """Roda uma etapa num subprocesso. Devolve (sucesso, minutos)."""
    path = ROOT / "pipelines" / f"{script}.py"
    if not path.exists():
        print(f"  !! {script}.py não existe — etapa ignorada.", flush=True)
        return True, 0.0

    out_file = log_dir / f"chain_{script}.out"
    t0 = time.time()
    with open(out_file, "w", encoding="utf-8", errors="replace") as fh:
        proc = subprocess.run(
            [sys.executable, str(path), *extra],
            stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT),
        )
    mins = (time.time() - t0) / 60

    if proc.returncode != 0:
        print(f"  FALHOU (código {proc.returncode}) após {mins:.1f} min", flush=True)
        tail = out_file.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
        print("  --- fim do log ---", flush=True)
        for line in tail:
            print(f"  | {line}", flush=True)
        return False, mins

    print(f"  OK em {mins:.1f} min  (log: {out_file.name})", flush=True)
    return True, mins


def main():
    ap = argparse.ArgumentParser(description="Roda a cadeia de processamento em sequência.")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--from", dest="start", default=None, help="retoma desta etapa")
    ap.add_argument("--only", nargs="+", default=None, help="roda só estas etapas")
    ap.add_argument("--skip", nargs="+", default=[], help="pula estas etapas")
    ap.add_argument("--name", default=None,
                    help="prefixo único dos experimentos versionados (default: perfil + UTC)")
    ap.add_argument("--list", action="store_true", help="lista as etapas e sai")
    args = ap.parse_args()

    run_name = args.name or f"{args.profile or 'jja'}_{datetime.utcnow():%Y%m%dT%H%M%SZ}"

    if args.list:
        total = sum(m for _, _, m, _ in CHAIN)
        for i, (s, d, m, a) in enumerate(CHAIN, 1):
            tag = f"  [{' '.join(a)}]" if a else ""
            print(f"{i:2d}. {s:<20} ~{m:>3d} min   {d}{tag}")
        print(f"\nTotal estimado: {total/60:.1f} h")
        return

    steps = CHAIN
    if args.only:
        steps = [s for s in CHAIN if s[0] in set(args.only)]
    elif args.start:
        names = [s[0] for s in CHAIN]
        if args.start not in names:
            raise SystemExit(f"Etapa desconhecida: {args.start}. Use --list.")
        steps = CHAIN[names.index(args.start):]
    steps = [s for s in steps if s[0] not in set(args.skip)]

    # Diretório de saídas RELATIVO à raiz, resolvido pela config — porque ele
    # depende da estação (`outputs/jja`, `outputs/djf`, ...). Caminhos fixos como
    # `outputs/experiments/...` ignorariam o isolamento por estação e fariam a
    # cadeia falhar ao localizar o grid.
    sys.path.insert(0, str(ROOT))
    from thwaites import load_config
    outputs_rel = load_config(args.profile).paths.outputs_dir.relative_to(ROOT)
    outputs_rel = outputs_rel.as_posix()

    extra = ["--profile", args.profile] if args.profile else []
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    est_total = sum(m for _, _, m, _ in steps)
    print("=" * 74, flush=True)
    print(f"CADEIA: {len(steps)} etapas | início {datetime.now():%H:%M:%S} | "
          f"estimativa {est_total/60:.1f} h "
          f"(fim ~{datetime.now() + timedelta(minutes=est_total):%H:%M})", flush=True)
    print("=" * 74, flush=True)

    elapsed = 0.0
    for i, (script, desc, est, own) in enumerate(steps, 1):
        print(f"\n[{i}/{len(steps)}] {script}  ({desc})", flush=True)
        print(f"  início {datetime.now():%H:%M:%S} | estimado ~{est} min", flush=True)
        own_resolved = [v.format(run_name=run_name, outputs=outputs_rel)
                        for v in own]
        ok, mins = run_step(script, [*extra, *own_resolved], log_dir)
        elapsed += mins
        if not ok:
            print(f"\nCADEIA INTERROMPIDA em {script} após {elapsed/60:.1f} h.", flush=True)
            print(f"Corrija e retome com: "
                  f"python pipelines/run_chain.py --from {script}", flush=True)
            raise SystemExit(1)

    print("\n" + "=" * 74, flush=True)
    print(f"CADEIA COMPLETA em {elapsed/60:.1f} h | fim {datetime.now():%H:%M:%S}", flush=True)
    print("=" * 74, flush=True)


if __name__ == "__main__":
    main()
