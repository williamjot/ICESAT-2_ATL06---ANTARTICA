"""
pipelines/run_cats_tide.py
==========================
Substitui a maré oceânica do ATL06 (GOT4.8, global) pelo CATS2008 (regional
antártico, resolve cavidades sob plataformas) usando pyTMD.

    data/interim/atl06_masked.parquet
        -> data/interim/atl06_masked_cats.parquet  (tide_ocean = CATS,
                                                    tide_ocean_got = original)
        -> outputs/tables/cats_vs_got.json

MEMÓRIA: tudo em streaming — o modelo é aberto e recortado à ROI UMA vez, e o
Parquet é lido/gravado em row groups. Nada é carregado inteiro na RAM.

Rode ENTRE run_mask.py e run_corrections.py (precisa de mask_class para o
gating de gelo flutuante; e as correções consomem tide_ocean).

PRÉ-REQUISITO MANUAL: CATS2008_v2023.nc em
data/tide_models/CATS2008_v2023/ (USAP-DC dataset 601772 — o download exige
reCAPTCHA e não pode ser automatizado).

Uso: python pipelines/run_cats_tide.py [--profile anual]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config
from thwaites.logging import setup_logging
from thwaites.corrections.cats_tide import (
    apply_cats_tide_streaming, resolve_model_dir, resolve_model_file,
)


def main():
    ap = argparse.ArgumentParser(description="Maré CATS2008 via pyTMD (streaming).")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--input", default="atl06_masked.parquet",
                    help="arquivo de entrada em data/interim/")
    ap.add_argument("--output", default="atl06_masked_cats.parquet")
    args = ap.parse_args()

    cfg = load_config(args.profile)
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level, run_name="cats_tide")

    if not cfg.cats.enabled:
        log.warning("cats.enabled=false na config — nada a fazer. "
                    "Baixe o modelo e ative para usar o CATS2008.")
        return

    model_path = resolve_model_file(cfg)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {model_path}\n"
            f"O pyTMD espera o .nc num subdiretório com o nome do modelo:\n"
            f"  {resolve_model_dir(cfg)}\\{Path(cfg.cats.model_file).stem}\\{cfg.cats.model_file}\n"
            f"Baixe 'CATS2008_v2023.nc' (1,8 GB) em "
            f"https://www.usap-dc.org/view/dataset/601772 (exige reCAPTCHA).")

    src = cfg.paths.interim / args.input
    dst = cfg.paths.interim / args.output
    if not src.exists():
        raise FileNotFoundError(f"{src} não existe (rode run_mask.py).")

    log.info(f"CATS streaming: {src.name} -> {dst.name} "
             f"(row_batch={cfg.cats.row_batch:,}, chunk={cfg.cats.chunk_size:,})")
    summary = apply_cats_tide_streaming(src, dst, cfg)

    cfg.paths.tables.mkdir(parents=True, exist_ok=True)
    (cfg.paths.tables / "cats_vs_got.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info(f"Resumo -> {cfg.paths.tables / 'cats_vs_got.json'}")
    log.info("Próximo: run_corrections.py --input atl06_masked_cats.parquet")


if __name__ == "__main__":
    main()
