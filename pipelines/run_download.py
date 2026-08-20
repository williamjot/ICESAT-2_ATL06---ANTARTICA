"""
pipelines/run_download.py
=========================
Orquestração fina do passo de download+extração. Não contém lógica — só
carrega a config, inicia o logging e chama thwaites.io.download.run_download.

Uso:
    python pipelines/run_download.py                # perfil base (default.yaml)
    python pipelines/run_download.py --profile anual
"""

import argparse
import sys
from pathlib import Path

# Permite rodar sem `pip install -e .` (adiciona a raiz do projeto ao path).
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config
from thwaites.logging import setup_logging
from thwaites.io.download import run_download


def main():
    ap = argparse.ArgumentParser(description="Download+extração ATL06 (grânulo a grânulo).")
    ap.add_argument("--profile", default=None, help="perfil de config (ex.: jja, anual)")
    args = ap.parse_args()

    cfg = load_config(args.profile)
    setup_logging(cfg.paths.logs, level=cfg.logging.level,
                  rotation=cfg.logging.rotation, retention=cfg.logging.retention,
                  run_name=f"download_{cfg.season.name}")
    run_download(cfg)


if __name__ == "__main__":
    main()
