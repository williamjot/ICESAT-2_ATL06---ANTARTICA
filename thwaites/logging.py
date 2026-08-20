"""
thwaites.logging
================
Logging estruturado com loguru: um sink no stderr e um sink auditável em arquivo
por execução, com rotação e registros rastreáveis.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

_CONFIGURED = False


def setup_logging(
    log_dir: Path,
    level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "30 days",
    run_name: str | None = None,
):
    """
    Configura o logger global (idempotente).

    Parâmetros
    ----------
    log_dir : Path
        Pasta onde gravar o arquivo de log.
    level : str
        Nível mínimo (DEBUG, INFO, WARNING, ...).
    rotation, retention : str
        Política de rotação/retenção do loguru.
    run_name : str | None
        Sufixo do arquivo de log; default = timestamp.

    Retorna
    -------
    O logger configurado.
    """
    global _CONFIGURED

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    stamp = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"thwaites_{stamp}.log"

    logger.remove()  # limpa handlers default
    logger.add(
        sys.stderr, level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | "
               "<cyan>{name}</cyan> - {message}",
    )
    logger.add(
        log_file, level=level, rotation=rotation, retention=retention,
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {name}:{function}:{line} - {message}",
    )
    _CONFIGURED = True
    logger.info(f"Logging iniciado -> {log_file}")
    return logger


def get_logger():
    """Retorna o logger (avisa se setup_logging ainda não foi chamado)."""
    if not _CONFIGURED:
        logger.warning("setup_logging() ainda não chamado — usando defaults do loguru.")
    return logger
