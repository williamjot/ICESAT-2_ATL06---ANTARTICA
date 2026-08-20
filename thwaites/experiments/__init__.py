"""
Infraestrutura de experimentos reprodutíveis (§8 do PLANO_PRIORIDADES_CIENTIFICAS).

Cada execução gera um manifesto rastreável e grava em diretório próprio, de modo
que um experimento nunca sobrescreve silenciosamente os produtos de outro.
"""

from thwaites.experiments.manifest import (
    Manifest, file_checksum, config_hash, git_commit, experiment_dir, source_tree_hash,
)

__all__ = ["Manifest", "file_checksum", "config_hash", "git_commit", "experiment_dir",
           "source_tree_hash"]
