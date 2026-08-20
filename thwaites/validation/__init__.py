"""
Validação sem vazamento de observações (Prioridade 4, §5 do PLANO).

Diferente da validação por nós — que deixa treino e teste compartilharem as
mesmas observações ATL06 pelos raios de busca sobrepostos — aqui a partição é
feita no nível da OBSERVAÇÃO, e os nós são recalculados em cada fold.
"""

from thwaites.validation.folds import (
    Fold, spatial_buffer_folds, track_folds, temporal_folds, verify_no_leakage,
)
from thwaites.validation.evaluate import (
    evaluate_fold, run_validation, fold_metrics,
)
from thwaites.validation.agreement import (
    assess_agreement, match_xovers_to_nodes, paired_differences,
    robust_regression, find_hotspots, subsampling_sensitivity,
    INDEPENDENCE_CAVEAT,
)

__all__ = ["Fold", "spatial_buffer_folds", "track_folds", "temporal_folds",
           "verify_no_leakage", "evaluate_fold", "run_validation", "fold_metrics",
           "assess_agreement", "match_xovers_to_nodes", "paired_differences",
           "robust_regression", "find_hotspots", "subsampling_sensitivity",
           "INDEPENDENCE_CAVEAT"]
