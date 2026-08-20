"""
thwaites.validation.folds
=========================
Geração de folds RASTREÁVEIS sem vazamento de observações (§5.3).

O PROBLEMA (§5.1): a validação cruzada por NÓS separa nós já calculados, mas
nós vizinhos podem ter sido estimados com as MESMAS observações ATL06, porque
os raios de busca se sobrepõem. Um nó de treino e um de teste então compartilham
dados de origem, e o erro de validação parece menor do que o erro real de
generalização.

A SOLUÇÃO: particionar no nível da OBSERVAÇÃO e recalcular os nós em cada fold
usando somente observações de treino. Três estratégias, que medem capacidades
DIFERENTES e devem ser reportadas separadamente (§5.3):

  A. **espacial com buffer** — generalizar para regiões não amostradas;
  B. **por trilha** — generalizar para trilhas não usadas no ajuste;
  C. **temporal** — generalizar para épocas não usadas.

O BUFFER (A) não é decorativo: sem ele, observações de treino coladas na borda
do bloco de teste tornam a previsão artificialmente fácil (autocorrelação
espacial). O buffer precisa ser ≥ raio de busca — senão um nó ajustado logo
fora do bloco de teste usaria observações de dentro dele.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from thwaites.logging import get_logger


@dataclass
class Fold:
    """
    Uma partição treino/teste, com metadados para o manifesto (§5.6).

    `train` e `test` são máscaras booleanas sobre o array de observações.
    """
    strategy: str
    index: int
    train: np.ndarray = field(repr=False)
    test: np.ndarray = field(repr=False)
    info: dict = field(default_factory=dict)

    @property
    def n_train(self) -> int:
        return int(self.train.sum())

    @property
    def n_test(self) -> int:
        return int(self.test.sum())

    def summary(self) -> dict:
        return {"strategy": self.strategy, "fold": self.index,
                "n_train": self.n_train, "n_test": self.n_test, **self.info}


def verify_no_leakage(fold: Fold) -> bool:
    """
    Garante que NENHUMA observação está em treino e teste ao mesmo tempo (§5.5).

    É uma verificação barata e absoluta — deve ser chamada em todo fold antes de
    qualquer ajuste.
    """
    overlap = int(np.sum(fold.train & fold.test))
    if overlap:
        raise AssertionError(
            f"vazamento: {overlap} observações em treino E teste "
            f"(estratégia {fold.strategy}, fold {fold.index})")
    return True


# ------------------------------------------------------------------ A: espacial
def spatial_buffer_folds(x, y, block_m: float, n_folds: int, buffer_m: float,
                         seed: int = 0) -> list[Fold]:
    """
    Blocos espaciais inteiros como teste, com ZONA TAMPÃO excluída do treino.

    Observações no buffer não entram nem no treino nem no teste — é o preço de
    medir generalização honesta para regiões não amostradas.
    """
    logger = get_logger()
    x = np.asarray(x, float); y = np.asarray(y, float)
    bi = np.floor(x / block_m).astype(np.int64)
    bj = np.floor(y / block_m).astype(np.int64)
    keys = bi * 100003 + bj
    uniq = np.unique(keys)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    assign = {b: i % n_folds for i, b in enumerate(uniq)}
    block_fold = np.array([assign[k] for k in keys])

    folds = []
    for f in range(n_folds):
        test = block_fold == f
        if not test.any():
            continue
        # buffer: distância aos blocos de teste, medida no espaço das bordas.
        # Uma observação de treino é descartada se estiver a menos de buffer_m
        # de QUALQUER observação de teste.
        from scipy.spatial import cKDTree
        tree = cKDTree(np.c_[x[test], y[test]])
        cand = ~test
        d, _ = tree.query(np.c_[x[cand], y[cand]], k=1)
        near = np.zeros(len(x), dtype=bool)
        near[np.flatnonzero(cand)] = d < buffer_m
        train = cand & ~near
        folds.append(Fold("spatial_buffer", f, train, test, {
            "block_m": block_m, "buffer_m": buffer_m,
            "n_buffer_excluded": int(near.sum()),
            "n_blocks_test": int(np.unique(keys[test]).size),
            "seed": seed,
        }))
    logger.info(f"folds espaciais: {len(folds)} (bloco {block_m/1000:.0f} km, "
                f"buffer {buffer_m/1000:.0f} km)")
    return folds


# --------------------------------------------------------------- B: por trilha
def track_folds(track_id, n_folds: int, seed: int = 0) -> list[Fold]:
    """
    Retém TRILHAS inteiras fora do ajuste (§5.3.B).

    Mede se o modelo generaliza para trilhas que não viu — um teste diferente do
    espacial, porque uma trilha retida atravessa regiões que o treino cobre.
    """
    logger = get_logger()
    tid = np.asarray(track_id)
    uniq = np.unique(tid)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    assign = {t: i % n_folds for i, t in enumerate(uniq)}
    tfold = np.array([assign[t] for t in tid])

    folds = []
    for f in range(n_folds):
        test = tfold == f
        if not test.any():
            continue
        folds.append(Fold("track", f, ~test, test, {
            "n_tracks_test": int(np.unique(tid[test]).size),
            "n_tracks_train": int(np.unique(tid[~test]).size),
            "seed": seed,
        }))
    logger.info(f"folds por trilha: {len(folds)} ({uniq.size} trilhas no total)")
    return folds


# ---------------------------------------------------------------- C: temporal
def temporal_folds(t_year, mode: str = "leave_one_year") -> list[Fold]:
    """
    Retém ANOS inteiros (§5.3.C). Testa generalização para épocas não vistas.
    """
    logger = get_logger()
    t = np.asarray(t_year, float)
    years = np.unique(np.floor(t))
    folds = []
    for i, y in enumerate(years):
        test = np.floor(t) == y
        if not test.any():
            continue
        folds.append(Fold("temporal", i, ~test, test,
                          {"year_held_out": float(y), "mode": mode}))
    logger.info(f"folds temporais: {len(folds)} anos "
                f"({years.min():.0f}–{years.max():.0f})")
    return folds


def default_buffer_m(cfg, variogram_range_m: float | None = None) -> float:
    """
    Buffer padrão (§5.3.A): precisa ser ≥ raio de busca, senão um nó ajustado
    fora do bloco de teste usaria observações de dentro dele. Se o alcance do
    variograma for conhecido e maior, ele domina (autocorrelação espacial).
    """
    b = float(cfg.dhdt.search_radius_m)
    if variogram_range_m and np.isfinite(variogram_range_m):
        b = max(b, float(variogram_range_m))
    return b
