"""
Máscara de frente de calving dependente da época (IceLines/Sentinel-1).

As linhas são tratadas separadamente por plataforma. Isso é obrigatório:
escolher primeiro a época globalmente e só depois a linha mais próxima mistura
plataformas com calendários de aquisição diferentes e pode comparar uma
observação da Thwaites com uma frente de outra plataforma.

Como o IceLines fornece linhas abertas, o lado da frente é aproximado pela
distância ao gelo aterrado do BedMachine. Uma posição é aceita quando sua
distância ao aterrado não ultrapassa a da frente local, descontado um buffer de
segurança. O critério e suas falhas ficam explícitos nas saídas: plataforma
atribuída, época usada, defasagem temporal, distância à linha e margem até a
fronteira dinâmica.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from thwaites.logging import get_logger


def densify_fronts(fronts: pd.DataFrame, step_m: float = 500.0) -> pd.DataFrame:
    """Converte linhas WKT em pontos separados por no máximo ``step_m``."""
    from shapely import wkt
    from shapely.geometry import LineString, MultiLineString

    required = {"shelf", "epoch_year", "wkt"}
    missing = required.difference(fronts.columns)
    if missing:
        raise ValueError(f"frentes sem colunas obrigatórias: {sorted(missing)}")
    if not np.isfinite(step_m) or step_m <= 0:
        raise ValueError("step_m deve ser positivo e finito.")

    rows: list[tuple[str, float, float, float]] = []
    for r in fronts.itertuples():
        try:
            geom = wkt.loads(r.wkt)
        except Exception:
            continue
        lines = (list(geom.geoms) if isinstance(geom, MultiLineString)
                 else [geom] if isinstance(geom, LineString) else [])
        for line in lines:
            if not np.isfinite(line.length) or line.length <= 0:
                continue
            n_segments = max(1, int(np.ceil(line.length / step_m)))
            for distance in np.linspace(0.0, line.length, n_segments + 1):
                point = line.interpolate(distance)
                rows.append((str(r.shelf), float(r.epoch_year),
                             float(point.x), float(point.y)))
    return pd.DataFrame(rows, columns=["shelf", "epoch_year", "x", "y"])


@dataclass(frozen=True)
class FrontMaskResult:
    """Resultado vetorial, incluindo motivos para pontos não testados."""

    inside: np.ndarray
    tested: np.ndarray
    assigned: np.ndarray
    epoch_valid: np.ndarray
    near_front: np.ndarray
    grounded_distance_valid: np.ndarray
    shelf: np.ndarray
    front_epoch: np.ndarray
    epoch_gap_years: np.ndarray
    front_distance_m: np.ndarray
    margin_m: np.ndarray


class FrontMask:
    """Classifica posições contra a frente da plataforma e época corretas."""

    def __init__(
        self,
        fronts: pd.DataFrame,
        sx,
        sy,
        dist_grounded_field,
        step_m: float = 500.0,
        reference_step_m: float | None = None,
        densified_points: pd.DataFrame | None = None,
    ):
        from scipy.spatial import cKDTree

        logger = get_logger()
        points = (densified_points.copy() if densified_points is not None
                  else densify_fronts(fronts, step_m=step_m))
        point_columns = {"shelf", "epoch_year", "x", "y"}
        if not point_columns.issubset(points.columns):
            raise ValueError("cache densificado sem shelf/epoch_year/x/y.")
        if points.empty:
            raise ValueError("nenhuma frente pôde ser densificada.")

        self.sx = np.asarray(sx, dtype=float)
        self.sy = np.asarray(sy, dtype=float)
        self.DG = np.asarray(dist_grounded_field, dtype=float)
        if self.DG.shape != (len(self.sy), len(self.sx)):
            raise ValueError("dist_grounded_field incompatível com sx/sy.")
        if len(self.sx) < 2 or len(self.sy) < 2:
            raise ValueError("sx e sy precisam ter ao menos dois elementos.")
        self.dx = float(self.sx[1] - self.sx[0])
        self.dy = float(self.sy[1] - self.sy[0])
        if self.dx <= 0 or self.dy <= 0:
            raise ValueError("sx e sy devem ser estritamente crescentes.")

        points["dist_grounded"] = self._sample(
            points["x"].to_numpy(), points["y"].to_numpy())
        points = points[np.isfinite(points["dist_grounded"])].copy()
        if points.empty:
            raise ValueError("nenhuma frente intersecta o campo BedMachine.")

        self.shelves = np.array(sorted(points["shelf"].unique()), dtype=object)
        self.epochs_by_shelf: dict[str, np.ndarray] = {}
        self._trees: dict[tuple[str, float], object] = {}
        self._dg: dict[tuple[str, float], np.ndarray] = {}
        for shelf, group in points.groupby("shelf", sort=True):
            epochs = np.array(sorted(group["epoch_year"].unique()), dtype=float)
            self.epochs_by_shelf[str(shelf)] = epochs
            for epoch, epoch_points in group.groupby("epoch_year", sort=True):
                key = (str(shelf), float(epoch))
                self._trees[key] = cKDTree(
                    epoch_points[["x", "y"]].to_numpy(dtype=float))
                self._dg[key] = epoch_points["dist_grounded"].to_numpy(dtype=float)

        # Árvore apenas para atribuir a plataforma. A quantização remove as
        # muitas linhas mensais quase coincidentes, sem alterar as árvores
        # usadas no teste geométrico de cada época.
        ref_step = float(reference_step_m or max(4.0 * step_m, 2_000.0))
        ref = points[["shelf", "x", "y"]].copy()
        ref["qx"] = np.rint(ref["x"] / ref_step).astype(np.int64)
        ref["qy"] = np.rint(ref["y"] / ref_step).astype(np.int64)
        ref = ref.drop_duplicates(["shelf", "qx", "qy"])
        self._reference_tree = cKDTree(ref[["x", "y"]].to_numpy(dtype=float))
        self._reference_shelf = ref["shelf"].to_numpy(dtype=object)

        epoch_count = sum(len(v) for v in self.epochs_by_shelf.values())
        logger.info(
            f"máscara IceLines: {len(points):,} pontos | "
            f"{len(self.shelves)} plataformas | {epoch_count} épocas-plataforma")

    def _sample(self, px, py) -> np.ndarray:
        px = np.asarray(px, dtype=float)
        py = np.asarray(py, dtype=float)
        px, py = np.broadcast_arrays(px, py)
        flat_x, flat_y = px.ravel(), py.ravel()
        j = np.rint((flat_x - self.sx[0]) / self.dx).astype(np.int64)
        i = np.rint((flat_y - self.sy[0]) / self.dy).astype(np.int64)
        valid = ((j >= 0) & (j < len(self.sx)) &
                 (i >= 0) & (i < len(self.sy)) &
                 np.isfinite(flat_x) & np.isfinite(flat_y))
        out = np.full(flat_x.shape, np.nan, dtype=float)
        out[valid] = self.DG[i[valid], j[valid]]
        return out.reshape(px.shape)

    @staticmethod
    def _nearest_epochs(times: np.ndarray, epochs: np.ndarray):
        """Vizinho temporal mais próximo em vetor ordenado, sem matriz NxM."""
        times = np.asarray(times, dtype=float)
        pos = np.searchsorted(epochs, times)
        right = np.clip(pos, 0, len(epochs) - 1)
        left = np.clip(pos - 1, 0, len(epochs) - 1)
        use_right = np.abs(epochs[right] - times) < np.abs(epochs[left] - times)
        chosen = np.where(use_right, epochs[right], epochs[left])
        return chosen, np.abs(chosen - times)

    def nearest_epoch(self, t_year: float, shelf: str) -> tuple[float, float]:
        """Época mais próxima *dentro da plataforma informada*."""
        if shelf not in self.epochs_by_shelf:
            raise KeyError(f"plataforma sem frentes: {shelf!r}")
        chosen, gap = self._nearest_epochs(
            np.array([t_year], dtype=float), self.epochs_by_shelf[shelf])
        return float(chosen[0]), float(gap[0])

    def assign_shelf(self, px, py, max_distance_m: float = 150_000.0):
        """Atribui a plataforma cuja coleção de frentes é mais próxima."""
        xy = np.c_[np.asarray(px, dtype=float), np.asarray(py, dtype=float)]
        distance, index = self._reference_tree.query(
            xy, k=1, distance_upper_bound=max_distance_m)
        assigned = np.isfinite(distance)
        safe_index = np.where(assigned, index, 0)
        shelf = np.full(len(xy), None, dtype=object)
        shelf[assigned] = self._reference_shelf[safe_index[assigned]]
        return shelf, distance, assigned

    def classify(
        self,
        px,
        py,
        t_year,
        shelf=None,
        *,
        tolerance_m: float = 0.0,
        assignment_max_m: float = 150_000.0,
        max_search_m: float = 100_000.0,
        max_epoch_gap_years: float = 1.0,
        fail_closed: bool = True,
    ) -> FrontMaskResult:
        """Classifica vetores e devolve também a rastreabilidade da decisão."""
        px = np.asarray(px, dtype=float).ravel()
        py = np.asarray(py, dtype=float).ravel()
        time = np.asarray(t_year, dtype=float)
        if time.ndim == 0:
            time = np.full(len(px), float(time))
        else:
            time = time.ravel()
        if not (len(px) == len(py) == len(time)):
            raise ValueError("px, py e t_year devem ter o mesmo tamanho.")

        if shelf is None:
            shelf_arr, _, assigned = self.assign_shelf(
                px, py, max_distance_m=assignment_max_m)
        else:
            shelf_arr = np.asarray(shelf, dtype=object)
            if shelf_arr.ndim == 0:
                shelf_arr = np.full(len(px), str(shelf_arr), dtype=object)
            else:
                shelf_arr = shelf_arr.ravel()
            if len(shelf_arr) != len(px):
                raise ValueError("shelf deve ser escalar ou ter o tamanho de px.")
            assigned = np.array(
                [s in self.epochs_by_shelf for s in shelf_arr], dtype=bool)

        n = len(px)
        inside = np.zeros(n, dtype=bool) if fail_closed else np.ones(n, dtype=bool)
        tested = np.zeros(n, dtype=bool)
        epoch_valid = np.zeros(n, dtype=bool)
        near_front = np.zeros(n, dtype=bool)
        dg_valid = np.zeros(n, dtype=bool)
        front_epoch = np.full(n, np.nan)
        epoch_gap = np.full(n, np.nan)
        front_distance = np.full(n, np.nan)
        margin = np.full(n, np.nan)

        for shelf_name in self.shelves:
            shelf_idx = np.flatnonzero(assigned & (shelf_arr == shelf_name))
            if shelf_idx.size == 0:
                continue
            finite_time = np.isfinite(time[shelf_idx])
            if not finite_time.any():
                continue
            valid_idx = shelf_idx[finite_time]
            chosen, gap = self._nearest_epochs(
                time[valid_idx], self.epochs_by_shelf[str(shelf_name)])
            front_epoch[valid_idx] = chosen
            epoch_gap[valid_idx] = gap
            good_epoch = gap <= max_epoch_gap_years
            epoch_valid[valid_idx] = good_epoch

            for epoch in np.unique(chosen[good_epoch]):
                query_idx = valid_idx[good_epoch & (chosen == epoch)]
                key = (str(shelf_name), float(epoch))
                distance, tree_index = self._trees[key].query(
                    np.c_[px[query_idx], py[query_idx]], k=1,
                    distance_upper_bound=max_search_m)
                near = np.isfinite(distance)
                front_distance[query_idx[near]] = distance[near]
                near_front[query_idx] = near
                if not near.any():
                    continue

                q = query_idx[near]
                dg_point = self._sample(px[q], py[q]).ravel()
                good_dg = np.isfinite(dg_point)
                dg_valid[q] = good_dg
                if not good_dg.any():
                    continue
                q_good = q[good_dg]
                dg_front = self._dg[key][tree_index[near][good_dg]]
                margin[q_good] = dg_front - tolerance_m - dg_point[good_dg]
                tested[q_good] = True
                inside[q_good] = margin[q_good] >= 0.0

        return FrontMaskResult(
            inside=inside, tested=tested, assigned=assigned,
            epoch_valid=epoch_valid, near_front=near_front,
            grounded_distance_valid=dg_valid, shelf=shelf_arr,
            front_epoch=front_epoch, epoch_gap_years=epoch_gap,
            front_distance_m=front_distance, margin_m=margin)

    def inside(
        self,
        px,
        py,
        t_year,
        tolerance_m: float = 0.0,
        max_search_m: float = 100_000.0,
        shelf=None,
    ):
        """Compatibilidade: devolve ``(inside, untested)``."""
        result = self.classify(
            px, py, t_year, shelf=shelf, tolerance_m=tolerance_m,
            max_search_m=max_search_m, max_epoch_gap_years=np.inf,
            fail_closed=False)
        return result.inside, ~result.tested
