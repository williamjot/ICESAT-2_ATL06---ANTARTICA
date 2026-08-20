"""Regressões da máscara IceLines por plataforma e época."""

import numpy as np
import pandas as pd

from thwaites.qc.front_mask import FrontMask, densify_fronts


def _fronts():
    return pd.DataFrame({
        "shelf": ["A", "A", "B"],
        "epoch_year": [2020.0, 2022.0, 2021.0],
        "wkt": [
            "LINESTRING (400 0, 400 400)",
            "LINESTRING (600 0, 600 400)",
            "LINESTRING (700 600, 700 1000)",
        ],
    })


def _mask():
    sx = np.arange(0.0, 1001.0, 100.0)
    sy = np.arange(0.0, 1001.0, 100.0)
    # Coordenada de fluxo sintética: distância ao aterrado cresce com x.
    dg = np.broadcast_to(sx, (len(sy), len(sx))).copy()
    return FrontMask(_fronts(), sx, sy, dg, step_m=100.0,
                     reference_step_m=100.0)


def test_densification_never_exceeds_requested_step():
    points = densify_fronts(_fronts().iloc[[0]], step_m=150.0)
    distance = np.hypot(np.diff(points["x"]), np.diff(points["y"]))
    assert distance.max() <= 150.0


def test_nearest_epoch_is_selected_within_each_shelf():
    mask = _mask()
    # Em empate, a época anterior é escolhida de forma determinística.
    assert mask.nearest_epoch(2021.0, "A") == (2020.0, 1.0)
    assert mask.nearest_epoch(2021.0, "B") == (2021.0, 0.0)


def test_same_flow_coordinate_differs_by_shelf_front():
    mask = _mask()
    result = mask.classify(
        px=[500.0, 500.0], py=[200.0, 800.0], t_year=[2020.1, 2021.0],
        shelf=["A", "B"], tolerance_m=0.0, max_epoch_gap_years=1.0)
    # x=500 está além da frente A de x=400, mas dentro da frente B de x=700.
    assert result.tested.tolist() == [True, True]
    assert result.inside.tolist() == [False, True]
    assert result.front_epoch.tolist() == [2020.0, 2021.0]


def test_epoch_gap_fails_closed_and_is_reported():
    mask = _mask()
    result = mask.classify(
        px=[500.0], py=[800.0], t_year=[2024.0], shelf=["B"],
        max_epoch_gap_years=1.0, fail_closed=True)
    assert result.assigned[0]
    assert not result.epoch_valid[0]
    assert not result.tested[0]
    assert not result.inside[0]
    assert result.epoch_gap_years[0] == 3.0


def test_automatic_shelf_assignment_uses_spatial_collection():
    mask = _mask()
    shelf, distance, assigned = mask.assign_shelf(
        [410.0, 690.0], [200.0, 800.0], max_distance_m=200.0)
    assert assigned.tolist() == [True, True]
    assert shelf.tolist() == ["A", "B"]
    assert np.all(distance <= 10.0)
