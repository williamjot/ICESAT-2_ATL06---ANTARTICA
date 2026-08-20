"""
Testes da Prioridade 2 (sensibilidade dos filtros) e do manifesto (§8/§9).
"""

import json

import numpy as np
import pandas as pd
import pytest

from thwaites.experiments.manifest import (
    Manifest, config_hash, experiment_dir, file_checksum, source_tree_hash,
)
from thwaites.experiments.sensitivity import (
    apply_overrides, default_param_grid, compare_to_baseline,
    evaluate_acceptance, _bootstrap_median_ci,
)


# ------------------------------------------------------------------ manifesto
def test_manifest_records_provenance(cfg):
    man = Manifest(cfg, "teste_prov", purpose="unit test")
    d = man.data
    assert d["product"]["short_name"] == "ATL06"
    assert d["period"]["season"] == cfg.season.name
    assert "numpy" in d["dependencies"]
    assert len(d["config_hash"]) == 64          # sha256
    assert len(d["source_tree_sha256"]) == 64
    assert d["run_id"]
    p = man.write()
    assert p.exists()
    loaded = json.loads(p.read_text(encoding="utf-8"))
    assert loaded["experiment"] == "teste_prov"


def test_manifest_refuses_to_overwrite_completed(cfg):
    man = Manifest(cfg, "dup")
    man.write()                                  # manifest.json = concluído
    with pytest.raises(FileExistsError):
        Manifest(cfg, "dup")                     # protege execução anterior (§8)
    Manifest(cfg, "dup", overwrite=True)         # explícito é permitido


def test_manifest_allows_retry_after_crash(cfg):
    """
    Execução que morreu no meio deixa a pasta com config_snapshot mas SEM
    manifest.json. Isso não pode bloquear a retentativa — não há resultado a
    preservar.
    """
    Manifest(cfg, "crashed")                     # cria a pasta, não conclui
    d = cfg.paths.outputs_dir / "experiments" / "crashed"
    assert (d / "config_snapshot.json").exists()
    assert not (d / "manifest.json").exists()
    # sem overwrite: deve funcionar
    man2 = Manifest(cfg, "crashed")
    assert man2.write().exists()


def test_config_hash_changes_with_config(cfg):
    h0 = config_hash(cfg)
    cfg2 = apply_overrides(cfg, {"dhdt.min_points": 999})
    assert config_hash(cfg2) != h0


def test_manifest_input_output_checksums(cfg, tmp_path):
    f = tmp_path / "entrada.txt"
    f.write_text("dado")
    man = Manifest(cfg, "chk")
    man.add_input(f, columns=["x", "y"])
    man.add_output(f)
    assert man.data["inputs"][0]["digest"]
    assert man.data["columns_used"] == ["x", "y"]


def test_sampled_checksum_detects_tail_change(tmp_path):
    p = tmp_path / "large.bin"
    p.write_bytes(b"A" * 16 + b"B" * 16)
    first = file_checksum(p, max_bytes=16)
    p.write_bytes(b"A" * 16 + b"C" * 16)
    second = file_checksum(p, max_bytes=16)
    assert first["partial"] is True
    assert first["sample_strategy"] == "head_tail_equal"
    assert first["digest"] != second["digest"]


def test_source_tree_hash_is_stable(cfg):
    assert source_tree_hash(cfg.paths.base_dir) == source_tree_hash(cfg.paths.base_dir)


# ------------------------------------------------------------- overrides
def test_apply_overrides_does_not_mutate_original(cfg):
    base = cfg.dhdt.min_points
    new = apply_overrides(cfg, {"dhdt.min_points": base + 7})
    assert new.dhdt.min_points == base + 7
    assert cfg.dhdt.min_points == base          # original intacto


def test_apply_overrides_rejects_unknown_param(cfg):
    with pytest.raises(AttributeError):
        apply_overrides(cfg, {"dhdt.parametro_inexistente": 1})


def test_param_grid_varies_one_at_a_time():
    grid = default_param_grid()
    assert grid[0]["name"] == "baseline" and grid[0]["overrides"] == {}
    for g in grid[1:]:
        assert len(g["overrides"]) == 1         # §3.3: um parâmetro por vez
    names = [g["name"] for g in grid]
    assert len(names) == len(set(names))


# ------------------------------------------------------------- comparação
def _nodes(n=200, dhdt=-0.5, rmse=0.6, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    x = np.arange(n) * 5000.0
    return pd.DataFrame({
        "x": x, "y": np.zeros(n),
        "dhdt": dhdt + shift + rng.normal(0, 0.01, n),
        "rmse": np.full(n, rmse),
    })


def test_compare_pairs_only_matching_nodes(cfg):
    base = _nodes(200, seed=1)
    test = _nodes(150, seed=2)                  # menos nós
    comp = compare_to_baseline(base, test, cfg)
    assert comp["n_paired"] == 150
    assert comp["nodes_only_baseline"] == 50
    assert comp["coverage_change"] < 0


def test_compare_detects_real_shift(cfg):
    cfg.sensitivity.bootstrap_iters = 200
    base = _nodes(300, seed=3)
    test = _nodes(300, seed=4, shift=0.20)      # deslocamento imposto
    comp = compare_to_baseline(base, test, cfg)
    assert np.isclose(comp["dhdt_diff_median"], 0.20, atol=0.02)
    assert comp["diff_significant"] is True     # IC não cruza zero


def test_compare_no_shift_is_not_significant(cfg):
    cfg.sensitivity.bootstrap_iters = 200
    base = _nodes(300, seed=5)
    test = _nodes(300, seed=5)                  # idêntico
    comp = compare_to_baseline(base, test, cfg)
    assert abs(comp["dhdt_diff_median"]) < 1e-9


def test_bootstrap_ci_brackets_median():
    rng = np.random.default_rng(0)
    v = rng.normal(1.0, 0.1, 500)
    med, lo, hi = _bootstrap_median_ci(v, 300, seed=0)
    assert lo < med < hi
    assert np.isclose(med, 1.0, atol=0.05)


# ------------------------------------------------------------- aceite
def test_acceptance_uses_predefined_thresholds(cfg):
    cfg.sensitivity.max_median_dhdt_shift = 0.05
    comp = {"dhdt_diff_median": 0.20, "coverage_change": 0.0}   # desloca demais
    acc = evaluate_acceptance(comp, {"rmse_median": 0.6}, {"rmse_median": 0.6}, cfg)
    assert acc["dhdt_shift_ok"] is False
    assert acc["passes"] is False
    assert acc["thresholds"]["max_median_dhdt_shift"] == 0.05


def test_acceptance_flags_residual_degradation(cfg):
    comp = {"dhdt_diff_median": 0.0, "coverage_change": 0.0}
    acc = evaluate_acceptance(comp, {"rmse_median": 0.5}, {"rmse_median": 0.8}, cfg)
    assert acc["residual_ok"] is False          # +60% de resíduo
    assert acc["passes"] is False


def test_acceptance_replacement_requires_coverage_gain(cfg):
    comp = {"dhdt_diff_median": 0.0, "coverage_change": 0.0}
    acc = evaluate_acceptance(comp, {"rmse_median": 0.6}, {"rmse_median": 0.6}, cfg)
    assert acc["passes"] is True                # não degradou
    assert acc["would_replace_baseline"] is False   # mas não ganhou cobertura
