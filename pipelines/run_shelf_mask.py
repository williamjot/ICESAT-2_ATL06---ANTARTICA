"""
Máscara de plataforma com frente IceLines dependente de época e plataforma.

Entrada ICESat-2: exclusivamente ATL06. Produtos externos auxiliares:
BedMachine v4 (flutuação/linha de aterramento) e IceLines/Sentinel-1 (frentes).

Este estágio corrige recuos da frente no ramo ATL06 já processado. Como a
entrada padrão já passou pela máscara BedMachine nominal de 2015, ele não pode
recuperar sozinho gelo que avançou sobre pixels classificados como oceano em
2015. Essa recuperação exige repetir o ramo a partir de atl06_merged.parquet,
não baixar ou usar outro produto ICESat-2.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config
from thwaites.io.memory import free_memory_gb, iter_points, write_points_streaming
from thwaites.logging import setup_logging
from thwaites.qc.front_mask import FrontMask, densify_fronts
from thwaites.qc.grounded_mask import (
    BM_FLOATING_ICE,
    distance_to_grounded,
    distance_to_open_water,
    load_bedmachine_roi,
    sample_fields_at,
)
from thwaites.qc.grounding_zone import FLOATING_CONFIDENT

CACHE = "bedmachine_shelf_distfields_v2.npz"


def get_shelf_fields(cfg, log, rebuild=False):
    """Campos BedMachine do domínio de plataforma, com cache local."""
    cache = cfg.paths.interim / CACHE
    if cache.exists() and not rebuild:
        z = np.load(cache)
        log.info(f"campos BedMachine do cache: {cache.name}")
        return z["sx"], z["sy"], z["mask"], {
            "dist_to_grounded": z["dist_to_grounded"],
            "dist_to_open_water": z["dist_to_open_water"],
        }
    sx, sy, mask = load_bedmachine_roi(cfg)
    pixel_m = abs(sx[1] - sx[0])
    fields = {
        "dist_to_grounded": distance_to_grounded(mask, pixel_m),
        "dist_to_open_water": distance_to_open_water(mask, pixel_m),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache, sx=sx, sy=sy, mask=mask,
        **{name: value.astype(np.float32) for name, value in fields.items()})
    log.info(f"campos BedMachine calculados -> {cache.name}")
    return sx, sy, mask, fields


def get_densified_fronts(fronts_path, fronts, step_m, log):
    """Cacheia a operação Shapely lenta; o cache é inválido se a fonte mudar."""
    step_tag = f"{step_m:g}".replace(".", "p")
    cache = fronts_path.with_name(
        f"{fronts_path.stem}_densified_{step_tag}m.parquet")
    if (cache.exists() and
            cache.stat().st_mtime_ns >= fronts_path.stat().st_mtime_ns):
        points = pd.read_parquet(cache)
        log.info(f"frentes densificadas do cache: {cache.name} ({len(points):,})")
        return points, cache
    points = densify_fronts(fronts, step_m=step_m)
    points.to_parquet(cache, index=False)
    log.info(f"frentes densificadas -> {cache.name} ({len(points):,})")
    return points, cache


def main():
    parser = argparse.ArgumentParser(
        description="Máscara de plataforma ATL06 + IceLines por época.")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--input", default="atl06_filtered.parquet")
    parser.add_argument("--output", default="atl06_shelf_dynamic.parquet")
    parser.add_argument("--fronts", default=None,
                        help="Parquet IceLines; default: shelf.fronts_path")
    parser.add_argument("--gz-buffer", type=float, default=None)
    parser.add_argument("--front-buffer", type=float, default=None)
    parser.add_argument("--max-epoch-gap-years", type=float, default=None)
    parser.add_argument("--allow-untested", action="store_true",
                        help="sensibilidade: mantém pontos não testados")
    parser.add_argument("--rebuild-fields", action="store_true")
    parser.add_argument("--batch-rows", type=int, default=1_000_000)
    args = parser.parse_args()

    cfg = load_config(args.profile)
    if cfg.product.short_name != "ATL06":
        raise ValueError("decisão metodológica: somente ATL06 é permitido.")
    if not cfg.shelf.dynamic_front_enabled:
        raise ValueError("shelf.dynamic_front_enabled deve estar ativo.")

    log = setup_logging(cfg.paths.logs, level=cfg.logging.level,
                        run_name="shelf_mask_dynamic")
    grounding_buffer = (args.gz_buffer if args.gz_buffer is not None
                        else cfg.shelf.buffer_grounding_zone_m)
    front_buffer = (args.front_buffer if args.front_buffer is not None
                    else cfg.shelf.buffer_front_m)
    max_epoch_gap = (
        args.max_epoch_gap_years if args.max_epoch_gap_years is not None
        else cfg.shelf.front_max_epoch_gap_years)
    fail_closed = cfg.shelf.front_fail_closed and not args.allow_untested

    source = cfg.paths.interim / args.input
    destination = cfg.paths.interim / args.output
    fronts_path = (Path(args.fronts) if args.fronts else
                   cfg.paths.data_dir / cfg.shelf.fronts_path)
    if not source.exists():
        raise FileNotFoundError(f"{source} não existe.")
    if not fronts_path.exists():
        raise FileNotFoundError(
            f"{fronts_path} não existe (rode pipelines/fetch_icelines.py).")

    sx, sy, _, fields = get_shelf_fields(
        cfg, log, rebuild=args.rebuild_fields)
    fronts = pd.read_parquet(fronts_path, columns=["shelf", "epoch_year", "wkt"])
    densified, densified_path = get_densified_fronts(
        fronts_path, fronts, cfg.shelf.front_densify_step_m, log)
    front_mask = FrontMask(
        fronts, sx, sy, fields["dist_to_grounded"],
        step_m=cfg.shelf.front_densify_step_m, densified_points=densified)
    log.info(
        f"critérios: GZ >= {grounding_buffer:.0f} m | envelope da frente "
        f"{front_buffer:.0f} m | defasagem <= {max_epoch_gap:.2f} ano(s) | "
        f"fail_closed={fail_closed}")

    import pyarrow.parquet as pq

    columns = list(pq.ParquetFile(source).schema_arrow.names)
    required = {"x", "y", "t_year", "mask_class"}
    missing = required.difference(columns)
    if missing:
        raise ValueError(f"entrada sem colunas obrigatórias: {sorted(missing)}")

    stats = Counter()
    kept_by_shelf = Counter()
    tested_by_shelf = Counter()
    sensitivity_buffers = (0.0, 2_000.0, 5_000.0, 10_000.0)
    sensitivity_counts = {buffer: Counter() for buffer in sensitivity_buffers}
    epoch_gaps: list[np.ndarray] = []

    def chunks():
        for frame in iter_points(
                source, columns, batch_rows=args.batch_rows, do_downcast=False):
            x = frame["x"].to_numpy(dtype=float)
            y = frame["y"].to_numpy(dtype=float)
            sampled = sample_fields_at(x, y, sx, sy, fields)
            if "grounding_state" in frame.columns:
                # A transicao ja foi removida pelo classificador anual. Nao
                # reintroduzir aqui o buffer estatico do BedMachine.
                floating = (frame["grounding_state"].to_numpy() ==
                            FLOATING_CONFIDENT)
                outside_gz = np.ones(len(frame), dtype=bool)
            else:
                floating = frame["mask_class"].to_numpy() == BM_FLOATING_ICE
                outside_gz = sampled["dist_to_grounded"] >= grounding_buffer
            candidate = floating & outside_gz
            candidate_index = np.flatnonzero(candidate)

            stats["n_in"] += len(frame)
            stats["n_floating_raw"] += int(floating.sum())
            stats["removed_not_floating"] += int((~floating).sum())
            stats["removed_gz"] += int((floating & ~outside_gz).sum())
            stats["n_candidates"] += len(candidate_index)
            if candidate_index.size == 0:
                continue

            result = front_mask.classify(
                x[candidate], y[candidate],
                frame["t_year"].to_numpy(dtype=float)[candidate],
                tolerance_m=0.0,
                assignment_max_m=cfg.shelf.front_assignment_max_m,
                max_search_m=cfg.shelf.front_max_search_m,
                max_epoch_gap_years=max_epoch_gap,
                fail_closed=fail_closed)

            stats["front_assigned"] += int(result.assigned.sum())
            stats["front_tested"] += int(result.tested.sum())
            stats["front_untested"] += int((~result.tested).sum())
            stats["removed_unassigned_shelf"] += int((~result.assigned).sum())
            stats["removed_epoch_gap"] += int(
                (result.assigned & ~result.epoch_valid).sum())
            stats["removed_no_near_front"] += int(
                (result.assigned & result.epoch_valid & ~result.near_front).sum())
            stats["removed_invalid_grounded_distance"] += int(
                (result.assigned & result.epoch_valid & result.near_front &
                 ~result.grounded_distance_valid).sum())
            stats["removed_beyond_front_or_buffer"] += int(
                (result.tested & (result.margin_m < front_buffer)).sum())

            finite_gap = result.epoch_gap_years[np.isfinite(result.epoch_gap_years)]
            if finite_gap.size:
                epoch_gaps.append(finite_gap)
            for name in result.shelf[result.tested]:
                tested_by_shelf[str(name)] += 1

            for buffer in sensitivity_buffers:
                passes = result.tested & (result.margin_m >= buffer)
                sensitivity_counts[buffer]["TOTAL"] += int(passes.sum())
                for name in result.shelf[passes]:
                    sensitivity_counts[buffer][str(name)] += 1

            selected = result.tested & (result.margin_m >= front_buffer)
            if not fail_closed:
                selected |= ~result.tested
            local_keep = np.flatnonzero(selected)
            kept_index = candidate_index[local_keep]
            stats["n_out"] += len(kept_index)
            for name in result.shelf[local_keep]:
                kept_by_shelf[str(name)] += 1

            if kept_index.size == 0:
                continue
            out = frame.iloc[kept_index].copy()
            out["dist_gz"] = sampled["dist_to_grounded"][kept_index].astype(
                np.float32)
            out["shelf"] = result.shelf[local_keep]
            out["front_epoch"] = result.front_epoch[local_keep].astype(np.float32)
            out["front_epoch_gap_years"] = result.epoch_gap_years[
                local_keep].astype(np.float32)
            out["dist_dynamic_front"] = result.front_distance_m[
                local_keep].astype(np.float32)
            # Margem sem buffer: permite refiltrar sem repetir a geometria.
            out["front_margin_unbuffered_m"] = result.margin_m[
                local_keep].astype(np.float32)
            out["front_tested"] = result.tested[local_keep]

            log.info(
                f"{stats['n_in']:,} lidas -> {stats['n_out']:,} aceitas | "
                f"testadas {stats['front_tested']:,} | "
                f"não testadas {stats['front_untested']:,} | "
                f"livre {free_memory_gb():.1f} GB")
            yield out

    path, n_written = write_points_streaming(chunks(), destination)
    all_gaps = np.concatenate(epoch_gaps) if epoch_gaps else np.array([])
    report = {
        "STATUS": "EXPLORATORIO_DINAMICO_PARCIAL",
        "input": source.name,
        "output": destination.name,
        "proveniencia": {
            "produtos_ICESat2_usados": [
                f"{cfg.product.short_name} v{cfg.product.version}"],
            "outros_produtos_ICESat2_usados": [],
            "produtos_externos": [
                "IceLines/Sentinel-1 (frentes de calving)",
                "BedMachine Antarctica v4 (flutuação e distância ao aterrado)",
            ],
        },
        "fronts_file": fronts_path.name,
        "fronts_densified_cache": densified_path.name,
        "criterios": {
            "buffer_grounding_zone_m": grounding_buffer,
            "front_safety_buffer_m": front_buffer,
            "front_densify_step_m": cfg.shelf.front_densify_step_m,
            "front_assignment_max_m": cfg.shelf.front_assignment_max_m,
            "front_max_search_m": cfg.shelf.front_max_search_m,
            "front_max_epoch_gap_years": max_epoch_gap,
            "front_fail_closed": fail_closed,
        },
        "contagens": {key: int(value) for key, value in stats.items()},
        "n_written": int(n_written),
        "front_epoch_gap_years": {
            "median": float(np.median(all_gaps)) if all_gaps.size else None,
            "p95": float(np.percentile(all_gaps, 95)) if all_gaps.size else None,
            "max": float(np.max(all_gaps)) if all_gaps.size else None,
        },
        "tested_by_shelf": dict(sorted(tested_by_shelf.items())),
        "kept_by_shelf": dict(sorted(kept_by_shelf.items())),
        "sensibilidade_cobertura_por_buffer_m": {
            f"{buffer:g}": dict(sorted(counts.items()))
            for buffer, counts in sensitivity_counts.items()
        },
        "limitacoes_declaradas": [
            "a entrada já foi recortada pela classe flutuante do BedMachine "
            "nominal de 2015: corrige recuos, mas não recupera avanços sobre "
            "pixels então classificados como oceano",
            "IceLines fornece linhas abertas; o lado da frente é aproximado "
            "pela distância ao gelo aterrado, com plataforma e época explícitas",
            "plataformas/épocas sem cobertura dentro do limite são excluídas "
            "do produto principal quando front_fail_closed=true",
            "a geometria da linha de aterramento e a espessura do BedMachine "
            "continuam estáticas e precisam de tratamento separado",
        ],
        "proximo_reprocessamento": (
            "repetir a classificação desde atl06_merged.parquet para permitir "
            "avanços da frente; não requer outro produto ICESat-2"),
    }
    cfg.paths.tables.mkdir(parents=True, exist_ok=True)
    report_path = cfg.paths.tables / "shelf_mask_dynamic_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Máscara dinâmica -> {path} ({n_written:,} pontos)")
    log.info(f"Relatório -> {report_path}")


if __name__ == "__main__":
    main()
