"""
DH/Dt Lagrangiano por plataforma em janelas sazonais móveis.

Cada plataforma é processada isoladamente. Assim, uma vizinhança ou trajetória
da Thwaites nunca agrega observações de Pine Island/Crosson por proximidade.
Para estações que atravessam o ano civil (DJF), dezembro pertence ao ano da
estação seguinte: DJF-2021 = dez/2020 + jan-fev/2021.

Entrada ICESat-2: somente ATL06. A velocidade ITS_LIVE e o geoide do
BedMachine são produtos externos auxiliares.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config
from thwaites.corrections.datum import GeoidField, to_orthometric
from thwaites.glaciology.trajectory import VelocityField, track_parcels
from thwaites.io.memory import free_memory_gb
from thwaites.logging import setup_logging


def _decimal_month(t_year: np.ndarray) -> np.ndarray:
    """Mês aproximado a partir do ano decimal; suficiente longe das viradas."""
    fraction = np.asarray(t_year, dtype=float) - np.floor(t_year)
    return np.clip(np.floor(fraction * 12.0).astype(int) + 1, 1, 12)


def season_year(t_year: np.ndarray, months: list[int]) -> np.ndarray:
    """Rótulo anual da estação, corrigindo estações que cruzam dezembro."""
    year = np.floor(t_year).astype(int)
    ordered = np.array(sorted(set(months)), dtype=int)
    if 1 not in ordered or 12 not in ordered:
        return year
    gaps = np.diff(ordered)
    high_start = int(ordered[np.argmax(gaps) + 1])
    month = _decimal_month(t_year)
    return year + (month >= high_start)


def _fit_shelf(shelf_name, data, vel, args, log):
    """Cria parcelas e ajusta todas as janelas de uma única plataforma."""
    from scipy.spatial import cKDTree

    years = np.array(sorted(data["season_year"].unique()), dtype=int)
    if len(years) < args.min_epochs:
        log.warning(f"{shelf_name}: só {len(years)} estações — ignorada")
        return []

    step = args.spacing_km * 1000.0
    gx = np.arange(data["x"].min(), data["x"].max() + step, step)
    gy = np.arange(data["y"].min(), data["y"].max() + step, step)
    px, py = (array.ravel() for array in np.meshgrid(gx, gy))
    all_tree = cKDTree(data[["x", "y"]].to_numpy(dtype=float))
    nearest, _ = all_tree.query(np.c_[px, py], k=1)
    spatial_keep = nearest <= args.radius_km * 1000.0
    px, py = px[spatial_keep], py[spatial_keep]
    del all_tree
    if len(px) == 0:
        return []

    trees, heights, epoch_time = {}, {}, {}
    for year in years:
        sample = data[data["season_year"] == year]
        if sample.empty:
            continue
        trees[year] = cKDTree(sample[["x", "y"]].to_numpy(dtype=float))
        heights[year] = sample["h"].to_numpy(dtype=float)
        epoch_time[year] = float(sample["t_year"].median())

    window, stride = args.window_years, args.step_years
    starts = range(int(years.min()), int(years.max()) - window + 2, stride)
    radius_m = args.radius_km * 1000.0
    rows = []
    for start in starts:
        window_years = [year for year in range(start, start + window)
                        if year in trees]
        if len(window_years) < args.min_epochs:
            continue
        epochs = np.array([epoch_time[year] for year in window_years])
        t_ref = float(np.mean(epochs))
        xx, yy, valid = track_parcels(vel, px, py, t_ref, epochs)

        height = np.full((len(epochs), len(px)), np.nan)
        counts = np.zeros((len(epochs), len(px)), dtype=np.int64)
        for i, year in enumerate(window_years):
            good = valid[i] & np.isfinite(xx[i]) & np.isfinite(yy[i])
            parcel_index = np.flatnonzero(good)
            if parcel_index.size == 0:
                continue
            neighbor_lists = trees[year].query_ball_point(
                np.c_[xx[i, parcel_index], yy[i, parcel_index]], r=radius_m)
            values = heights[year]
            for parcel, neighbors in zip(parcel_index, neighbor_lists):
                if neighbors:
                    height[i, parcel] = float(np.median(values[neighbors]))
                    counts[i, parcel] = len(neighbors)

        displacement = np.hypot(xx[-1] - xx[0], yy[-1] - yy[0])
        observations = counts.sum(axis=0)
        produced = 0
        for parcel in range(len(px)):
            available = counts[:, parcel] > 0
            n_epochs = int(available.sum())
            if n_epochs < args.min_epochs or observations[parcel] < args.min_obs:
                continue
            time = epochs[available]
            centered_time = time - time.mean()
            observed_height = height[available, parcel]
            design = np.c_[centered_time, np.ones(n_epochs)]
            coefficient, *_ = np.linalg.lstsq(
                design, observed_height, rcond=None)
            residual = observed_height - design @ coefficient
            rmse = float(np.sqrt(np.mean(residual ** 2)))
            dof = n_epochs - 2
            sxx = float(np.sum(centered_time ** 2))
            sigma_rate = (float(np.sqrt(np.sum(residual ** 2) / dof / sxx))
                          if dof > 0 and sxx > 0 else np.nan)
            rows.append({
                "shelf": str(shelf_name),
                "season": args.season_name,
                "window_start": start,
                "window_end": start + window - 1,
                "t_center": float(time.mean()),
                "x_ref": float(px[parcel]),
                "y_ref": float(py[parcel]),
                "dhdt_lagrangian": float(coefficient[0]),
                "dhdt_sigma_stat": sigma_rate,
                "rmse": rmse,
                "n_epochs": n_epochs,
                "n_obs": int(observations[parcel]),
                "epoch_span_years": float(time.max() - time.min()),
                "displacement_m": float(displacement[parcel]),
            })
            produced += 1
        log.info(
            f"{shelf_name} {start}-{start+window-1}: {produced:,} parcelas | "
            f"livre {free_memory_gb():.1f} GB")
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="DH/Dt lagrangiano por plataforma e janela.")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--input", default="atl06_shelf_dynamic.parquet")
    parser.add_argument("--velocity", default="velocity_itslive_annual.nc")
    parser.add_argument("--window-years", type=int, default=3)
    parser.add_argument("--step-years", type=int, default=1)
    parser.add_argument("--spacing-km", type=float, default=2.0)
    parser.add_argument("--radius-km", type=float, default=3.0)
    parser.add_argument("--min-epochs", type=int, default=3)
    parser.add_argument("--min-obs", type=int, default=20)
    parser.add_argument("--max-rmse-m", type=float, default=2.0)
    parser.add_argument("--decimate", type=int, default=4)
    args = parser.parse_args()

    cfg = load_config(args.profile)
    if cfg.product.short_name != "ATL06":
        raise ValueError("decisão metodológica: somente ATL06 é permitido.")
    args.season_name = cfg.season.name
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level,
                        run_name="shelf_windows_dynamic")

    source = cfg.paths.interim / args.input
    velocity_path = cfg.paths.data_dir / args.velocity
    for path in (source, velocity_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} não existe.")

    columns = ["x", "y", "t_year", "h_corr", "shelf"]
    data = pd.read_parquet(source, columns=columns)
    data = data[
        np.isfinite(data["h_corr"]) & np.isfinite(data["t_year"]) &
        data["shelf"].notna()].copy()
    data["season_year"] = season_year(
        data["t_year"].to_numpy(), cfg.season.months)

    geoid = GeoidField(
        cfg, data["x"].min(), data["x"].max(),
        data["y"].min(), data["y"].max())
    data["h"] = to_orthometric(
        data["h_corr"].to_numpy(),
        geoid.at(data["x"].to_numpy(), data["y"].to_numpy()))
    data = data[np.isfinite(data["h"])]
    log.info(
        f"{len(data):,} observações em {data['shelf'].nunique()} plataformas")

    velocity = VelocityField(velocity_path, decimate=args.decimate)
    rows = []
    for shelf_name, shelf_data in data.groupby("shelf", sort=True):
        rows.extend(_fit_shelf(shelf_name, shelf_data, velocity, args, log))
    if not rows:
        raise SystemExit("nenhuma parcela ajustada.")

    output = pd.DataFrame(rows)
    output["reliable"] = (
        (output["rmse"] <= args.max_rmse_m) &
        np.isfinite(output["dhdt_sigma_stat"]))
    destination = cfg.paths.dhdt_dir / "shelf_lagrangian_windows.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(destination, index=False)

    per_shelf_window = {}
    for (shelf, start, end), group in output.groupby(
            ["shelf", "window_start", "window_end"]):
        reliable = group[group["reliable"]]
        per_shelf_window[f"{shelf}:{start}-{end}"] = {
            "n": int(len(group)),
            "n_reliable": int(len(reliable)),
            "dhdt_median": (float(reliable["dhdt_lagrangian"].median())
                            if len(reliable) else None),
            "dhdt_sigma_stat_median": (
                float(reliable["dhdt_sigma_stat"].median())
                if len(reliable) else None),
            "rmse_median": float(group["rmse"].median()),
        }

    unique_parcels = output.groupby(["shelf", "x_ref", "y_ref"]).ngroups
    report = {
        "STATUS": "DH/Dt_LAGRANGIANO_DINAMICO_POR_PLATAFORMA",
        "proveniencia": {
            "produtos_ICESat2_usados": [
                f"{cfg.product.short_name} v{cfg.product.version}"],
            "outros_produtos_ICESat2_usados": [],
            "produtos_externos": ["ITS_LIVE anual", "BedMachine/geoid"],
        },
        "season": cfg.season.name,
        "season_months": cfg.season.months,
        "djf_year_rule": "dezembro pertence ao ano da estação seguinte",
        "window_years": args.window_years,
        "step_years": args.step_years,
        "n_records": int(len(output)),
        "n_unique_parcels": int(unique_parcels),
        "n_shelves": int(output["shelf"].nunique()),
        "reliable_criterion": (
            f"rmse <= {args.max_rmse_m:.1f} m e sigma estatístico finito"),
        "per_shelf_window": per_shelf_window,
        "limitacoes": [
            "janelas móveis sobrepostas não são observações independentes e "
            "não devem ser empilhadas num único histograma",
            "dhdt_sigma_stat inclui somente o erro do ajuste entre épocas; "
            "incertezas sistemáticas entram na etapa de propagação",
            "a máscara dinâmica de entrada ainda não recupera avanços sobre "
            "o oceano da máscara BedMachine nominal de 2015",
        ],
    }
    report_path = cfg.paths.tables / "shelf_windows_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(
        f"Janelas -> {destination} ({len(output):,} registros; "
        f"{unique_parcels:,} parcelas distintas)")
    log.info(f"Relatório -> {report_path}")


if __name__ == "__main__":
    main()
