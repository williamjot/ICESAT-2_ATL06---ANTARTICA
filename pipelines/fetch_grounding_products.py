"""
Baixa e prepara produtos observacionais de linha/zona de aterramento.

Produtos confirmados para este projeto:
  - NSIDC-0498 v2 / release 2.1: linhas de aterramento DInSAR datadas
  - NSIDC-0778 v1 / release 1.1: limites da zona de aterramento
  - IS2GZANT v1: pontos F, H e Ib derivados de ICESat-2/ATL06 v3

O recorte espacial vem de ``cfg.roi`` — o mesmo usado pelos mapas de dh/dt.
Os GeoPackages oficiais são preservados em ``data/grounding/sources`` e
recortados em ``data/grounding/processed``. O HDF5 do IS2GZANT é processado em
``data/raw_temp`` e apagado em ``finally``; somente Parquet/CSV leves persistem.

Uso:
    python pipelines/fetch_grounding_products.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config
from thwaites.logging import setup_logging


PRODUCTS = {
    "grounding_line": {
        "short_name": "NSIDC-0498",
        "version": "2",
        "release": "2.1",
        "kind": "vector",
        "source_name": "InSAR_GL_Antarctica_v02.1.gpkg",
        "roi_name": "InSAR_GL_ASE_v02.1.gpkg",
    },
    "grounding_zone": {
        "short_name": "NSIDC-0778",
        "version": "1",
        "release": "1.1",
        "kind": "vector",
        "source_name": "Antarctic_GZ_2018-2020_v01.1.gpkg",
        "roi_name": "Antarctic_GZ_ASE_2018-2020_v01.1.gpkg",
    },
    "icesat2_grounding_zone": {
        "short_name": "IS2GZANT",
        "version": "1",
        "release": "1",
        "kind": "hdf5",
        "parquet_name": "IS2GZANT_v01_ASE.parquet",
        "csv_name": "IS2GZANT_v01_ASE.csv",
    },
}

IS2_GROUPS = ("Point_F", "Point_H", "Point_Ib")
IS2_FIELDS = (
    "latitude",
    "longitude",
    "beam",
    "beam_pair",
    "nominal_error",
    "repeat_cycles_no",
    "track",
    "tide_range",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _granule_name(granule: Any) -> str:
    try:
        return str(granule["umm"]["GranuleUR"])
    except (KeyError, TypeError):
        return str(granule)


def _data_links(granule: Any) -> list[str]:
    try:
        return [str(link) for link in granule.data_links()]
    except (AttributeError, TypeError):
        return []


def _granule_rank(granule: Any, kind: str) -> tuple[int, str]:
    text = " ".join([_granule_name(granule), *_data_links(granule)]).lower()
    if kind == "hdf5":
        rank = 0 if ".h5" in text else 1
    elif ".gpkg" in text:
        rank = 0
    elif ".zip" in text:
        rank = 1
    elif ".shp" in text:
        rank = 2
    else:
        rank = 3
    return rank, text


def _download_one(earthaccess, granule: Any, temp_dir: Path) -> list[Path]:
    """Baixa exatamente um grânulo e devolve os caminhos materializados."""
    downloaded = earthaccess.download([granule], str(temp_dir))
    if not downloaded:
        raise RuntimeError("earthaccess.download retornou vazio")
    return [Path(item) for item in downloaded]


def _find_vector_file(paths: Iterable[Path], temp_dir: Path) -> Path | None:
    paths = list(paths)
    for path in paths:
        if path.suffix.lower() == ".gpkg":
            return path

    for archive in paths:
        if archive.suffix.lower() != ".zip":
            continue
        target = temp_dir / f"unzipped_{archive.stem}"
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(target)
        gpkg = next(target.rglob("*.gpkg"), None)
        if gpkg is not None:
            return gpkg
        shp = next(target.rglob("*.shp"), None)
        if shp is not None:
            return shp

    return next((path for path in paths if path.suffix.lower() == ".shp"), None)


def _copy_or_convert_source(vector_path: Path, destination: Path) -> Path:
    """Preserva GeoPackage oficial ou converte um Shapefile de contingência."""
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    if vector_path.suffix.lower() == ".gpkg":
        shutil.copy2(vector_path, destination)
        return destination

    import geopandas as gpd

    gdf = gpd.read_file(vector_path)
    if gdf.crs is None:
        raise ValueError(f"fonte vetorial sem CRS: {vector_path.name}")
    gdf.to_file(destination, driver="GPKG", layer=destination.stem)
    return destination


def _clip_geopackage(source: Path, destination: Path, bbox: tuple[float, ...],
                     epsg_polar: int, overwrite: bool = False) -> dict[str, int]:
    import geopandas as gpd
    from shapely.geometry import box

    if destination.exists() and not overwrite:
        counts = {}
        for layer in gpd.list_layers(destination)["name"].tolist():
            counts[layer] = int(len(gpd.read_file(destination, layer=layer)))
        return counts
    if destination.exists():
        destination.unlink()

    lon_min, lat_min, lon_max, lat_max = bbox
    roi = box(lon_min, lat_min, lon_max, lat_max)
    layer_names = gpd.list_layers(source)["name"].tolist()
    counts: dict[str, int] = {}
    wrote = False
    for layer in layer_names:
        gdf = gpd.read_file(source, layer=layer)
        if gdf.empty:
            continue
        if gdf.crs is None:
            raise ValueError(f"camada {layer!r} sem CRS em {source.name}")
        geographic = gdf.to_crs(epsg=4326)
        selected = geographic[geographic.intersects(roi)].copy()
        if selected.empty:
            counts[layer] = 0
            continue
        selected.geometry = selected.geometry.intersection(roi)
        selected = selected[~selected.geometry.is_empty]
        selected = selected.to_crs(epsg=epsg_polar)
        selected.to_file(
            destination,
            layer=layer,
            driver="GPKG",
            mode="w" if not wrote else "a",
        )
        counts[layer] = int(len(selected))
        wrote = True
    if not wrote:
        raise ValueError(f"{source.name} não contém feições na ROI {bbox}")
    return counts


def _decode_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype.kind in {"S", "O", "U"}:
        return np.asarray([
            item.decode("utf-8", errors="replace")
            if isinstance(item, (bytes, np.bytes_)) else str(item)
            for item in values
        ])
    return values


def extract_is2gzant_h5(path: Path, bbox: tuple[float, ...]) -> pd.DataFrame:
    """Extrai somente os campos confirmados dos grupos F, H e Ib."""
    import h5py

    frames = []
    lon_min, lat_min, lon_max, lat_max = bbox
    with h5py.File(path, "r") as h5:
        for group_name in IS2_GROUPS:
            if group_name not in h5:
                raise KeyError(f"grupo obrigatório ausente: /{group_name}")
            group = h5[group_name]
            missing = [
                field for field in IS2_FIELDS
                if field != "tide_range" and field not in group
            ]
            if missing:
                raise KeyError(
                    f"campos obrigatórios ausentes em /{group_name}: {missing}")

            n_rows = len(group["latitude"])
            columns: dict[str, np.ndarray] = {}
            for field in IS2_FIELDS:
                if field not in group:
                    columns[field] = np.full(n_rows, np.nan)
                    continue
                values = np.asarray(group[field][()])
                if values.ndim == 0:
                    values = np.repeat(values, n_rows)
                if len(values) != n_rows:
                    raise ValueError(
                        f"/{group_name}/{field}: {len(values)} linhas; "
                        f"esperado {n_rows}")
                columns[field] = _decode_array(values)

            frame = pd.DataFrame(columns)
            frame.insert(0, "feature_type", group_name)
            frame.insert(1, "product", "IS2GZANT")
            frame.insert(2, "version", "1")
            finite = np.isfinite(frame["latitude"]) & np.isfinite(frame["longitude"])
            inside = (
                (frame["longitude"] >= lon_min)
                & (frame["longitude"] <= lon_max)
                & (frame["latitude"] >= lat_min)
                & (frame["latitude"] <= lat_max)
            )
            frames.append(frame[finite & inside].copy())

    if not frames:
        return pd.DataFrame(columns=["feature_type", "product", "version", *IS2_FIELDS])
    return pd.concat(frames, ignore_index=True)


def _write_is2_outputs(frame: pd.DataFrame, parquet_path: Path, csv_path: Path,
                       overwrite: bool = False) -> None:
    for output in (parquet_path, csv_path):
        if output.exists() and not overwrite:
            raise FileExistsError(
                f"{output} já existe; use --overwrite para substituir")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(parquet_path, index=False, engine="pyarrow", compression="snappy")
    frame.to_csv(csv_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Produtos de linha/zona de aterramento recortados à ROI dos mapas dh/dt.")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--count", type=int, default=50,
                        help="máximo de resultados CMR examinados por produto")
    args = parser.parse_args()

    import earthaccess

    cfg = load_config(args.profile)
    if cfg.roi is None:
        raise ValueError("cfg.roi é obrigatório: não usar silenciosamente a bbox ampla")
    bbox = cfg.roi.bounding_box
    base = cfg.paths.data_dir / "grounding"
    sources = base / "sources"
    processed = base / "processed"
    raw_parent = cfg.paths.data_dir / "raw_temp"
    for directory in (sources, processed, raw_parent):
        directory.mkdir(parents=True, exist_ok=True)

    log = setup_logging(cfg.paths.logs, level=cfg.logging.level,
                        run_name="fetch_grounding_products")
    log.info(f"ROI dos mapas dh/dt: {bbox}")

    # A autenticação interativa é obrigatória na primeira utilização. Quando ela
    # já foi persistida, reutilizamos o netrc para não solicitar credenciais no
    # processo automatizado nem expô-las em logs/argumentos.
    home = Path.home()
    has_netrc = any((home / name).exists() for name in (".netrc", "_netrc"))
    auth_strategy = "netrc" if has_netrc else "interactive"
    log.info(f"autenticação Earthdata: {auth_strategy}")
    earthaccess.login(strategy=auth_strategy)

    report: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "roi_label": cfg.roi.label,
        "bbox_epsg4326": {
            "lon_min": bbox[0], "lat_min": bbox[1],
            "lon_max": bbox[2], "lat_max": bbox[3],
        },
        "epsg_output": cfg.area.epsg_polar,
        "products": {},
    }

    with tempfile.TemporaryDirectory(dir=raw_parent, prefix="grounding_") as tmp_name:
        temp_dir = Path(tmp_name)
        for key, spec in PRODUCTS.items():
            log.info(f"buscando {spec['short_name']} v{spec['version']}")
            results = earthaccess.search_data(
                short_name=spec["short_name"],
                version=spec["version"],
                bounding_box=bbox,
                count=args.count,
            )
            if not results:
                raise RuntimeError(
                    f"nenhum resultado CMR para {spec['short_name']} v{spec['version']}")
            ordered = sorted(results, key=lambda item: _granule_rank(item, spec["kind"]))
            product_report: dict[str, Any] = {
                "short_name": spec["short_name"],
                "cmr_version": spec["version"],
                "release": spec["release"],
                "results_found": len(results),
            }

            if spec["kind"] == "vector":
                source = sources / spec["source_name"]
                selected_granule = None
                original_files: list[str] = []
                if not source.exists() or args.overwrite:
                    if source.exists():
                        source.unlink()
                    vector = None
                    for granule in ordered:
                        downloaded = _download_one(earthaccess, granule, temp_dir)
                        original_files.extend(path.name for path in downloaded)
                        vector = _find_vector_file(downloaded, temp_dir)
                        if vector is not None:
                            selected_granule = granule
                            break
                    if vector is None:
                        raise RuntimeError(
                            f"nenhum GeoPackage/Shapefile obtido para {spec['short_name']}")
                    _copy_or_convert_source(vector, source)
                roi_path = processed / spec["roi_name"]
                counts = _clip_geopackage(
                    source, roi_path, bbox, cfg.area.epsg_polar,
                    overwrite=args.overwrite)
                product_report.update({
                    "granule": _granule_name(selected_granule) if selected_granule else None,
                    "downloaded_files": original_files,
                    "source_file": str(source.relative_to(ROOT)),
                    "source_sha256": _sha256(source),
                    "roi_file": str(roi_path.relative_to(ROOT)),
                    "roi_sha256": _sha256(roi_path),
                    "features_by_layer": counts,
                })
                log.info(f"{spec['short_name']} -> {roi_path.name}: {counts}")

            else:
                parquet_path = processed / spec["parquet_name"]
                csv_path = processed / spec["csv_name"]
                if ((parquet_path.exists() or csv_path.exists()) and not args.overwrite):
                    frame = pd.read_parquet(parquet_path)
                    product_report.update({
                        "reused": True,
                        "n_rows": int(len(frame)),
                        "parquet_file": str(parquet_path.relative_to(ROOT)),
                        "csv_file": str(csv_path.relative_to(ROOT)),
                    })
                else:
                    extracted = []
                    used_granules = []
                    for granule in ordered:
                        downloaded = _download_one(earthaccess, granule, temp_dir)
                        h5_files = [path for path in downloaded if path.suffix.lower() == ".h5"]
                        for h5_path in h5_files:
                            try:
                                extracted.append(extract_is2gzant_h5(h5_path, bbox))
                                used_granules.append(_granule_name(granule))
                            finally:
                                # Regra inegociável: HDF5 bruto não persiste, mesmo em erro.
                                if h5_path.exists():
                                    h5_path.unlink()
                        if h5_files:
                            break
                    if not extracted:
                        raise RuntimeError("IS2GZANT localizado, mas nenhum HDF5 foi baixado")
                    frame = pd.concat(extracted, ignore_index=True)
                    _write_is2_outputs(
                        frame, parquet_path, csv_path, overwrite=args.overwrite)
                    product_report.update({
                        "granules": used_granules,
                        "n_rows": int(len(frame)),
                        "rows_by_feature": {
                            str(name): int(count)
                            for name, count in frame.groupby("feature_type").size().items()
                        },
                        "parquet_file": str(parquet_path.relative_to(ROOT)),
                        "parquet_sha256": _sha256(parquet_path),
                        "csv_file": str(csv_path.relative_to(ROOT)),
                        "csv_sha256": _sha256(csv_path),
                    })
                log.info(f"IS2GZANT -> {parquet_path.name}: {len(frame)} pontos")

            report["products"][key] = product_report

    # TemporaryDirectory já foi removido; esta asserção detecta qualquer violação.
    leaked_h5 = list(raw_parent.glob("grounding_*/*.h5"))
    if leaked_h5:
        raise RuntimeError(f"HDF5 temporário não removido: {leaked_h5}")

    manifest = base / "manifest.json"
    manifest.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"manifesto -> {manifest}")


if __name__ == "__main__":
    main()
