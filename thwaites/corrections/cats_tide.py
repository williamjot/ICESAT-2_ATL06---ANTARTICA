"""
thwaites.corrections.cats_tide
==============================
Maré oceânica pelo modelo regional CATS2008 (via pyTMD), em substituição ao
`tide_ocean` embutido no ATL06.

POR QUÊ: o `tide_ocean` do ATL06 vem do GOT4.8, um modelo GLOBAL. O CATS2008
(Circum-Antarctic Tidal Simulation) é regional, resolve as cavidades sob as
plataformas de gelo e tem grade de 2 km (v2023) — sobre a plataforma flutuante
da Thwaites tende a ser melhor. Afeta apenas os pontos de gelo FLUTUANTE
(~11% do conjunto); sobre gelo aterrado a maré não se aplica.

O tempo é reconstruído exatamente a partir de `t_year`:
    delta_time = (t_year − ATLAS_EPOCH) × seconds_per_year
(é a inversa exata da conversão feita na extração, sem perda de precisão).

--------------------------------------------------------------------------
ORÇAMENTO DE MEMÓRIA (lição aprendida da forma difícil — travou a máquina 2×)
--------------------------------------------------------------------------
Três fontes de consumo, TODAS precisam ser controladas:
  1. o modelo circum-antártico (1,8 GB)  -> aberto UMA vez e RECORTADO à ROI
     (4051×3325 -> ~132×180 células, ~500× menor);
  2. a tabela de pontos (20 M linhas)    -> NUNCA carregada inteira; o Parquet
     é lido/gravado em row groups (streaming);
  3. os arrays de predição               -> pontos processados em chunks.
Controlar só (1) e (3) não basta: foi (2) que derrubou a segunda tentativa.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from thwaites.config import Config
from thwaites.logging import get_logger


def resolve_model_dir(cfg: Config) -> Path:
    """Pasta RAIZ dos modelos, que é o que o pyTMD recebe em `directory`."""
    p = Path(cfg.cats.model_dir)
    return p if p.is_absolute() else cfg.paths.data_dir / p


def resolve_model_file(cfg: Config) -> Path:
    """
    Caminho do .nc no layout que o pyTMD espera:
        <model_dir>/<nome_do_modelo>/<arquivo>.nc
    (o pyTMD procura o arquivo num subdiretório com o nome do modelo).
    """
    return resolve_model_dir(cfg) / Path(cfg.cats.model_file).stem / cfg.cats.model_file


def _check_model(cfg: Config) -> Path:
    f = resolve_model_file(cfg)
    if not f.exists():
        raise FileNotFoundError(
            f"Modelo CATS não encontrado em: {f}\n"
            f"O pyTMD espera o .nc num subdiretório com o nome do modelo.\n"
            f"Baixe '{cfg.cats.model_file}' do USAP-DC (dataset 601772; exige "
            f"reCAPTCHA, portanto é manual) e coloque-o nesse caminho."
        )
    return f


class CatsTidePredictor:
    """
    Preditor de maré CATS2008 com o modelo aberto e RECORTADO uma única vez.

    Uso:
        with CatsTidePredictor(cfg) as pred:
            for lote in lotes:
                tide = pred.predict(lon, lat, t_year)

    O recorte usa a ROI da config (não exige varrer os dados antes), de modo que
    o modelo já entra pequeno na memória.
    """

    def __init__(self, cfg: Config, bounds_lonlat=None):
        import pyTMD.io

        _check_model(cfg)
        self.cfg = cfg
        self.logger = get_logger()
        p = cfg.product
        self.epoch = (int(p.atlas_epoch_year), 1, 1, 0, 0, 0)
        self._atlas_epoch = p.atlas_epoch_year
        self._spy = p.seconds_per_year

        m = pyTMD.io.model(str(resolve_model_dir(cfg))).from_database(cfg.cats.model_name)
        self.nodal = m.corrections
        self.minor = m.minor
        ds = m.open_dataset(group="z", chunks=cfg.cats.dask_chunks, append_node=False)

        if cfg.cats.apply_flexure:
            if "flexure" not in ds:
                raise ValueError("apply_flexure=true mas o modelo não traz o campo 'flexure'.")
            for c in ds.tmd.constituents:
                ds[c] = ds[c] * ds["flexure"]

        # bbox da ROI (config) -> coords do modelo; evita varrer os dados
        if bounds_lonlat is None:
            roi = cfg.roi or cfg.area
            bounds_lonlat = (roi.lon_min, roi.lon_max, roi.lat_min, roi.lat_max)
        lo0, lo1, la0, la1 = bounds_lonlat
        corner_lon = np.array([lo0, lo1, lo0, lo1], dtype=np.float64)
        corner_lat = np.array([la0, la0, la1, la1], dtype=np.float64)
        t0 = np.zeros(4, dtype=np.float64)
        X, Y = ds.tmd.coords_as(corner_lon, corner_lat, type="drift", time=t0, crs=4326)
        bounds = [float(np.min(X)), float(np.max(X)), float(np.min(Y)), float(np.max(Y))]
        self.ds = ds.tmd.crop(bounds, buffer=cfg.cats.buffer_km)
        self.logger.info(f"CATS: modelo recortado à ROI (buffer {cfg.cats.buffer_km} km) "
                         f"-> grade {dict(self.ds.sizes)}")

    # ------------------------------------------------------------------ API
    def predict(self, lon, lat, t_year) -> np.ndarray:
        """Maré (m) nos pontos/instantes dados. NaN fora do domínio do modelo."""
        import timescale

        lon = np.asarray(lon, dtype=np.float64)
        lat = np.asarray(lat, dtype=np.float64)
        n = lon.size
        if n == 0:
            return np.zeros(0, dtype=np.float64)

        delta_time = (np.asarray(t_year, dtype=np.float64) - self._atlas_epoch) * self._spy
        step = int(self.cfg.cats.chunk_size)
        out = np.full(n, np.nan, dtype=np.float64)

        for s in range(0, n, step):
            e = min(s + step, n)
            dt_c = delta_time[s:e]
            X, Y = self.ds.tmd.coords_as(lon[s:e], lat[s:e], type="drift",
                                         time=dt_c, crs=4326)
            ts = timescale.from_deltatime(dt_c, epoch=self.epoch, standard="UTC")
            # TMD3: delta time (TT-UT1) zerado, p/ casar com a saída do TMDv2.5
            deltat = np.zeros_like(ts.tt_ut1)
            local = self.ds.tmd.interp(X, Y, method=self.cfg.cats.method,
                                       extrapolate=self.cfg.cats.extrapolate,
                                       cutoff=self.cfg.cats.cutoff_km)
            tp = local.tmd.predict(ts.tide, deltat=deltat, corrections=self.nodal)
            tp = tp + local.tmd.infer(ts.tide, deltat=deltat,
                                      corrections=self.nodal, minor=self.minor)
            out[s:e] = np.ma.filled(np.asarray(tp, dtype=np.float64), np.nan).ravel()
        return out

    def close(self):
        try:
            self.ds.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def predict_cats_tide(lon, lat, t_year, cfg: Config, progress=True) -> np.ndarray:
    """Conveniência para lotes pequenos (abre e fecha o modelo a cada chamada)."""
    with CatsTidePredictor(cfg) as pred:
        return pred.predict(lon, lat, t_year)


def apply_cats_tide_streaming(src_path, dst_path, cfg: Config,
                              progress_every: int = 20) -> dict:
    """
    Substitui `tide_ocean` (GOT4.8) pelo CATS2008 em STREAMING, sem nunca
    carregar a tabela inteira.

    Lê o Parquet em row groups, prediz a maré só nas linhas elegíveis (gelo
    flutuante, se `corrections.gate_to_floating`) e grava o resultado
    incrementalmente. Preserva o valor original em `tide_ocean_got`.

    Retorna um dicionário-resumo (contagens e diferença CATS−GOT).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    logger = get_logger()
    src_path, dst_path = Path(src_path), Path(dst_path)
    if not src_path.exists():
        raise FileNotFoundError(f"entrada não encontrada: {src_path}")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        dst_path.unlink()

    pf = pq.ParquetFile(src_path)
    if cfg.corrections.gate_to_floating and "mask_class" not in pf.schema_arrow.names:
        raise ValueError("gate_to_floating=True exige 'mask_class' (rode run_mask.py).")

    n_total = n_pred = n_cov = 0
    diff_sum = diff_sq = 0.0
    diff_max = 0.0
    writer = None

    with CatsTidePredictor(cfg) as pred:
        try:
            for i, batch in enumerate(pf.iter_batches(batch_size=cfg.cats.row_batch), start=1):
                df = batch.to_pandas()
                n_total += len(df)
                got = df["tide_ocean"].to_numpy(dtype=np.float64)
                df["tide_ocean_got"] = df["tide_ocean"]

                if cfg.corrections.gate_to_floating:
                    sel = (df["mask_class"].to_numpy() == cfg.mask.floating_class)
                else:
                    sel = np.ones(len(df), dtype=bool)

                if sel.any():
                    tide = pred.predict(df.loc[sel, "lon"].to_numpy(),
                                        df.loc[sel, "lat"].to_numpy(),
                                        df.loc[sel, "t_year"].to_numpy())
                    ok = np.isfinite(tide)
                    # fora do domínio do CATS, mantém o GOT (não perde o ponto)
                    merged = np.where(ok, tide, got[sel])
                    df.loc[sel, "tide_ocean"] = merged.astype(np.float32)

                    d = tide[ok] - got[sel][ok]
                    if d.size:
                        diff_sum += float(d.sum())
                        diff_sq += float((d ** 2).sum())
                        diff_max = max(diff_max, float(np.abs(d).max()))
                    n_pred += int(sel.sum())
                    n_cov += int(ok.sum())

                table = pa.Table.from_pandas(df, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(dst_path, table.schema, compression="snappy")
                writer.write_table(table)
                if i % progress_every == 0:
                    logger.info(f"  CATS streaming: {n_total:,} linhas | "
                                f"{n_pred:,} preditas…")
        finally:
            if writer is not None:
                writer.close()

    mean = diff_sum / n_cov if n_cov else float("nan")
    rms = (diff_sq / n_cov) ** 0.5 if n_cov else float("nan")
    summary = {"n_rows": n_total, "n_predicted": n_pred, "n_covered": n_cov,
               "diff_mean_m": mean, "diff_rms_m": rms, "diff_absmax_m": diff_max}
    logger.info(f"CATS: {n_pred:,} pontos preditos, {n_cov:,} cobertos pelo modelo | "
                f"CATS−GOT: média {mean:+.4f} m, RMS {rms:.4f} m, |max| {diff_max:.3f} m")
    return summary


def apply_cats_tide(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Versão em memória (para lotes pequenos e testes). Para os dados completos,
    use `apply_cats_tide_streaming` — esta aqui carrega tudo de uma vez.
    """
    logger = get_logger()
    if "tide_ocean" not in df.columns:
        raise ValueError("coluna 'tide_ocean' ausente — rode a extração antes.")

    out = df.copy()
    out["tide_ocean_got"] = out["tide_ocean"]

    if cfg.corrections.gate_to_floating:
        if "mask_class" not in out.columns:
            raise ValueError("gate_to_floating=True exige 'mask_class' (rode run_mask.py).")
        sel = (out["mask_class"].to_numpy() == cfg.mask.floating_class)
    else:
        sel = np.ones(len(out), dtype=bool)

    if not sel.any():
        logger.warning("nenhum ponto elegível para maré CATS — nada a fazer.")
        return out

    with CatsTidePredictor(cfg) as pred:
        tide = pred.predict(out.loc[sel, "lon"].to_numpy(),
                            out.loc[sel, "lat"].to_numpy(),
                            out.loc[sel, "t_year"].to_numpy())

    got = out.loc[sel, "tide_ocean_got"].to_numpy(dtype=np.float64)
    ok = np.isfinite(tide)
    out.loc[sel, "tide_ocean"] = np.where(ok, tide, got).astype(np.float32)
    d = tide[ok] - got[ok]
    logger.info(f"CATS: {int(ok.sum()):,}/{int(sel.sum()):,} cobertos | "
                f"CATS−GOT média {np.nanmean(d):+.4f} m, desvio {np.nanstd(d):.4f} m")
    return out
