"""
pipelines/fetch_atl21.py
========================
Baixa o ATL21 (Gridded Monthly Sea Surface Height Anomaly) e recorta à ROI.

O QUE É: a anomalia de altura da superfície do mar (SSHA) do ICESat-2, já
agregada pela equipe da missão em grade de 25 km (NSIDC Polar Stereographic
South, EPSG:3412) e média MENSAL. A fonte primária é a mesma do ATL07 — os
segmentos classificados como superfície do mar (`ssh_flag == 1`, ou seja,
*leads* entre placas de gelo marinho).

POR QUE ESTE PRODUTO ANTES DO ATL07 (medido, não suposto)
---------------------------------------------------------
O aproveitamento do ATL07 nesta ROI é de 0,06% a 0,77% dos segmentos (amostra
de 10 grânulos; 6 deles sem NENHUM segmento dentro da ROI). Processar os 3.910
grânulos ATL07 custaria ~217 GB de tráfego e 8-21 h para reter ~50 MB. O ATL21
é o MESMO dado de entrada, já agregado: 84 arquivos, ~300 MB, minutos.

Se a cobertura do ATL21 não sustentar um mapa de tendência, o ATL07 também não
sustentará — a limitação está na física da amostragem (só há medida onde há
lead), não no nível de processamento. Este script existe para responder isso
antes de gastar as 14 h.

DISCIPLINA DE DISCO: um grânulo por vez; o
`.h5` bruto é deletado num `try/finally` logo após a extração. Aqui o custo é
baixo, mas a regra vale igual — o que sai é o recorte leve.

LIMITAÇÃO DECLARADA: a `mean_ssha` é anomalia sobre a MSS CryoSat-2/DTU13
(estática), com maré oceânica e barômetro invertido JÁ aplicados pelo produto.
Não é medida absoluta de nível do mar e não tem calibração de deriva contra
marégrafos. Ver run_sealevel.py para o enquadramento do resultado.

Saída: data/atl21_ase_ssha.parquet  (formato longo, uma linha por célula-mês)

Uso:
    python pipelines/fetch_atl21.py
    python pipelines/fetch_atl21.py --overwrite
"""

import argparse
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import h5py

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config
from thwaites.logging import setup_logging
from thwaites.grid.reproject import to_polar

MAX_RETRIES = 3
RETRY_WAIT_S = 5

# ATL21-02_YYYYMMDDhhmmss_...  -> o produto mensal é datado pelo início do mês
_FNAME_RE = re.compile(r"ATL21-\d\d_(\d{4})(\d{2})\d{2}")

# Variáveis do grupo `monthly` efetivamente usadas. `sigma` e `n_refsurfs` NÃO
# são opcionais: sem elas não há como distinguir célula bem amostrada de célula
# com uma única superfície de referência, e a média mensal de uma célula com
# n=1 não é uma média.
_MONTHLY_VARS = {"mean_ssha": "ssha", "sigma": "sigma", "n_refsurfs": "n_refsurfs"}


def granule_filename(granule) -> str | None:
    try:
        return granule.data_links()[0].split("/")[-1]
    except (IndexError, KeyError, AttributeError):
        return None


def granule_year_month(fname: str) -> tuple[int, int] | None:
    m = _FNAME_RE.match(fname)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def extract_atl21(h5_path: Path, cfg, roi) -> pd.DataFrame:
    """
    Recorta um arquivo ATL21 mensal à ROI e devolve o formato longo.

    A grade do ATL21 é EPSG:3412 (esferoide Hughes 1980), diferente do EPSG:3031
    usado no resto do projeto. Em vez de reprojetar a grade (que exigiria
    reamostragem e introduziria erro), reprojetamos os CENTROS de célula a
    partir de `grid_lat`/`grid_lon` — que o próprio arquivo fornece — para o
    EPSG:3031. A célula continua sendo a original; só a coordenada do centro
    muda de sistema.
    """
    with h5py.File(h5_path, "r") as f:
        lat = f["grid_lat"][:]
        lon = f["grid_lon"][:]
        monthly = f["monthly"]

        in_roi = ((lon >= roi.lon_min) & (lon <= roi.lon_max) &
                  (lat >= roi.lat_min) & (lat <= roi.lat_max))
        if not in_roi.any():
            return pd.DataFrame()

        data = {}
        for h5name, out in _MONTHLY_VARS.items():
            if h5name not in monthly:
                raise KeyError(
                    f"{h5_path.name}: variável '{h5name}' ausente no grupo "
                    f"'monthly'. Preencher com NaN aqui esconderia um produto "
                    f"em formato diferente do esperado.")
            arr = monthly[h5name][:]
            fv = monthly[h5name].attrs.get("_FillValue")
            arr = np.asarray(arr, dtype="float64")
            if fv is not None:
                arr = np.where(arr == float(np.asarray(fv).ravel()[0]), np.nan, arr)
            data[out] = arr[in_roi]

        jj, ii = np.nonzero(in_roi)
        lon_r, lat_r = lon[in_roi], lat[in_roi]

    x, y = to_polar(lon_r, lat_r, cfg)
    df = pd.DataFrame({
        "cell_j": jj.astype(np.int16), "cell_i": ii.astype(np.int16),
        "x": np.asarray(x, dtype="float64"), "y": np.asarray(y, dtype="float64"),
        "lon": lon_r.astype("float64"), "lat": lat_r.astype("float64"),
        **{k: v.astype("float32") for k, v in data.items()},
    })
    # Célula sem SSHA não é dado — é ausência de lead naquele mês. Guardar a
    # linha vazia só inflaria o arquivo; a ausência fica registrada pelo
    # simples fato de o par (célula, mês) não existir na série.
    return df[np.isfinite(df["ssha"])].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(
        description="Baixa o ATL21 (SSHA mensal gridada) e recorta à ROI.")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--out", default=None,
                    help="nome do Parquet de saída sob data/ (default: cfg.sealevel.path)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    import earthaccess

    cfg = load_config(args.profile)
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level, run_name="fetch_atl21")

    sl = cfg.sealevel
    roi = cfg.roi or cfg.area
    out_path = cfg.paths.data_dir / (args.out or sl.path)
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"{out_path} já existe — use --overwrite para refazer.")

    tmp = cfg.paths.raw_temp
    tmp.mkdir(parents=True, exist_ok=True)
    for leftover in tmp.glob("ATL21*.h5"):
        log.warning(f"resíduo removido: {leftover.name}")
        leftover.unlink()

    log.info("Autenticando no NASA Earthdata...")
    earthaccess.login(strategy="all")

    log.info(f"Buscando {sl.short_name} v{sl.version} em "
             f"{roi.bounding_box} / {cfg.temporal.temporal_range}...")
    results = earthaccess.search_data(
        short_name=sl.short_name, version=sl.version,
        bounding_box=roi.bounding_box,
        temporal=cfg.temporal.temporal_range, count=-1)
    log.info(f"Grânulos mensais encontrados: {len(results)}")
    if not results:
        raise SystemExit("Nenhum grânulo ATL21 — verifique bbox/período/versão.")

    # Um arquivo por mês. Duplicatas (reprocessamentos) ficam com a versão mais
    # recente pelo nome, e a chave (ano, mês) impede contar o mesmo mês 2x.
    por_mes: dict[tuple[int, int], object] = {}
    for g in results:
        fn = granule_filename(g)
        if fn is None:
            log.warning("grânulo sem data_link utilizável, ignorado.")
            continue
        ym = granule_year_month(fn)
        if ym is None:
            log.warning(f"nome fora do padrão, ignorado: {fn}")
            continue
        anterior = por_mes.get(ym)
        if anterior is None or fn > granule_filename(anterior):
            por_mes[ym] = g
    log.info(f"Meses distintos: {len(por_mes)}")

    partes: list[pd.DataFrame] = []
    falhas: list[str] = []
    t_ini = time.time()
    bytes_baixados = 0

    for k, (ym, granule) in enumerate(sorted(por_mes.items()), start=1):
        ano, mes = ym
        fname = granule_filename(granule)
        h5_path = None
        for tentativa in range(1, MAX_RETRIES + 1):
            try:
                got = earthaccess.download([granule], str(tmp))
                if not got:
                    raise RuntimeError("earthaccess.download retornou vazio")
                h5_path = Path(got[0])
                if not h5_path.exists() or h5_path.stat().st_size < 100_000:
                    raise RuntimeError("arquivo baixado ausente ou truncado")
                bytes_baixados += h5_path.stat().st_size

                df = extract_atl21(h5_path, cfg, roi)
                if df.empty:
                    log.info(f"[{k}/{len(por_mes)}] {ano}-{mes:02d}: "
                             f"nenhuma célula com SSHA na ROI")
                else:
                    df["year"] = ano
                    df["month"] = mes
                    partes.append(df)
                    log.info(f"[{k}/{len(por_mes)}] {ano}-{mes:02d}: "
                             f"{len(df):,} células (n_refsurfs mediano "
                             f"{df['n_refsurfs'].median():.0f})")
                break
            except Exception as e:
                log.warning(f"   tentativa {tentativa}/{MAX_RETRIES} falhou: {e}")
                if tentativa < MAX_RETRIES:
                    time.sleep(RETRY_WAIT_S)
                else:
                    falhas.append(fname or f"{ano}-{mes:02d}")
                    log.error(traceback.format_exc())
            finally:
                # DELETA o .h5 SEMPRE — mesmo se a extração falhou no meio.
                if h5_path is not None and h5_path.exists():
                    try:
                        h5_path.unlink()
                    except OSError as err:
                        log.warning(f"   não removeu {h5_path.name}: {err}")
                h5_path = None

    if not partes:
        raise SystemExit("Nenhuma célula extraída — ROI provavelmente fora da "
                         "cobertura de gelo marinho do ATL21.")

    serie = pd.concat(partes, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serie.to_parquet(out_path, compression="snappy", index=False)

    dur = time.time() - t_ini
    n_cel = serie.groupby(["cell_j", "cell_i"]).ngroups
    meses_por_cel = serie.groupby(["cell_j", "cell_i"]).size()
    log.info("=" * 70)
    log.info(f"Saída: {out_path}  ({out_path.stat().st_size/1e6:.2f} MB)")
    log.info(f"Linhas (célula-mês): {len(serie):,}  |  células distintas: {n_cel:,}")
    log.info(f"Meses por célula: mediana {meses_por_cel.median():.0f}, "
             f"máx {meses_por_cel.max()}, "
             f">= {cfg.sealevel.min_months} meses: "
             f"{int((meses_por_cel >= cfg.sealevel.min_months).sum()):,} células")
    log.info(f"Tráfego: {bytes_baixados/1e9:.2f} GB  |  tempo: {dur/60:.1f} min "
             f"({bytes_baixados/1e6/max(dur,1e-9):.1f} MB/s)")
    if falhas:
        (cfg.paths.logs / "failed_atl21.txt").write_text("\n".join(falhas), encoding="utf-8")
        log.error(f"{len(falhas)} meses falharam — ver logs/failed_atl21.txt")


if __name__ == "__main__":
    main()
