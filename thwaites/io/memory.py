"""
thwaites.io.memory
==================
Carregamento de dados com ORÇAMENTO DE MEMÓRIA.

MOTIVAÇÃO MEDIDA (não teórica): a máquina alvo tem 8 GB de RAM total, dos quais
~3,3 GB ficam livres. Um `pd.read_parquet` de `atl06_filtered.parquet`
(19,7 M linhas × 17 colunas) tem pico de ~2,3 GB — porque a tabela Arrow e o
DataFrame coexistem durante a conversão. Somando uma cópia (um `sort`, um
`assign`, uma máscara booleana) o processo entra em swap e a máquina congela.

Três mecanismos aqui:

1. **`self_destruct`** — libera cada buffer Arrow logo após convertê-lo em
   coluna pandas. Corta o pico praticamente pela metade. É a mudança de maior
   efeito e a mais barata.
2. **Seleção obrigatória de colunas** — `read_points()` exige `columns`. Ler 17
   colunas quando se usa 6 é o desperdício mais comum do projeto.
3. **Orçamento explícito** — estima o custo pelos METADADOS antes de ler e
   avisa (ou recusa) se passar do disponível, em vez de descobrir travando.
"""

from __future__ import annotations

import gc
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

# Colunas cuja precisão exige float64. x/y em metros polares chegam a ~1e6:
# float32 daria ~0,1 m de resolução, aceitável para plotar mas não para
# reconstruir vizinhanças. t_year precisa de float64 para separar trilhas
# (o passo entre segmentos é ~1e-10 ano).
_KEEP_FLOAT64 = {"x", "y", "lon", "lat", "t_year", "delta_time"}


def free_memory_gb() -> float:
    """
    Memória física livre (GB). Usa psutil se existir; no Windows cai para wmic.
    Retorna NaN se não conseguir determinar — nunca inventa um número.
    """
    try:
        import psutil
        return float(psutil.virtual_memory().available) / 1024 ** 3
    except Exception:
        pass
    if os.name == "nt":
        try:
            out = subprocess.run(
                ["wmic", "OS", "get", "FreePhysicalMemory", "/Value"],
                capture_output=True, text=True, timeout=15).stdout
            for line in out.splitlines():
                if "=" in line:
                    kb = float(line.split("=")[1].strip())
                    return kb / 1024 ** 2
        except Exception:
            pass
    return float("nan")


def estimate_bytes(path: str | Path, columns: list[str] | None = None,
                   pandas_overhead: float = 2.0) -> dict:
    """
    Estima o custo em RAM de carregar um Parquet, a partir dos METADADOS.

    `pandas_overhead=2.0` reflete o pico real medido (Arrow + DataFrame vivos
    ao mesmo tempo). Com `self_destruct=True` o fator cai para ~1,2.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    sch = pf.schema_arrow
    n = pf.metadata.num_rows
    bpr = 0
    used = []
    for name, typ in zip(sch.names, sch.types):
        if columns is not None and name not in columns:
            continue
        s = str(typ)
        bpr += 8 if ("64" in s or "double" in s) else 4 if "32" in s else 1
        used.append(name)
    final = n * bpr
    return {"rows": n, "columns": used, "bytes_per_row": bpr,
            "final_gb": final / 1024 ** 3,
            "peak_gb": final * pandas_overhead / 1024 ** 3,
            "n_row_groups": pf.metadata.num_row_groups}


def downcast(df: pd.DataFrame, keep64: set[str] | None = None) -> pd.DataFrame:
    """
    Converte float64 → float32 nas colunas em que a precisão não é crítica.

    Feito IN-PLACE por coluna (sem criar um DataFrame novo), justamente para não
    dobrar a memória enquanto se tenta reduzi-la.
    """
    keep = keep64 or _KEEP_FLOAT64
    for c in df.columns:
        if c in keep:
            continue
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32, copy=False)
    return df


def read_points(path: str | Path, columns: list[str],
                budget_gb: float | None = None,
                do_downcast: bool = True,
                strict: bool = False) -> pd.DataFrame:
    """
    Lê um Parquet de forma econômica: só as colunas pedidas, com `self_destruct`
    e downcast opcional.

    `columns` é OBRIGATÓRIO — a maior fonte de desperdício do projeto era ler
    todas as colunas quando se usava um terço delas.

    Se `strict=True` e a estimativa passar do orçamento, levanta MemoryError em
    vez de deixar a máquina entrar em swap.
    """
    import pyarrow.parquet as pq
    from thwaites.logging import get_logger

    logger = get_logger()
    path = Path(path)
    names = pq.ParquetFile(path).schema_arrow.names
    cols = [c for c in columns if c in names]
    missing = [c for c in columns if c not in names]
    if missing:
        logger.debug(f"colunas ausentes em {path.name}: {missing}")

    est = estimate_bytes(path, cols, pandas_overhead=1.2)   # com self_destruct
    free = free_memory_gb()
    budget = budget_gb if budget_gb is not None else (
        free * 0.6 if np.isfinite(free) else float("inf"))
    logger.info(f"lendo {path.name}: {est['rows']:,} linhas × {len(cols)} cols | "
                f"pico estimado {est['peak_gb']:.2f} GB | "
                f"livre {free:.1f} GB | orçamento {budget:.2f} GB")
    if est["peak_gb"] > budget:
        msg = (f"{path.name} exige ~{est['peak_gb']:.2f} GB, acima do orçamento "
               f"({budget:.2f} GB). Use streaming (iter_points) ou menos colunas.")
        if strict:
            raise MemoryError(msg)
        logger.warning(msg + " Prosseguindo — risco de swap.")

    table = pq.read_table(path, columns=cols)
    # self_destruct libera os buffers Arrow durante a conversão (corta o pico)
    df = table.to_pandas(self_destruct=True, split_blocks=True)
    del table
    gc.collect()
    if do_downcast:
        df = downcast(df)
    return df


def iter_points(path: str | Path, columns: list[str], batch_rows: int = 2_000_000,
                do_downcast: bool = True):
    """
    Itera um Parquet em lotes, sem nunca materializar a tabela inteira.

    É a alternativa a `read_points` quando o arquivo não cabe no orçamento.
    """
    import pyarrow.parquet as pq

    path = Path(path)
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    cols = [c for c in columns if c in names]
    for batch in pf.iter_batches(batch_size=batch_rows, columns=cols):
        df = batch.to_pandas(self_destruct=True, split_blocks=True)
        del batch
        if do_downcast:
            df = downcast(df)
        yield df
        del df
        gc.collect()


def write_points_streaming(chunks, path: str | Path, compression: str = "snappy"):
    """
    Grava um iterável de DataFrames num único Parquet, em row groups.

    Contrapartida de `iter_points`: permite transformar um arquivo grande sem
    manter a saída inteira na memória.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    writer = None
    total = 0
    try:
        for df in chunks:
            if df is None or len(df) == 0:
                continue
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression=compression)
            writer.write_table(table)
            total += len(df)
            del table, df
            gc.collect()
    finally:
        if writer is not None:
            writer.close()
    return path, total
