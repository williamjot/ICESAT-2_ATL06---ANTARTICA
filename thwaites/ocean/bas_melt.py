"""Leitura e diagnósticos do conjunto observacional BAS/ITGC MELT.

O arquivo contém médias de 15 minutos, amostradas a cada duas horas, cerca de
1,5 m abaixo da base da Thwaites Eastern Ice Shelf. Ele mede temperatura,
salinidade prática reportada e as duas componentes horizontais da velocidade.
Não mede diretamente fluxo turbulento vertical de calor; qualquer grandeza de
transporte calculada aqui é, portanto, um *proxy horizontal*, não uma taxa de
derretimento observada.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BAS_MELT_SIZE = 365_639
BAS_MELT_SHA256 = "7eb15071fcd21d8bb42274dbcb75a6f1eba9af474bdb67153d218916600cf5ce"
BAS_MELT_DOI = "10.5285/4ffad557-1c3c-4ea7-a73d-6d782331b08a"
BAS_MELT_LAT = -75.20718
BAS_MELT_LON = -104.8254
BAS_MELT_PRESSURE_DBAR = 521.3


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_bas_melt_file(path: str | Path) -> None:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size != BAS_MELT_SIZE:
        raise ValueError(
            f"tamanho inesperado para {path}: {path.stat().st_size}; "
            f"esperado {BAS_MELT_SIZE}")
    observed = file_sha256(path)
    if observed.lower() != BAS_MELT_SHA256:
        raise ValueError(
            f"SHA256 inesperado para {path}: {observed}; "
            f"esperado {BAS_MELT_SHA256}")


def freezing_temperature(salinity_psu, pressure_dbar=BAS_MELT_PRESSURE_DBAR):
    """Ponto de congelamento aproximado da água do mar em graus Celsius."""
    salinity = np.asarray(salinity_psu, dtype=float)
    pressure = np.asarray(pressure_dbar, dtype=float)
    with np.errstate(invalid="ignore"):
        return (
            -0.0575 * salinity
            + 1.710523e-3 * np.power(np.maximum(salinity, 0.0), 1.5)
            - 2.154996e-4 * pressure
        )


def practical_salinity_from_conductivity(
    conductivity_mS_cm, temperature_c, pressure_dbar=BAS_MELT_PRESSURE_DBAR
):
    """Converte condutividade em salinidade pratica (PSS-78/EOS-80).

    Implementa o algoritmo UNESCO 1983. O TSV rotula a unidade como PSU, mas
    os metadados BAS declaram a precisao da variavel em S/m e os valores em
    torno de 28 sao coerentes com mS/cm, nao com salinidade sob a plataforma.
    """
    c = np.asarray(conductivity_mS_cm, dtype=float)
    t = np.asarray(temperature_c, dtype=float)
    p = np.asarray(pressure_dbar, dtype=float)
    ratio = c / 42.9140
    rt = (0.6766097 + 2.00564e-2 * t + 1.104259e-4 * t**2
          - 6.9698e-7 * t**3 + 1.0031e-9 * t**4)
    rp = 1.0 + (
        p * (2.070e-5 + p * (-6.370e-10 + 3.989e-15 * p))
        / (1.0 + 3.426e-2 * t + 4.464e-4 * t**2
           + (4.215e-1 - 3.107e-3 * t) * ratio)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        rtx = np.sqrt(np.maximum(ratio / (rp * rt), 0.0))
        salinity = (
            0.0080
            + (-0.1692 + (25.3851 + (14.0941 + (-7.0261 + 2.7081 * rtx)
               * rtx) * rtx) * rtx) * rtx
        )
        delta = ((t - 15.0) / (1.0 + 0.0162 * (t - 15.0))) * (
            0.0005
            + (-0.0056 + (-0.0066 + (-0.0375 + (0.0636 - 0.0144 * rtx)
               * rtx) * rtx) * rtx) * rtx
        )
    return salinity + delta


def load_bas_melt(
    path: str | Path,
    *,
    pressure_dbar: float = BAS_MELT_PRESSURE_DBAR,
    validate_hash: bool = True,
) -> pd.DataFrame:
    """Lê, normaliza e controla a qualidade do TSV BAS MELT."""
    path = Path(path)
    if validate_hash:
        validate_bas_melt_file(path)

    raw = pd.read_csv(path, sep="\t", skiprows=[1], engine="python")
    raw.columns = [str(column).strip().lower() for column in raw.columns]

    aliases = {}
    for column in raw.columns:
        if column.startswith("timestamp"):
            aliases[column] = "time"
        elif column.startswith("eastward"):
            aliases[column] = "u_cm_s"
        elif column.startswith("northward"):
            aliases[column] = "v_cm_s"
        elif column.startswith("temperature"):
            aliases[column] = "temperature_c"
        elif column.startswith("conductivity"):
            # O cabeçalho é consistente com condutividade; a unidade PSU da
            # segunda linha contradiz os metadados BAS (precisão em S/m).
            aliases[column] = "conductivity_mS_cm"
    data = raw.rename(columns=aliases)
    required = ["time", "u_cm_s", "v_cm_s", "temperature_c", "conductivity_mS_cm"]
    missing = set(required).difference(data.columns)
    if missing:
        raise ValueError(f"BAS MELT sem colunas obrigatórias: {sorted(missing)}")
    data = data[required].copy()
    data["time"] = pd.to_datetime(data["time"].astype(str).str.strip(), utc=True,
                                  errors="coerce")
    for column in required[1:]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        data.loc[data[column] <= -9_000, column] = np.nan

    # Faixas largas removem corrupção/fill sem selecionar o resultado esperado.
    valid = (
        data["time"].notna()
        & data["temperature_c"].between(-3.0, 5.0)
        & data["conductivity_mS_cm"].between(5.0, 70.0)
        & data["u_cm_s"].between(-200.0, 200.0)
        & data["v_cm_s"].between(-200.0, 200.0)
    )
    data = data.loc[valid].sort_values("time").drop_duplicates("time").reset_index(drop=True)
    data["speed_cm_s"] = np.hypot(data["u_cm_s"], data["v_cm_s"])
    data["direction_deg_from_north"] = (
        np.degrees(np.arctan2(data["u_cm_s"], data["v_cm_s"])) + 360.0
    ) % 360.0
    data["salinity_psu"] = practical_salinity_from_conductivity(
        data["conductivity_mS_cm"].to_numpy(),
        data["temperature_c"].to_numpy(),
        pressure_dbar,
    )
    data["freezing_temperature_c"] = freezing_temperature(
        data["salinity_psu"].to_numpy(), pressure_dbar)
    data["thermal_driving_c"] = (
        data["temperature_c"] - data["freezing_temperature_c"])
    data["horizontal_heat_proxy_m_s_c"] = (
        data["speed_cm_s"] / 100.0 * data["thermal_driving_c"])
    return data


def harmonic_summary(time, values, periods_hours=(12.0, 12.4206, 23.9345, 25.8193)):
    """Amplitude e R² de regressões harmônicas independentes."""
    time = pd.DatetimeIndex(pd.to_datetime(time, utc=True))
    values = np.asarray(values, dtype=float)
    elapsed_hours = np.asarray((time - time.min()).total_seconds() / 3600.0)
    finite = np.isfinite(elapsed_hours) & np.isfinite(values)
    x = elapsed_hours[finite]
    y = values[finite]
    if len(y) < 10:
        return {}
    total = float(np.sum((y - y.mean()) ** 2))
    result = {}
    for period in periods_hours:
        omega = 2.0 * np.pi / period
        design = np.c_[np.ones(len(x)), np.sin(omega * x), np.cos(omega * x)]
        coefficient, *_ = np.linalg.lstsq(design, y, rcond=None)
        fitted = design @ coefficient
        residual = float(np.sum((y - fitted) ** 2))
        result[f"{period:g}h"] = {
            "amplitude": float(np.hypot(coefficient[1], coefficient[2])),
            "r2": float(1.0 - residual / total) if total > 0 else 0.0,
        }
    return result


def summarize_ocean_forcing(data: pd.DataFrame) -> dict:
    """Resumo reprodutível, sem converter associação em causalidade."""
    speed = data["speed_cm_s"].to_numpy(dtype=float)
    thermal = data["thermal_driving_c"].to_numpy(dtype=float)
    rho, p_value = spearmanr(speed, thermal, nan_policy="omit")
    dt_hours = data["time"].sort_values().diff().dt.total_seconds().div(3600.0)
    return {
        "n": int(len(data)),
        "start": data["time"].min().isoformat(),
        "end": data["time"].max().isoformat(),
        "sampling_hours_median": float(dt_hours.median()),
        "temperature_c": {
            "median": float(data["temperature_c"].median()),
            "p10": float(data["temperature_c"].quantile(0.10)),
            "p90": float(data["temperature_c"].quantile(0.90)),
        },
        "conductivity_mS_cm_reported": {
            "median": float(data["conductivity_mS_cm"].median()),
            "p10": float(data["conductivity_mS_cm"].quantile(0.10)),
            "p90": float(data["conductivity_mS_cm"].quantile(0.90)),
        },
        "salinity_psu_converted_pss78": {
            "median": float(data["salinity_psu"].median()),
            "p10": float(data["salinity_psu"].quantile(0.10)),
            "p90": float(data["salinity_psu"].quantile(0.90)),
        },
        "speed_cm_s": {
            "median": float(data["speed_cm_s"].median()),
            "p10": float(data["speed_cm_s"].quantile(0.10)),
            "p90": float(data["speed_cm_s"].quantile(0.90)),
        },
        "thermal_driving_c": {
            "median": float(data["thermal_driving_c"].median()),
            "p10": float(data["thermal_driving_c"].quantile(0.10)),
            "p90": float(data["thermal_driving_c"].quantile(0.90)),
        },
        "horizontal_heat_proxy_m_s_c": {
            "median": float(data["horizontal_heat_proxy_m_s_c"].median()),
            "p90": float(data["horizontal_heat_proxy_m_s_c"].quantile(0.90)),
        },
        "speed_vs_thermal_driving_spearman": {
            "rho": float(rho), "p_value_naive": float(p_value),
            "warning": "p não corrige autocorrelação temporal; usar como diagnóstico",
        },
        "harmonics": {
            "eastward_velocity": harmonic_summary(data["time"], data["u_cm_s"]),
            "northward_velocity": harmonic_summary(data["time"], data["v_cm_s"]),
            "temperature": harmonic_summary(data["time"], data["temperature_c"]),
        },
    }
