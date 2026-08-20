"""
thwaites.validate.velocity
==========================
Prioridade 7 (§7): integração com a velocidade do gelo.

PERGUNTAS (§7.1)
  - Regiões com dh/dt ≈ 0 estão realmente estáveis?
  - Há aceleração de fluxo onde ainda não há adelgaçamento significativo?
  - Adelgaçamento, dinâmica e proximidade da linha de aterramento formam
    padrões espacialmente coerentes?

DUAS TRAVAS DE HONESTIDADE IMPLEMENTADAS AQUI
---------------------------------------------
1. **Aceleração de fluxo é BLOQUEADA com mosaico único** (§7.2/§7.5): o
   NSIDC-0754 é uma composição de 1996–2018. Uma composição média caracteriza
   o regime dinâmico, mas NÃO permite afirmar aceleração — para isso seriam
   necessárias múltiplas épocas comparáveis com 2019–2025. `flow_acceleration`
   recusa-se a calcular a partir de uma única época.

2. **Reamostragem por AGREGAÇÃO, não interpolação** (§7.3): a velocidade tem
   450 m e os nós têm quilômetros. Interpolar ponto-a-ponto criaria resolução
   falsa; aqui a velocidade é agregada (mediana) dentro da vizinhança do nó,
   com a dispersão preservada.

CLASSIFICAÇÃO CONJUNTA (§7.4) usa INTERVALOS DE CONFIANÇA, não cortes
arbitrários: "estável" só é atribuído quando dh/dt é compatível com zero
DENTRO da sua incerteza (§7.5), e cobertura insuficiente vira "inconclusivo"
em vez de virar "estável" por omissão.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from thwaites.config import Config
from thwaites.logging import get_logger

# Classes da classificação conjunta (§7.4)
CLASS_THIN_FAST = "adelgaçamento_significativo_fluxo_rapido"
CLASS_THIN_SLOW = "adelgaçamento_significativo_fluxo_lento"
CLASS_STABLE_FAST = "sem_tendencia_mas_fluxo_rapido"    # possível precursor
CLASS_STABLE = "estavel_conjunto"
CLASS_INCONCLUSIVE = "inconclusivo"

ACCELERATION_BLOCKED_MSG = (
    "Aceleração de fluxo NÃO pode ser derivada de um mosaico único de "
    "velocidade (§7.2). Uma composição média caracteriza o regime dinâmico, "
    "mas afirmar aceleração exige múltiplas épocas comparáveis com o período "
    "do dh/dt (2019–2025).")


# ------------------------------------------------------------------ amostragem
def sample_velocity_at(x, y, cfg: Config, path=None) -> pd.DataFrame:
    """Amostra vx, vy e a magnitude nos pontos (x, y) EPSG:3031 (bilinear)."""
    from scipy.interpolate import RegularGridInterpolator
    from thwaites.glaciology.flux import load_velocity

    gx, gy, vx, vy = load_velocity(cfg, path)
    pts = np.c_[np.asarray(y, dtype=float), np.asarray(x, dtype=float)]
    kw = dict(method="linear", bounds_error=False, fill_value=np.nan)
    VX = RegularGridInterpolator((gy, gx), vx, **kw)(pts)
    VY = RegularGridInterpolator((gy, gx), vy, **kw)(pts)
    return pd.DataFrame({"vx": VX, "vy": VY, "speed": np.hypot(VX, VY)})


def aggregate_velocity_to_nodes(x, y, cfg: Config, radius_m: float | None = None,
                                path=None) -> pd.DataFrame:
    """
    Agrega a velocidade na VIZINHANÇA de cada nó (§7.3), em vez de interpolar
    num ponto.

    Interpolar um produto de 450 m em nós de vários km sugeriria uma resolução
    que o resultado não tem. Aqui usa-se a MEDIANA dos pixels dentro do raio do
    nó (robusta ao ruído do mosaico) e guarda-se a dispersão, que é a incerteza
    de representatividade.
    """
    from scipy.spatial import cKDTree
    from thwaites.glaciology.flux import load_velocity

    logger = get_logger()
    radius = radius_m if radius_m is not None else cfg.dhdt.search_radius_m
    gx, gy, vx, vy = load_velocity(cfg, path)
    GX, GY = np.meshgrid(gx, gy)
    speed = np.hypot(vx, vy)
    ok = np.isfinite(speed)
    if not ok.any():
        raise ValueError("mosaico de velocidade sem pixels válidos.")

    pts = np.c_[GX[ok], GY[ok]]
    sp = speed[ok]
    vxo, vyo = vx[ok], vy[ok]
    tree = cKDTree(pts)

    x = np.asarray(x, float); y = np.asarray(y, float)
    rows = []
    for xi, yi in zip(x, y):
        idx = tree.query_ball_point([xi, yi], r=radius)
        if not idx:
            rows.append((np.nan, np.nan, np.nan, np.nan, 0))
            continue
        idx = np.asarray(idx, dtype=int)
        s = sp[idx]
        rows.append((float(np.median(s)),
                     float(1.4826 * np.median(np.abs(s - np.median(s)))),
                     float(np.median(vxo[idx])), float(np.median(vyo[idx])),
                     int(idx.size)))
    out = pd.DataFrame(rows, columns=["speed", "speed_mad", "vx", "vy", "n_pixels"])
    logger.info(f"velocidade agregada em raio de {radius/1000:.0f} km | "
                f"cobertura {100*np.mean(out['n_pixels'] > 0):.1f}%")
    return out


def flow_acceleration(*args, **kwargs):
    """
    Aceleração de fluxo — BLOQUEADA com mosaico único (§7.2/§7.5).

    Existe deliberadamente para falhar de forma explícita: é fácil calcular uma
    "aceleração" a partir de um único mosaico por engano, e o resultado seria
    inválido.
    """
    raise NotImplementedError(ACCELERATION_BLOCKED_MSG)


# ------------------------------------------------- linha de aterramento (§7.3)
def distance_to_grounding_line(x, y, cfg: Config, tif_path=None) -> np.ndarray:
    """
    Distância (m) de cada ponto à linha de aterramento, derivada da máscara
    BedMachine (fronteira entre gelo aterrado e flutuante).

    Positiva em qualquer direção — é distância, não deslocamento com sinal.
    """
    import rasterio
    from rasterio.windows import from_bounds, Window
    from scipy.ndimage import distance_transform_edt, binary_dilation
    from thwaites.qc.mask import resolve_mask_path

    tif_path = tif_path or resolve_mask_path(cfg)
    x = np.asarray(x, float); y = np.asarray(y, float)
    pad = 50_000.0
    with rasterio.open(tif_path) as src:
        win = from_bounds(x.min() - pad, y.min() - pad, x.max() + pad, y.max() + pad,
                          transform=src.transform)
        win = win.intersection(Window(0, 0, src.width, src.height))
        band = src.read(1, window=win)
        tr = src.window_transform(win)
        res = float(abs(src.res[0]))

    grounded = band == 2
    floating = band == cfg.mask.floating_class
    # linha de aterramento = pixels aterrados adjacentes a flutuantes
    gl = grounded & binary_dilation(floating)
    if not gl.any():
        return np.full(x.size, np.nan)

    dist_px = distance_transform_edt(~gl)
    dist_m = dist_px * res

    from rasterio.transform import rowcol
    rows, cols = rowcol(tr, x, y)
    rows = np.clip(np.asarray(rows), 0, band.shape[0] - 1)
    cols = np.clip(np.asarray(cols), 0, band.shape[1] - 1)
    return dist_m[rows, cols]


# ------------------------------------------ estatística com autocorrelação
def effective_sample_size(x, y, correlation_length_m: float) -> float:
    """
    Nº EFETIVO de amostras independentes dado o comprimento de correlação
    espacial (§7.4).

    Tratar milhares de nós autocorrelacionados como independentes infla
    drasticamente a significância de qualquer correlação. n_eff ≈ área do
    domínio / área de correlação.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size < 3 or not np.isfinite(correlation_length_m) or correlation_length_m <= 0:
        return float(x.size)
    area = (np.nanmax(x) - np.nanmin(x)) * (np.nanmax(y) - np.nanmin(y))
    a_corr = np.pi * correlation_length_m ** 2
    return float(max(min(area / a_corr, x.size), 2.0))


def correlation_with_autocorrelation(x, y, vx, vy, correlation_length_m) -> dict:
    """
    Correlação entre duas variáveis espaciais, com significância corrigida pela
    autocorrelação (§7.4/§7.5).

    Reporta o p-valor ingênuo (n) e o corrigido (n_eff) lado a lado — a
    diferença entre eles costuma ser a diferença entre "altamente significativo"
    e "não conclusivo".
    """
    from scipy import stats

    a = np.asarray(vx, float); b = np.asarray(vy, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return {"n": int(ok.sum()), "status": "pontos insuficientes"}
    r_s, p_naive = stats.spearmanr(a[ok], b[ok])
    n = int(ok.sum())
    n_eff = effective_sample_size(np.asarray(x)[ok], np.asarray(y)[ok],
                                  correlation_length_m)
    # t com n_eff-2 graus de liberdade
    if n_eff > 3 and abs(r_s) < 1:
        t = r_s * np.sqrt((n_eff - 2) / (1 - r_s ** 2))
        p_eff = float(2 * stats.t.sf(abs(t), df=n_eff - 2))
    else:
        p_eff = np.nan
    return {
        "n": n, "n_effective": float(n_eff),
        "spearman_r": float(r_s),
        "p_naive": float(p_naive), "p_autocorr_corrected": p_eff,
        "significant_naive": bool(p_naive < 0.05),
        "significant_corrected": bool(np.isfinite(p_eff) and p_eff < 0.05),
        "correlation_length_m": float(correlation_length_m),
    }


# --------------------------------------------- classificação conjunta (§7.4)
def joint_classification(nodes: pd.DataFrame, cfg: Config,
                         fast_speed_m_yr: float = 100.0,
                         min_pixels: int = 5) -> pd.DataFrame:
    """
    Classifica cada nó combinando dh/dt (com sua incerteza) e velocidade.

    "Estável" NÃO é atribuído por |dh/dt| pequeno: exige que o dh/dt seja
    compatível com zero DENTRO da sua incerteza (§7.5). Sem velocidade ou sem
    incerteza, a classe é `inconclusivo` — nunca "estável" por omissão.

    A classe `sem_tendencia_mas_fluxo_rapido` é o possível PRECURSOR de mudança
    dinâmica destacado no §7.4.
    """
    out = nodes.copy()
    dhdt = out["dhdt"].to_numpy(float)
    err = (out["dhdt_err"].to_numpy(float) if "dhdt_err" in out.columns
           else np.full(len(out), np.nan))
    speed = (out["speed"].to_numpy(float) if "speed" in out.columns
             else np.full(len(out), np.nan))
    npix = (out["n_pixels"].to_numpy() if "n_pixels" in out.columns
            else np.full(len(out), min_pixels))

    has_vel = np.isfinite(speed) & (npix >= min_pixels)
    has_err = np.isfinite(err) & (err > 0)
    # adelgaçamento significativo: IC de 95% não cruza zero
    thinning_sig = has_err & (dhdt + 1.96 * err < 0)
    # compatível com zero dentro da incerteza
    stable_sig = has_err & (np.abs(dhdt) <= 1.96 * err)
    fast = has_vel & (speed >= fast_speed_m_yr)

    cls = np.full(len(out), CLASS_INCONCLUSIVE, dtype=object)
    cls[thinning_sig & fast] = CLASS_THIN_FAST
    cls[thinning_sig & has_vel & ~fast] = CLASS_THIN_SLOW
    cls[stable_sig & fast] = CLASS_STABLE_FAST
    cls[stable_sig & has_vel & ~fast] = CLASS_STABLE

    out["velocity_available"] = has_vel
    out["thinning_significant"] = thinning_sig
    out["compatible_with_zero"] = stable_sig
    out["fast_flow"] = fast
    out["joint_class"] = cls
    return out


def summarize_dynamics(nodes: pd.DataFrame, cfg: Config,
                       correlation_length_m: float | None = None) -> dict:
    """Resumo do §7.6: contagens por classe + associações com autocorrelação."""
    logger = get_logger()
    L = correlation_length_m or cfg.mass_balance.correlation_length_m or 20_000.0
    counts = nodes["joint_class"].value_counts().to_dict()
    res = {
        "n_nodes": int(len(nodes)),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "velocity_epoch": cfg.velocity.epoch_note,
        "acceleration_status": ACCELERATION_BLOCKED_MSG,
    }
    if "speed" in nodes.columns:
        res["assoc_dhdt_speed"] = correlation_with_autocorrelation(
            nodes["x"], nodes["y"], nodes["dhdt"], nodes["speed"], L)
    if "dist_gl_m" in nodes.columns:
        res["assoc_dhdt_dist_gl"] = correlation_with_autocorrelation(
            nodes["x"], nodes["y"], nodes["dhdt"], nodes["dist_gl_m"], L)

    n_prec = int((nodes["joint_class"] == CLASS_STABLE_FAST).sum())
    res["n_possible_precursor"] = n_prec
    logger.info(f"classificação conjunta: {res['class_counts']}")
    if n_prec:
        logger.warning(
            f"{n_prec} nós SEM tendência significativa mas em fluxo rápido — "
            f"possível precursor dinâmico; não declarar essas regiões estáveis.")
    return res


# ---------------------------------------------------------- compatibilidade
def crosscheck_stable_zones(nodes: pd.DataFrame, cfg: Config,
                            stable_abs_dhdt: float = 0.1,
                            fast_speed_m_yr: float = 100.0,
                            path=None):
    """
    Versão simples mantida por compatibilidade. Prefira
    `joint_classification` + `summarize_dynamics`, que usam INTERVALOS DE
    CONFIANÇA em vez do corte arbitrário `stable_abs_dhdt` (§7.4).
    """
    vel = sample_velocity_at(nodes["x"].to_numpy(), nodes["y"].to_numpy(), cfg, path)
    out = nodes.copy().reset_index(drop=True)
    out[["vx", "vy", "speed"]] = vel[["vx", "vy", "speed"]].to_numpy()
    stable = out["dhdt"].abs() <= stable_abs_dhdt
    fast = out["speed"] >= fast_speed_m_yr
    out["apparent_stability"] = stable & fast
    summary = {
        "n_nodes": int(len(out)),
        "n_stable_by_dhdt": int(stable.sum()),
        "n_apparent_stability": int((stable & fast).sum()),
        "velocity_epoch": cfg.velocity.epoch_note,
        "note": "corte arbitrário; prefira joint_classification (§7.4)",
    }
    return out, summary
