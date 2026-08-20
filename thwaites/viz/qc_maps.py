"""
thwaites.viz.qc_maps
====================
Mapas de controle de qualidade do recorte espacial e da amostragem.

Tornam auditáveis os pontos que entram na análise, os critérios de rejeição e a
suficiência da amostragem que sustenta cada estimativa de dh/dt.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm

from thwaites.config import Config
from thwaites.viz.figures import _DPI, _footer, _region, _season
from thwaites.qc.grounded_mask import BM_NAMES

# cores por classe BedMachine — fixas, para que todas as figuras concordem
_CLASS_COLORS = {
    0: "#9ecae1",   # ocean
    1: "#8c6d31",   # ice_free_land (rocha)
    2: "#f7f7f7",   # grounded_ice (o alvo — claro, é o fundo da análise)
    3: "#c994c7",   # floating_ice
    4: "#31a354",   # lake_vostok
}


def fig_mask_map(sx, sy, bm, cfg: Config, out: Path,
                 keep_field: np.ndarray | None = None) -> Path:
    """
    Máscara final: classes do BedMachine e a área efetivamente analisada.

    `keep_field` é a máscara booleana do recorte de gelo aterrado na mesma
    grade; sobreposta como contorno, mostra quanto o buffer retira em relação à
    classe bruta.
    """
    vals = sorted(BM_NAMES)
    cmap = ListedColormap([_CLASS_COLORS[v] for v in vals])
    norm = BoundaryNorm([v - 0.5 for v in vals] + [vals[-1] + 0.5], cmap.N)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15.5, 7), facecolor="white")
    ax.pcolormesh(sx / 1000, sy / 1000, bm, cmap=cmap, norm=norm,
                  shading="auto", rasterized=True)
    ax.set_aspect("equal")
    ax.set_xlabel("X polar (km)")
    ax.set_ylabel("Y polar (km)")
    ax.set_title(f"Classes BedMachine v4 — {_region(cfg)}", fontweight="bold",
                 loc="left", fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=_CLASS_COLORS[v], ec="#666")
               for v in vals]
    ax.legend(handles, [f"{v} {BM_NAMES[v]}" for v in vals], fontsize=8,
              loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=0.2, lw=0.4, ls="--")

    if keep_field is not None:
        ax2.pcolormesh(sx / 1000, sy / 1000, keep_field.astype(float),
                       cmap=ListedColormap(["#eeeeee", "#2171b5"]),
                       vmin=0, vmax=1, shading="auto", rasterized=True)
        frac = 100 * keep_field.mean()
        ax2.set_title(f"Área analisada (gelo aterrado + buffers)\n"
                      f"{frac:.1f}% da janela | GL "
                      f"{cfg.grounded.buffer_grounding_line_m/1000:.0f} km, costa "
                      f"{cfg.grounded.buffer_coast_m/1000:.0f} km",
                      fontweight="bold", loc="left", fontsize=11)
        h2 = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="#666")
              for c in ("#2171b5", "#eeeeee")]
        ax2.legend(h2, ["analisado", "excluído"], fontsize=8, loc="upper right")
    ax2.set_aspect("equal")
    ax2.set_xlabel("X polar (km)")
    ax2.set_ylabel("Y polar (km)")
    ax2.grid(True, alpha=0.2, lw=0.4, ls="--")

    _footer(fig, cfg, "| BedMachine Antarctica v4, 500 m, EPSG:3031")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def fig_points_kept_removed(kept_xy: np.ndarray, removed_xy: np.ndarray,
                            removed_reason: np.ndarray, cfg: Config,
                            out: Path) -> Path:
    """
    Pontos mantidos vs removidos, com o MOTIVO da remoção.

    Desenhado sobre amostras (não os 54 M de pontos): a densidade satura muito
    antes disso e o arquivo ficaria impraticável. A amostragem é aleatória e a
    fração está declarada na figura.
    """
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15.5, 7), facecolor="white")

    ax.scatter(kept_xy[:, 0] / 1000, kept_xy[:, 1] / 1000, s=0.6, c="#2171b5",
               alpha=0.35, edgecolor="none", rasterized=True,
               label=f"mantidos (amostra de {len(kept_xy):,})")
    ax.set_aspect("equal")
    ax.set_xlabel("X polar (km)")
    ax.set_ylabel("Y polar (km)")
    ax.set_title(f"Pontos ATL06 MANTIDOS — {_region(cfg)}", fontweight="bold",
                 loc="left", fontsize=11)
    ax.legend(fontsize=8, loc="upper right", markerscale=8)
    ax.grid(True, alpha=0.2, lw=0.4, ls="--")

    colors = {"não-aterrado": "#d94801", "buffer costa": "#6a51a3",
              "buffer linha de aterramento": "#238b45"}
    for reason, c in colors.items():
        s = removed_reason == reason
        if s.any():
            ax2.scatter(removed_xy[s, 0] / 1000, removed_xy[s, 1] / 1000, s=0.6,
                        c=c, alpha=0.4, edgecolor="none", rasterized=True,
                        label=f"{reason} ({int(s.sum()):,})")
    ax2.set_aspect("equal")
    ax2.set_xlabel("X polar (km)")
    ax2.set_ylabel("Y polar (km)")
    ax2.set_title("Pontos REMOVIDOS, por motivo", fontweight="bold",
                  loc="left", fontsize=11)
    ax2.legend(fontsize=8, loc="upper right", markerscale=8)
    ax2.grid(True, alpha=0.2, lw=0.4, ls="--")
    # os eixos precisam coincidir para a comparação visual ser honesta
    ax2.set_xlim(ax.get_xlim())
    ax2.set_ylim(ax.get_ylim())

    _footer(fig, cfg, "| amostragem aleatória apenas para desenho; "
                      "as contagens do relatório são sobre o total")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def fig_nobs_map(nodes_df: pd.DataFrame, cfg: Config, out: Path) -> Path:
    """
    Número de observações por nó e o span temporal.

    São os dois controles de amostragem que decidem se um dh/dt é sustentável:
    muitas observações concentradas em poucos anos não estimam uma taxa.
    """
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15.5, 7), facecolor="white")

    x = nodes_df["x"].to_numpy() / 1000
    y = nodes_df["y"].to_numpy() / 1000
    nobs = nodes_df["nobs"].to_numpy()
    sc = ax.scatter(x, y, c=np.maximum(nobs, 1), s=7, cmap="viridis",
                    norm=LogNorm(), edgecolor="none", rasterized=True)
    ax.set_aspect("equal")
    ax.set_xlabel("X polar (km)")
    ax.set_ylabel("Y polar (km)")
    ax.set_title(f"Observações por nó — {_region(cfg)}\n"
                 f"mediana {np.median(nobs):,.0f}", fontweight="bold",
                 loc="left", fontsize=11)
    cb = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("nº de observações (log)")
    ax.grid(True, alpha=0.2, lw=0.4, ls="--")

    col = "n_years_node" if "n_years_node" in nodes_df.columns else "tspan"
    v = nodes_df[col].to_numpy()
    lab = ("anos distintos com dado" if col == "n_years_node"
           else "span temporal (anos)")
    sc2 = ax2.scatter(x, y, c=v, s=7, cmap="magma", edgecolor="none",
                      rasterized=True)
    ax2.set_aspect("equal")
    ax2.set_xlabel("X polar (km)")
    ax2.set_ylabel("Y polar (km)")
    ax2.set_title(f"Cobertura temporal — mediana {np.nanmedian(v):.1f}",
                  fontweight="bold", loc="left", fontsize=11)
    cb2 = plt.colorbar(sc2, ax=ax2, shrink=0.8, pad=0.02)
    cb2.set_label(lab)
    ax2.grid(True, alpha=0.2, lw=0.4, ls="--")

    _footer(fig, cfg)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def fig_temporal_distribution(t_year: np.ndarray, cfg: Config, out: Path,
                              t_before: np.ndarray | None = None) -> Path:
    """
    Distribuição temporal das observações.

    Uma taxa ajustada sobre épocas mal distribuídas é frágil mesmo com muitas
    observações — daí o painel de contagem por ano e o de cobertura acumulada.
    Com `t_before`, compara a amostra antes e depois da máscara, para mostrar
    que o recorte espacial não introduziu viés temporal.
    """
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.2), facecolor="white")

    y0, y1 = cfg.temporal.year_start, cfg.temporal.year_end
    bins = np.arange(y0, y1 + 2) - 0.5 + 0.5
    years = np.arange(y0, y1 + 1)

    if t_before is not None:
        cb, _ = np.histogram(np.floor(t_before), bins=np.append(years, years[-1] + 1))
        ax.bar(years - 0.2, cb, width=0.4, color="#bdbdbd",
               label=f"antes da máscara (n={len(t_before):,})")
    ca, _ = np.histogram(np.floor(t_year), bins=np.append(years, years[-1] + 1))
    ax.bar(years + (0.2 if t_before is not None else 0.0), ca, width=0.4,
           color="#2171b5", label=f"após a máscara (n={len(t_year):,})")
    ax.set_xlabel("ano")
    ax.set_ylabel("nº de observações")
    ax.set_title(f"Observações por ano — {_season(cfg)}", fontweight="bold",
                 loc="left", fontsize=11)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks(years)

    # fração relativa: revela viés temporal que a contagem bruta esconde
    if t_before is not None and cb.sum() > 0:
        keep_frac = np.divide(ca, cb, out=np.zeros_like(ca, dtype=float),
                              where=cb > 0)
        ax2.bar(years, 100 * keep_frac, color="#6a51a3", width=0.6)
        ax2.axhline(100 * len(t_year) / len(t_before), color="k", ls="--", lw=1,
                    label=f"média {100*len(t_year)/len(t_before):.1f}%")
        ax2.set_ylabel("% das observações do ano que sobrevivem")
        ax2.set_title("Retenção por ano (uniforme = sem viés temporal)",
                      fontweight="bold", loc="left", fontsize=11)
        ax2.legend(fontsize=8)
        ax2.set_ylim(0, 105)
    else:
        ax2.hist(t_year, bins=60, color="#2171b5")
        ax2.set_ylabel("nº de observações")
        ax2.set_title("Distribuição fina", fontweight="bold", loc="left",
                      fontsize=11)
    ax2.set_xlabel("ano")
    ax2.set_xticks(years)
    ax2.spines[["top", "right"]].set_visible(False)

    _footer(fig, cfg)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def fig_reliability_map(nodes_df: pd.DataFrame, cfg: Config, out: Path) -> Path:
    """
    Classificação de confiabilidade por nó (coluna `reliability`).

    Três classes, com os critérios impressos na própria figura para que a
    leitura não dependa de consultar a documentação.
    """
    order = ["confiável", "aceitável com ressalvas", "não confiável"]
    colors = {"confiável": "#238b45", "aceitável com ressalvas": "#fe9929",
              "não confiável": "#cb181d"}

    fig, ax = plt.subplots(figsize=(9.5, 8), facecolor="white")
    for k in order:
        s = nodes_df["reliability"] == k
        if s.any():
            ax.scatter(nodes_df.loc[s, "x"] / 1000, nodes_df.loc[s, "y"] / 1000,
                       s=8, c=colors[k], edgecolor="none", rasterized=True,
                       label=f"{k}: {int(s.sum()):,} ({100*s.mean():.1f}%)")
    ax.set_aspect("equal")
    ax.set_xlabel("X polar (km)")
    ax.set_ylabel("Y polar (km)")
    ax.set_title(f"Confiabilidade do dh/dt por nó — {_region(cfg)}",
                 fontweight="bold", loc="left")
    ax.legend(fontsize=9, loc="upper right", markerscale=2.5)
    ax.grid(True, alpha=0.2, lw=0.4, ls="--")
    _footer(fig, cfg, "| critérios em outputs/tables/reliability_criteria.json")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out
