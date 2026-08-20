"""
thwaites.viz.figures
====================
Figuras do projeto, em EPSG:3031 (eixos em km) — sem dependência de download
do cartopy (robusto/offline). Convenção glaciológica de cor: vermelho =
afinamento (dh/dt < 0), azul = espessamento (dh/dt > 0).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")   # sem display
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from thwaites.config import Config

# vermelho=negativo (perda), azul=positivo (ganho); branco no zero
_CMAP = "RdBu"
_DPI = 200


def _region(cfg: Config) -> str:
    """
    Nome da região para títulos. Vem de `roi.label` (ou `area.label`); se não
    houver rótulo, cai num descritor de coordenadas em vez de assumir um nome.
    Escrever o nome da geleira no código já produziu figuras rotuladas
    "Thwaites" para um domínio que na verdade cobria todo o Amundsen.
    """
    for a in (cfg.roi, cfg.area):
        if a is not None and getattr(a, "label", None):
            return a.label
    r = cfg.roi or cfg.area
    return (f"{abs(r.lon_min):.0f}-{abs(r.lon_max):.0f}°W, "
            f"{abs(r.lat_max):.0f}-{abs(r.lat_min):.0f}°S")


def _season(cfg: Config) -> str:
    """Rótulo de estação + anos, derivado da config (não fixo em JJA)."""
    return (f"{cfg.season.name.upper()} "
            f"{cfg.temporal.year_start}–{cfg.temporal.year_end}")


def _dhdt_norm(vmin=-3.0, vmax=1.5):
    return TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)


def _footer(fig, cfg: Config, extra=""):
    fig.text(0.01, 0.01,
             f"ICESat-2 ATL06 | {_season(cfg)} | EPSG:3031 | "
             f"ROI {cfg.roi.lon_min}..{cfg.roi.lon_max}°/"
             f"{cfg.roi.lat_min}..{cfg.roi.lat_max}° {extra}",
             fontsize=6.5, color="#666666")


def fig_dhdt_map(grid_df: pd.DataFrame, nodes_df: pd.DataFrame, cfg: Config,
                 out: Path, coverage_dist_m: float | None = None) -> Path:
    """Mapa do dh/dt interpolado (grade regular), mascarado à cobertura de dados."""
    from scipy.spatial import cKDTree

    gx = np.sort(grid_df["x"].unique())
    gy = np.sort(grid_df["y"].unique())
    Z = np.full((len(gy), len(gx)), np.nan)
    ix = np.searchsorted(gx, grid_df["x"].to_numpy())
    iy = np.searchsorted(gy, grid_df["y"].to_numpy())
    Z[iy, ix] = grid_df["pred"].to_numpy()

    # máscara de cobertura: célula longe de qualquer nó -> NaN (não extrapola)
    cov = coverage_dist_m or cfg.mass_balance.coverage_dist_m
    tree = cKDTree(np.c_[nodes_df["x"], nodes_df["y"]])
    GX, GY = np.meshgrid(gx, gy)
    d, _ = tree.query(np.c_[GX.ravel(), GY.ravel()], k=1)
    Z.ravel()[d > cov] = np.nan

    fig, ax = plt.subplots(figsize=(9, 8), facecolor="white")
    im = ax.pcolormesh(gx / 1000, gy / 1000, Z, cmap=_CMAP, norm=_dhdt_norm(),
                       shading="auto", rasterized=True)
    ax.set_aspect("equal")
    ax.set_xlabel("X polar (km)"); ax.set_ylabel("Y polar (km)")
    ax.set_title(f"Taxa de mudança de elevação  dh/dt — {_region(cfg)}\n"
                 f"{_season(cfg)}", fontweight="bold", loc="left")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, extend="both", pad=0.02)
    cbar.set_label("dh/dt (m/ano)")
    ax.grid(True, alpha=0.25, linewidth=0.4, linestyle="--")
    # rótulo do método = vencedor real da seleção por CV (não hardcoded)
    import json
    sel = cfg.paths.tables / "interp_selection.json"
    method = json.loads(sel.read_text()).get("winner", "?") if sel.exists() else "?"
    _footer(fig, cfg, f"| SR={cfg.dhdt.search_radius_m/1000:.0f} km | interp: {method}")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def fig_dhdt_hist(nodes_df: pd.DataFrame, cfg: Config, out: Path) -> Path:
    """Histograma da distribuição de dh/dt (nós), com média/mediana e estatísticas."""
    from scipy import stats

    p1 = nodes_df["dhdt"].to_numpy()
    p1 = p1[np.isfinite(p1)]
    mean, med = np.mean(p1), np.median(p1)
    norm = _dhdt_norm()

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="white")
    bins = np.arange(-4, 2.55, 0.1)
    counts, edges = np.histogram(p1, bins=bins, density=True)
    # contorno preto fino: com 65 barras de 0,1 m/ano e o preenchimento vindo de
    # uma rampa divergente, as barras claras perto do zero somem contra o fundo
    # branco. 0,4 pt separa cada pilha sem virar grade.
    for i in range(len(counts)):
        mid = 0.5 * (edges[i] + edges[i + 1])
        ax.bar(edges[i], counts[i], width=edges[i + 1] - edges[i], align="edge",
               color=plt.get_cmap(_CMAP)(norm(mid)), edgecolor="black",
               linewidth=0.4)
    ax.axvline(0, color="k", lw=1, ls="--", alpha=0.5)
    ax.axvline(mean, color="#0D47A1", lw=2, label=f"média {mean:+.3f} m/ano")
    ax.axvline(med, color="#0D47A1", lw=1.5, ls=":", label=f"mediana {med:+.3f} m/ano")
    ax.set_xlabel("dh/dt (m/ano)"); ax.set_ylabel("densidade de probabilidade")
    ax.set_title(f"Distribuição de dh/dt — {_region(cfg)} ({_season(cfg)})",
                 fontweight="bold", loc="left")
    ax.set_xlim(-4, 2)
    thin = 100 * np.mean(p1 < 0)
    ax.text(0.02, 0.97,
            f"n = {len(p1):,} nós\nafinamento: {thin:.0f}%\n"
            f"skewness: {stats.skew(p1):+.2f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#BBB", alpha=0.9))
    ax.legend(loc="upper right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    _footer(fig, cfg)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def fig_xover_validation(xovers: pd.DataFrame, nodes_df: pd.DataFrame,
                         cfg: Config, out: Path,
                         paired_stats: dict | None = None) -> Path:
    """
    Comparação de método: distribuição de dh/dt por CRUZAMENTO vs. pelo fitsec.

    DUAS CORREÇÕES DE HONESTIDADE nesta figura:

    1. Crossovers NÃO são validação independente (§6.5 do plano): usam o mesmo
       produto altimétrico (ATL06), a mesma máscara e as mesmas correções. São
       um estimador ALTERNATIVO EM MÉTODO, não uma fonte externa.
    2. Distribuições parecidas NÃO significam concordância ponto a ponto. A
       análise pareada mediu MAD de ~0,2 m/ano e |z| mediano ~11,7 (só 12% das
       diferenças dentro de 2σ). Sem esse aviso, a figura sugeriria uma
       validação que os dados não sustentam — por isso a caixa de texto.
    """
    xv = xovers["dhdt"].dropna().to_numpy()
    fv = nodes_df["dhdt"].dropna().to_numpy()
    bins = np.arange(-4, 2.55, 0.1)

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="white")
    ax.hist(fv, bins=bins, density=True, color="#c0392b", alpha=0.55,
            label=f"fitsec (n={len(fv):,})  mediana {np.median(fv):+.3f}")
    ax.hist(xv, bins=bins, density=True, histtype="step", linewidth=2,
            color="#154360",
            label=f"crossover (n={len(xv):,})  mediana {np.median(xv):+.3f}")
    ax.axvline(0, color="k", lw=1, ls="--", alpha=0.5)
    ax.set_xlabel("dh/dt (m/ano)"); ax.set_ylabel("densidade de probabilidade")
    ax.set_title("Comparação de método: crossovers vs. ajuste de superfície\n"
                 f"{_region(cfg)} — estimador ALTERNATIVO (mesmo produto ATL06)",
                 fontweight="bold", loc="left")
    ax.set_xlim(-4, 2)
    ax.legend(fontsize=9, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    # aviso contra a leitura errada: distribuições semelhantes ≠ concordância
    txt = ("As distribuições coincidirem NÃO implica concordância ponto a ponto.")
    if paired_stats:
        txt += (f"\nDiferença pareada: MAD {paired_stats.get('mad_diff', float('nan')):.3f} m/ano"
                f"\n|z| mediano {paired_stats.get('median_abs_z', float('nan')):.1f}"
                f"  ·  dentro de 2σ: {100*paired_stats.get('frac_within_2sigma', float('nan')):.0f}%")
    ax.text(0.98, 0.97, txt, transform=ax.transAxes, fontsize=8.5,
            va="top", ha="right", color="#333333",
            bbox=dict(boxstyle="round,pad=0.5", fc="#FFF6E5", ec="#D4A017", alpha=0.95))

    _footer(fig, cfg, "| crossovers NÃO são validação independente (mesmo ATL06)")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def fig_uncertainty_map(grid_df: pd.DataFrame, nodes_df: pd.DataFrame,
                        cfg: Config, out: Path,
                        coverage_dist_m: float | None = None) -> Path:
    """
    Mapa da incerteza (1σ) do dh/dt, com a decomposição das duas fontes.

    Painel esquerdo: σ total por célula.
    Painel direito: fração da variância que vem do erro dos NÓS (o termo
    dominante, ~0,79 de 0,88 m/ano) versus o erro de predição espacial.

    Esta figura só faz sentido depois de `run_uncertainty.py`: antes disso o
    `dhdt_err` vinha do erro formal do ajuste, que é otimista por ~56× porque
    trata observações do mesmo ano como independentes.
    """
    from scipy.spatial import cKDTree

    gx = np.sort(grid_df["x"].unique())
    gy = np.sort(grid_df["y"].unique())
    ix = np.searchsorted(gx, grid_df["x"].to_numpy())
    iy = np.searchsorted(gy, grid_df["y"].to_numpy())

    sigma = np.full((len(gy), len(gx)), np.nan)
    sigma[iy, ix] = np.sqrt(np.abs(grid_df["var"].to_numpy()))

    frac = None
    if "sigma_input" in grid_df.columns and "var_interp" in grid_df.columns:
        si = np.abs(grid_df["sigma_input"].to_numpy()) ** 2
        vi = np.abs(grid_df["var_interp"].to_numpy())
        with np.errstate(invalid="ignore", divide="ignore"):
            f = np.where((si + vi) > 0, si / (si + vi), np.nan)
        frac = np.full((len(gy), len(gx)), np.nan)
        frac[iy, ix] = f

    cov = coverage_dist_m or cfg.mass_balance.coverage_dist_m
    tree = cKDTree(np.c_[nodes_df["x"], nodes_df["y"]])
    GX, GY = np.meshgrid(gx, gy)
    d, _ = tree.query(np.c_[GX.ravel(), GY.ravel()], k=1)
    far = (d > cov).reshape(GX.shape)
    sigma[far] = np.nan
    if frac is not None:
        frac[far] = np.nan

    ncol = 2 if frac is not None else 1
    fig, axes = plt.subplots(1, ncol, figsize=(7.5 * ncol, 7), facecolor="white")
    axes = np.atleast_1d(axes)

    vmax = float(np.nanpercentile(sigma, 98)) if np.isfinite(sigma).any() else 1.0
    im0 = axes[0].pcolormesh(gx / 1000, gy / 1000, sigma, cmap="YlOrRd",
                             vmin=0, vmax=vmax, shading="auto", rasterized=True)
    axes[0].set_title("Incerteza do dh/dt (1σ)", fontweight="bold", loc="left")
    cb0 = plt.colorbar(im0, ax=axes[0], shrink=0.8, extend="max", pad=0.02)
    cb0.set_label("σ (m/ano)")

    if frac is not None:
        im1 = axes[1].pcolormesh(gx / 1000, gy / 1000, 100 * frac, cmap="PuBu",
                                 vmin=0, vmax=100, shading="auto", rasterized=True)
        axes[1].set_title("Fração da variância vinda do erro dos NÓS",
                          fontweight="bold", loc="left")
        cb1 = plt.colorbar(im1, ax=axes[1], shrink=0.8, pad=0.02)
        cb1.set_label("% da variância total")

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("X polar (km)")
        ax.grid(True, alpha=0.25, linewidth=0.4, linestyle="--")
    axes[0].set_ylabel("Y polar (km)")

    med = float(np.nanmedian(sigma))
    axes[0].text(0.03, 0.03, f"σ mediano: {med:.3f} m/ano",
                 transform=axes[0].transAxes, fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#BBB", alpha=0.9))
    _footer(fig, cfg, "| σ por jackknife sobre anos (erro formal era ~56× otimista)")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def fig_trend_significance(trends_df: pd.DataFrame, cfg: Config, out: Path) -> Path:
    """Mapa dos nós por Sen's slope, destacando os significativos (FDR)."""
    x = trends_df["node_x"].to_numpy() / 1000
    y = trends_df["node_y"].to_numpy() / 1000
    ss = trends_df["sens_slope"].to_numpy()
    sig = trends_df["significant"].to_numpy().astype(bool)

    n_sig = int(sig.sum()); n_tot = len(sig)
    fig, ax = plt.subplots(figsize=(9, 8), facecolor="white")
    # não significativos: cinza vazado, discretos
    ax.scatter(x[~sig], y[~sig], facecolors="none", edgecolors="#bbbbbb",
               s=10, linewidths=0.5, label=f"não signif.: {n_tot - n_sig}")
    # significativos: preenchidos, coloridos por Sen's slope (a história)
    sc = ax.scatter(x[sig], y[sig], c=ss[sig], cmap=_CMAP, norm=_dhdt_norm(),
                    s=22, edgecolor="none", label=f"signif. (FDR): {n_sig}")
    ax.set_aspect("equal")
    ax.set_xlabel("X polar (km)"); ax.set_ylabel("Y polar (km)")
    ax.set_title(f"Tendência (Sen's slope) e significância — {_region(cfg)}\n"
                 f"Mann-Kendall + FDR (α=0,05), série {_season(cfg)}",
                 fontweight="bold", loc="left")
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8, extend="both", pad=0.02)
    cbar.set_label("Sen's slope (m/ano)")
    ax.grid(True, alpha=0.25, linewidth=0.4, linestyle="--")
    ax.legend(loc="upper right", fontsize=9)
    _footer(fig, cfg, f"| {100*n_sig/n_tot:.0f}% dos nós com tendência significativa")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def fig_dhdt_with_confidence(grid_df: pd.DataFrame, nodes_df: pd.DataFrame,
                             cfg: Config, out: Path, k_sigma: float = 2.0,
                             coverage_dist_m: float | None = None) -> Path:
    """
    Mapa de dh/dt com a INCERTEZA incorporada: onde |dh/dt| < k_sigma·σ o valor
    não se distingue de zero e recebe hachura, em vez de cor cheia.

    Motivo: no mapa simples, células com pouquíssimas observações produzem
    valores extremos que dominam a atenção do leitor justamente onde a
    estimativa é pior. Medido nesta ROI: os 46 nós com dh/dt > +1 m/ano tinham
    6.299 observações contra 155.818 da mediana geral, RMSE 2,45 contra 0,45, e
    razão |dh/dt|/σ = 0,9 — ou seja, dentro de 1σ de zero. A hachura mantém o
    dado visível (não o apaga, o que seria esconder informação) mas retira o
    peso visual do que não é significativo.

    Painéis: (1) dh/dt com o não-significativo hachurado; (2) σ (1 sigma).
    """
    from scipy.spatial import cKDTree

    gx = np.sort(grid_df["x"].unique())
    gy = np.sort(grid_df["y"].unique())
    shape = (len(gy), len(gx))
    ix = np.searchsorted(gx, grid_df["x"].to_numpy())
    iy = np.searchsorted(gy, grid_df["y"].to_numpy())

    Z = np.full(shape, np.nan)
    Z[iy, ix] = grid_df["pred"].to_numpy()
    S = np.full(shape, np.nan)
    if "var" in grid_df.columns:
        S[iy, ix] = np.sqrt(np.clip(grid_df["var"].to_numpy(), 0, None))

    # não extrapola: célula longe de qualquer nó vira NaN
    cov = coverage_dist_m or cfg.mass_balance.coverage_dist_m
    tree = cKDTree(np.c_[nodes_df["x"], nodes_df["y"]])
    GX, GY = np.meshgrid(gx, gy)
    dist, _ = tree.query(np.c_[GX.ravel(), GY.ravel()], k=1)
    far = (dist > cov).reshape(shape)
    Z[far] = np.nan
    S[far] = np.nan

    with np.errstate(invalid="ignore"):
        signif = np.abs(Z) >= k_sigma * S
    valid = np.isfinite(Z) & np.isfinite(S)
    frac_sig = float(signif[valid].mean()) if valid.any() else float("nan")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor="white")

    im = ax.pcolormesh(gx / 1000, gy / 1000, Z, cmap=_CMAP, norm=_dhdt_norm(),
                       shading="auto", rasterized=True)
    weak = valid & ~signif
    if weak.any():
        ax.contourf(gx / 1000, gy / 1000, weak.astype(float), levels=[0.5, 1.5],
                    colors="none", hatches=["////"], zorder=3)
        ax.contour(gx / 1000, gy / 1000, weak.astype(float), levels=[0.5],
                   colors="#444444", linewidths=0.4, zorder=4)
    ax.set_aspect("equal")
    ax.set_xlabel("X polar (km)")
    ax.set_ylabel("Y polar (km)")
    ax.set_title(f"dh/dt com significância — {_region(cfg)}\n{_season(cfg)}",
                 fontweight="bold", loc="left", fontsize=11)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, extend="both", pad=0.02)
    cbar.set_label("dh/dt (m/ano)")
    ax.grid(True, alpha=0.25, linewidth=0.4, linestyle="--")
    ax.text(0.02, 0.02,
            f"hachurado: |dh/dt| < {k_sigma:.0f}σ\n"
            f"significativo: {100*frac_sig:.0f}% das células",
            transform=ax.transAxes, fontsize=8.5, va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#BBB", alpha=0.92))

    # painel 2: σ com escala robusta — a cauda longa satura tudo se usar o máximo
    fin = S[np.isfinite(S)]
    vmax = float(np.percentile(fin, 98)) if fin.size else 1.0
    im2 = ax2.pcolormesh(gx / 1000, gy / 1000, S, cmap="YlOrRd", vmin=0,
                         vmax=vmax, shading="auto", rasterized=True)
    ax2.set_aspect("equal")
    ax2.set_xlabel("X polar (km)")
    ax2.set_ylabel("Y polar (km)")
    ax2.set_title("Incerteza 1σ do dh/dt (jackknife sobre anos)",
                  fontweight="bold", loc="left", fontsize=11)
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8, extend="max", pad=0.02)
    cbar2.set_label("σ (m/ano)")
    ax2.grid(True, alpha=0.25, linewidth=0.4, linestyle="--")
    if fin.size:
        ax2.text(0.02, 0.02,
                 f"σ mediano {np.median(fin):.3f} m/ano\n"
                 f"escala cortada no p98 ({vmax:.2f})",
                 transform=ax2.transAxes, fontsize=8.5, va="bottom",
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#BBB", alpha=0.92))

    import json
    sel = cfg.paths.tables / "interp_selection.json"
    method = json.loads(sel.read_text()).get("winner", "?") if sel.exists() else "?"
    _footer(fig, cfg, f"| interp: {method} | σ = erro do nó (jackknife) + "
                      f"variância de interpolação")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out
