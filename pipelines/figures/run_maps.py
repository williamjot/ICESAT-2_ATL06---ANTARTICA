"""
pipelines/run_maps.py
=====================
Mapas cartográficos finais sobre base de relevo sombreado (REMA) com linha de
costa, linha de aterramento e frentes de calving datadas.

    -> outputs/figures/mapa_dhdt_basemap.png
    -> outputs/figures/mapa_basal_melt.png
    -> outputs/figures/mapa_velocidade.png
    -> outputs/figures/diagrama_balanco.png

Diferença para as figuras de diagnóstico: aqui o objetivo é comunicação. A base
de relevo dá contexto geográfico e as linhas de costa/aterramento vêm do mesmo
BedMachine que define as máscaras — a geometria desenhada é a geometria usada
no cálculo.

Uso: python pipelines/run_maps.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from thwaites import load_config
from thwaites.logging import setup_logging
from thwaites.viz.basemap import (draw_basemap, draw_calving_fronts,
                                  add_scale_bar)

DPI = 220


def _legend(ax, extra=None, loc="upper right"):
    h = [Line2D([], [], color="#1a1a1a", lw=1.1, label="linha de costa"),
         Line2D([], [], color="#8b0000", lw=1.0, ls="--",
                label="linha de aterramento"),
         Patch(facecolor="#b9d6ec", edgecolor="none", label="plataforma"),
         Line2D([], [], color="#00509e", lw=1.2, label="frente 2022 (IceLines)")]
    if extra:
        h += extra
    lg = ax.legend(handles=h, loc=loc, fontsize=7.5, framealpha=1.0,
                   facecolor="white", edgecolor="#999")
    lg.set_zorder(20)


def _finish(ax, title, sub=None):
    ax.set_xlabel("X polar (km)")
    ax.set_ylabel("Y polar (km)")
    ax.set_title(title + (f"\n{sub}" if sub else ""), fontweight="bold",
                 loc="left", fontsize=11.5)
    add_scale_bar(ax, 100)
    ax.grid(alpha=0.18, lw=0.4, ls="--", zorder=3)


def map_dhdt(cfg, log, figs):
    n = pd.read_parquet(cfg.paths.dhdt_dir / "dhdt_nodes_qc.parquet")
    x0, x1 = n.x.min() - 25e3, n.x.max() + 25e3
    y0, y1 = n.y.min() - 25e3, n.y.max() + 25e3

    fig, ax = plt.subplots(figsize=(10, 8.6), facecolor="white")
    draw_basemap(ax, cfg, x0, x1, y0, y1)
    draw_calving_fronts(ax, cfg, 2022.5)

    v = n["dhdt"].to_numpy()
    sc = ax.scatter(n.x / 1000, n.y / 1000, c=v, s=7,
                    cmap="RdBu", norm=TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=1.0),
                    edgecolor="none", zorder=7, rasterized=True)
    cb = plt.colorbar(sc, ax=ax, shrink=0.72, extend="both", pad=0.02)
    cb.set_label("dh/dt (m/ano)")
    _legend(ax)
    _finish(ax, "Taxa de mudança de elevação — gelo aterrado",
            f"ICESat-2 ATL06 · JJA 2019–2025 · {len(n):,} nós · "
            f"mediana {np.median(v):+.3f} m/ano")
    fig.text(0.01, 0.005, "relevo: REMA v2 | costa e linha de aterramento: "
             "BedMachine v4 | frentes: IceLines/Sentinel-1", fontsize=6.5,
             color="#555")
    p = figs / "mapa_dhdt_basemap.png"
    plt.savefig(p, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"figura -> {p}")


def map_basal_melt(cfg, log, figs):
    p_in = cfg.paths.dhdt_dir / "shelf_basal_melt.parquet"
    if not p_in.exists():
        log.warning("shelf_basal_melt.parquet ausente — mapa pulado.")
        return
    d = pd.read_parquet(p_in)
    if "reliable" in d.columns:
        d = d[d["reliable"].astype(bool)]
    # uma linha por parcela: mediana das janelas
    g = (d.groupby(["x_ref", "y_ref"])
           .agg(basal_melt=("basal_melt", "median"),
                n=("basal_melt", "size")).reset_index())
    g = g[np.isfinite(g["basal_melt"])]

    x0, x1 = g.x_ref.min() - 40e3, g.x_ref.max() + 40e3
    y0, y1 = g.y_ref.min() - 40e3, g.y_ref.max() + 40e3
    fig, ax = plt.subplots(figsize=(10, 8.6), facecolor="white")
    draw_basemap(ax, cfg, x0, x1, y0, y1)
    draw_calving_fronts(ax, cfg, 2022.5)

    v = g["basal_melt"].to_numpy()
    lim = float(np.nanpercentile(np.abs(v), 95))
    sc = ax.scatter(g.x_ref / 1000, g.y_ref / 1000, c=v, s=11,
                    cmap="RdYlBu_r", vmin=-lim, vmax=lim,
                    edgecolor="none", zorder=7, rasterized=True)
    cb = plt.colorbar(sc, ax=ax, shrink=0.72, extend="both", pad=0.02)
    cb.set_label("ṁ_b (m gelo/ano)   +derrete / −congela")
    _legend(ax)
    _finish(ax, "Derretimento basal — plataformas (Lagrangiano)",
            f"ṁ_b = a_s − DH/Dt − H·∇·v · {len(g):,} parcelas · "
            f"mediana {np.median(v):+.2f} m/ano")
    ax.text(0.015, 0.975, "PRELIMINAR — máscara estática,\n"
            "H de época 2015, FAC extrapolado após 2022",
            transform=ax.transAxes, va="top", fontsize=7.2, color="#8b0000",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#8b0000",
                      alpha=0.9), zorder=9)
    p = figs / "mapa_basal_melt.png"
    plt.savefig(p, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"figura -> {p}")


def map_velocity(cfg, log, figs):
    from netCDF4 import Dataset
    vp = cfg.paths.data_dir / "velocity_itslive_annual.nc"
    if not vp.exists():
        log.warning("ITS_LIVE ausente — mapa de velocidade pulado.")
        return
    D = 6
    with Dataset(vp) as f:
        gx = np.asarray(f["x"][::D], float)
        gy = np.asarray(f["y"][::D], float)
        vx = np.ma.filled(np.asarray(f["vx"][:, ::D, ::D], float), np.nan)
        vy = np.ma.filled(np.asarray(f["vy"][:, ::D, ::D], float), np.nan)
    if gy[0] > gy[-1]:
        gy, vx, vy = gy[::-1], vx[:, ::-1], vy[:, ::-1]
    sp = np.nanmedian(np.hypot(vx, vy), axis=0)

    n = pd.read_parquet(cfg.paths.dhdt_dir / "dhdt_nodes_qc.parquet")
    x0, x1 = n.x.min() - 25e3, n.x.max() + 25e3
    y0, y1 = n.y.min() - 25e3, n.y.max() + 25e3
    fig, ax = plt.subplots(figsize=(10, 8.6), facecolor="white")
    draw_basemap(ax, cfg, x0, x1, y0, y1, shelf_fill=False)

    from matplotlib.colors import LogNorm
    m = np.isfinite(sp) & (sp > 1)
    S = np.where(m, sp, np.nan)
    im = ax.pcolormesh(gx / 1000, gy / 1000, S, cmap="turbo",
                       norm=LogNorm(vmin=5, vmax=3000), alpha=0.82,
                       shading="auto", zorder=6, rasterized=True)
    cb = plt.colorbar(im, ax=ax, shrink=0.72, extend="both", pad=0.02)
    cb.set_label("velocidade superficial (m/ano, log)")
    draw_calving_fronts(ax, cfg, 2022.5)
    _legend(ax)
    _finish(ax, "Velocidade do gelo — ITS_LIVE v2 (média 2019–2025)",
            "componentes anuais vx, vy a 120 m · substitui o mosaico "
            "MEaSUREs 1996–2018")
    p = figs / "mapa_velocidade.png"
    plt.savefig(p, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"figura -> {p}")


def diagram_budget(cfg, log, figs):
    """Termos do balanço + série temporal — o resultado quantitativo."""
    bm_p = cfg.paths.tables / "basal_melt_report.json"
    mb_p = cfg.paths.tables / "mass_balance_raw_elevation.json"
    if not bm_p.exists():
        log.warning("basal_melt_report.json ausente — diagrama pulado.")
        return
    bm = json.loads(bm_p.read_text(encoding="utf-8"))
    mb = json.loads(mb_p.read_text(encoding="utf-8")) if mb_p.exists() else {}

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 5),
                                     facecolor="white",
                                     gridspec_kw={"width_ratios": [1, 1.15, 1]})

    # (1) termos do balanço basal
    t = bm["termos_medianos"]
    labs = ["a_s\n(SMB)", "−DH/Dt", "−H·∇·v", "= ṁ_b"]
    vals = [t["a_s (SMB)"], -t["DH/Dt"], -t["H*div(v)"],
            bm["basal_melt_mediana_m_ano"]]
    cols = ["#43a047", "#1e88e5", "#8e24aa", "#d81b60"]
    b = a1.bar(labs, vals, color=cols, width=0.62)
    a1.bar_label(b, fmt="%+.2f", fontsize=9, padding=2)
    a1.axhline(0, color="k", lw=1)
    a1.set_ylabel("m gelo/ano")
    a1.set_title("Balanço basal (mediana)", fontweight="bold", loc="left",
                 fontsize=11)
    a1.spines[["top", "right"]].set_visible(False)

    # (2) série temporal por janela
    pw = bm.get("por_janela", {})
    ks = sorted(pw)
    if ks:
        cen = [np.mean([float(x) for x in k.split("-")]) for k in ks]
        mbv = [pw[k]["basal_melt_mediana"] for k in ks]
        dh = [pw[k]["DHDt_mediana"] for k in ks]
        a2.plot(cen, mbv, "o-", color="#d81b60", lw=2, label="ṁ_b")
        a2.plot(cen, dh, "s--", color="#1e88e5", lw=1.6, label="DH/Dt")
        a2.axhline(0, color="k", lw=0.8, ls=":")
        a2.set_xlabel("centro da janela (ano)")
        a2.set_ylabel("m gelo/ano")
        a2.set_title("Evolução — janelas móveis de 3 anos", fontweight="bold",
                     loc="left", fontsize=11)
        a2.legend(fontsize=9)
        a2.grid(alpha=0.25, ls="--", lw=0.5)
        a2.spines[["top", "right"]].set_visible(False)

    # (3) resultado integrado do gelo aterrado
    a3.axis("off")
    rows = []
    if mb:
        rows += [("dM/dt", f"{mb.get('dMdt_Gt_yr', float('nan')):+.1f} "
                           f"± {mb.get('sigma_dMdt_Gt_yr_correlated', float('nan')):.1f} Gt/ano"),
                 ("nível do mar", f"{mb.get('sle_mm_yr', float('nan')):+.3f} "
                                  f"± {mb.get('sigma_sle_mm_yr_correlated', float('nan')):.3f} mm/ano"),
                 ("área", f"{mb.get('area_total_km2', float('nan')):,.0f} km²"),
                 ("dh/dt médio", f"{mb.get('dhdt_mean_m_yr', float('nan')):+.3f} m/ano")]
    rows += [("ṁ_b mediano", f"{bm['basal_melt_mediana_m_ano']:+.2f} m/ano"),
             ("parcelas", f"{bm['n_parcelas']:,}")]
    a3.text(0, 0.97, "Resultado integrado", fontsize=12.5, fontweight="bold",
            va="top")
    y = 0.83
    for k, v in rows:
        a3.text(0.0, y, k, fontsize=10, color="#444", va="top")
        a3.text(1.0, y, v, fontsize=11, fontweight="bold", ha="right", va="top")
        y -= 0.115
    a3.text(0, 0.06,
            "Gelo aterrado: Euleriano, ∂h/∂t.\n"
            "Plataforma: Lagrangiano, ṁ_b = a_s − DH/Dt − H·∇·v\n"
            "(H·∇·v, não ∇·(H·v): a advecção já está no seguimento da parcela).",
            fontsize=7.6, color="#666", va="bottom")

    fig.suptitle("Amundsen Sea Embayment — ICESat-2 ATL06, 2019–2025",
                 fontweight="bold", x=0.01, ha="left", fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    p = figs / "diagrama_balanco.png"
    plt.savefig(p, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"figura -> {p}")


def _shelf_nodes(cfg):
    """
    dh/dt das parcelas de plataforma, mediana das janelas móveis.

    Grandeza DIFERENTE da do gelo aterrado: aqui é DH/Dt lagrangiano (segue a
    parcela), com maré CATS2008 e DAC removidos e datum ortométrico; lá é
    ∂h/∂t euleriano num nó fixo, sem maré aplicada (gating). Por isso os dois
    NUNCA compartilham a mesma barra de cor.
    """
    import pandas as pd
    p = cfg.paths.dhdt_dir / "shelf_lagrangian_windows.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    if "reliable" in d.columns:
        d = d[d["reliable"].astype(bool)]
    g = (d.groupby(["x_ref", "y_ref"])
           .agg(dhdt=("dhdt_lagrangian", "median"),
                n=("dhdt_lagrangian", "size")).reset_index())
    return g[np.isfinite(g["dhdt"])]


def map_two_panel(cfg, log, figs):
    """Painéis lado a lado: gelo aterrado e plataforma, escalas independentes."""
    n = pd.read_parquet(cfg.paths.dhdt_dir / "dhdt_nodes_qc.parquet")
    sh = _shelf_nodes(cfg)
    if sh is None or sh.empty:
        log.warning("sem parcelas de plataforma — painel duplo pulado.")
        return
    x0 = min(n.x.min(), sh.x_ref.min()) - 25e3
    x1 = max(n.x.max(), sh.x_ref.max()) + 25e3
    y0 = min(n.y.min(), sh.y_ref.min()) - 25e3
    y1 = max(n.y.max(), sh.y_ref.max()) + 25e3

    fig, axes = plt.subplots(1, 2, figsize=(17, 8.2), facecolor="white")
    for ax, (dat, xc, yc, cmap, norm, lab, ttl, sub) in zip(axes, [
        (n, "x", "y", "RdBu",
         TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=1.0),
         "∂h/∂t (m/ano)", "Gelo aterrado — Euleriano",
         f"{len(n):,} nós · mediana {np.median(n['dhdt']):+.3f} m/ano · "
         f"maré NÃO aplicada"),
        (sh, "x_ref", "y_ref", "PuOr",
         TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=1.5),
         "DH/Dt (m/ano)", "Plataforma — Lagrangiano",
         f"{len(sh):,} parcelas · mediana {np.median(sh['dhdt']):+.3f} m/ano · "
         f"maré CATS2008 + DAC removidos")]):
        draw_basemap(ax, cfg, x0, x1, y0, y1)
        draw_calving_fronts(ax, cfg, 2022.5)
        sc = ax.scatter(dat[xc] / 1000, dat[yc] / 1000, c=dat["dhdt"], s=6,
                        cmap=cmap, norm=norm, edgecolor="none", zorder=7,
                        rasterized=True)
        cb = plt.colorbar(sc, ax=ax, shrink=0.7, extend="both", pad=0.02)
        cb.set_label(lab)
        _finish(ax, ttl, sub)
    _legend(axes[0])
    fig.suptitle("Amundsen Sea Embayment — dois domínios, duas grandezas",
                 fontweight="bold", x=0.01, ha="left", fontsize=13)
    fig.text(0.01, 0.005,
             "As escalas são INDEPENDENTES de propósito: ∂h/∂t euleriano e "
             "DH/Dt lagrangiano medem processos distintos e não são comparáveis "
             "valor a valor.", fontsize=7, color="#8b0000")
    plt.tight_layout(rect=(0, 0.02, 1, 0.95))
    p = figs / "mapa_dois_paineis.png"
    plt.savefig(p, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"figura -> {p}")


def map_combined(cfg, log, figs):
    """Mapa único: círculos = aterrado, losangos = plataforma, 2 barras de cor."""
    n = pd.read_parquet(cfg.paths.dhdt_dir / "dhdt_nodes_qc.parquet")
    sh = _shelf_nodes(cfg)
    if sh is None or sh.empty:
        log.warning("sem parcelas de plataforma — mapa combinado pulado.")
        return
    x0 = min(n.x.min(), sh.x_ref.min()) - 25e3
    x1 = max(n.x.max(), sh.x_ref.max()) + 25e3
    y0 = min(n.y.min(), sh.y_ref.min()) - 25e3
    y1 = max(n.y.max(), sh.y_ref.max()) + 25e3

    fig, ax = plt.subplots(figsize=(11.5, 9), facecolor="white")
    draw_basemap(ax, cfg, x0, x1, y0, y1)
    draw_calving_fronts(ax, cfg, 2022.5)

    s1 = ax.scatter(n.x / 1000, n.y / 1000, c=n["dhdt"], s=6, marker="o",
                    cmap="RdBu", norm=TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=1.0),
                    edgecolor="none", zorder=7, rasterized=True)
    s2 = ax.scatter(sh.x_ref / 1000, sh.y_ref / 1000, c=sh["dhdt"], s=5,
                    marker="D", cmap="PuOr",
                    norm=TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=1.5),
                    edgecolor="none", zorder=8, rasterized=True)

    cb1 = plt.colorbar(s1, ax=ax, shrink=0.42, extend="both", pad=0.02,
                       anchor=(0.0, 0.58))
    cb1.set_label("∂h/∂t aterrado (m/ano)", fontsize=8.5)
    cb1.ax.tick_params(labelsize=7.5)
    cb2 = plt.colorbar(s2, ax=ax, shrink=0.42, extend="both", pad=0.09,
                       anchor=(0.0, 0.0))
    cb2.set_label("DH/Dt plataforma (m/ano)", fontsize=8.5)
    cb2.ax.tick_params(labelsize=7.5)

    _legend(ax, extra=[
        Line2D([], [], marker="o", ls="", color="#777", ms=5,
               label=f"nó aterrado (Euleriano) — {len(n):,}"),
        Line2D([], [], marker="D", ls="", color="#777", ms=5,
               label=f"parcela de plataforma (Lagrangiano) — {len(sh):,}")],
        loc="lower right")
    _finish(ax, "Mudança de elevação — gelo aterrado e plataforma",
            "ICESat-2 ATL06 · 2019–2025 · símbolos e escalas distintos porque "
            "as grandezas são distintas")
    fig.text(0.01, 0.005,
             "Círculos: ∂h/∂t euleriano, maré não aplicada (gelo aterrado). "
             "Losangos: DH/Dt lagrangiano, maré CATS2008 + DAC removidos, datum "
             "ortométrico (plataforma).", fontsize=7, color="#555")
    p = figs / "mapa_combinado.png"
    plt.savefig(p, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"figura -> {p}")


def main():
    ap = argparse.ArgumentParser(description="Mapas finais com basemap.")
    ap.add_argument("--profile", default=None)
    args = ap.parse_args()
    cfg = load_config(args.profile)
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level, run_name="maps")
    figs = cfg.paths.figures
    figs.mkdir(parents=True, exist_ok=True)
    for fn in (map_dhdt, map_two_panel, map_combined, map_basal_melt,
               map_velocity, diagram_budget):
        try:
            fn(cfg, log, figs)
        except Exception as e:
            log.error(f"{fn.__name__}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
