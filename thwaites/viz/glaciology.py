"""
thwaites.viz.glaciology
=======================
Diagramas glaciológicos: derretimento basal, dh/dt × velocidade do gelo e o
diagrama de orçamento de massa.

Existe porque as etapas run_flux / run_dynamics / run_firn produziam apenas JSON
e NetCDF — os números existiam, mas não havia como inspecionar sua estrutura
espacial, que é justamente onde se vê se o resultado é físico ou artefato.

Compartilha estilo e rótulos com thwaites.viz.figures (mesma paleta, mesmo
rodapé) para que as figuras do artigo sejam visualmente coerentes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from thwaites.config import Config
from thwaites.viz.figures import _CMAP, _DPI, _dhdt_norm, _footer, _region, _season


def fig_basal_melt_map(nc_path: Path, cfg: Config, out: Path) -> Path:
    """
    Três painéis do balanço de massa local sobre gelo flutuante: divergência de
    fluxo ∇·(H·v), dh/dt amplificado hidrostaticamente e o derretimento basal.

    Só é fisicamente válido sobre gelo flutuante — a amplificação hidrostática
    pressupõe a plataforma em flutuação livre. O rodapé declara essa restrição
    em vez de deixá-la implícita.
    """
    from netCDF4 import Dataset

    with Dataset(nc_path) as d:
        x = np.asarray(d["x"][:], dtype=float) / 1000.0
        y = np.asarray(d["y"][:], dtype=float) / 1000.0
        get = lambda k: np.ma.filled(np.asarray(d[k][:], dtype=float), np.nan)
        fdiv, dHdt, melt = get("flux_divergence"), get("dHdt_hydrostatic"), get("basal_melt")

    panels = [
        (fdiv, "div(H*v)  divergencia de fluxo", "m gelo/ano", "PuOr_r"),
        (dHdt, "dH/dt  (dh/dt amplificado)", "m gelo/ano", _CMAP),
        (melt, "m_b  derretimento basal", "m gelo/ano", "magma_r"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6), facecolor="white")
    for ax, (Z, title, unit, cmap) in zip(axes, panels):
        fin = Z[np.isfinite(Z)]
        if fin.size == 0:
            ax.text(0.5, 0.5, "sem dado valido", ha="center", transform=ax.transAxes)
            ax.set_title(title, fontsize=10, loc="left")
            continue
        # Escala robusta (percentis 2/98): os extremos da zona de aterramento
        # saturariam a paleta e apagariam todo o resto do padrão espacial.
        lo, hi = np.percentile(fin, [2, 98])
        if cmap == _CMAP:
            nrm = TwoSlopeNorm(vmin=min(lo, -1e-3), vcenter=0.0, vmax=max(hi, 1e-3))
            im = ax.pcolormesh(x, y, Z, cmap=cmap, norm=nrm, shading="auto",
                               rasterized=True)
        else:
            im = ax.pcolormesh(x, y, Z, cmap=cmap, vmin=lo, vmax=hi,
                               shading="auto", rasterized=True)
        ax.set_aspect("equal")
        ax.set_xlabel("X polar (km)")
        ax.set_title(f"{title}\nmediana {np.median(fin):+.2f} {unit}",
                     fontsize=10, loc="left")
        ax.grid(True, alpha=0.2, lw=0.4, ls="--")
        cb = plt.colorbar(im, ax=ax, shrink=0.85, extend="both", pad=0.02)
        cb.set_label(unit, fontsize=8)
        cb.ax.tick_params(labelsize=7)
    axes[0].set_ylabel("Y polar (km)")
    fig.suptitle(f"Balanco de massa local sobre gelo flutuante - {_region(cfg)}",
                 fontweight="bold", x=0.01, ha="left")
    _footer(fig, cfg, "| dH/dt + div(H*v) = SMB - m_b | valido so sobre gelo "
                      "flutuante | velocidade MEaSUREs 1996-2018 vs dh/dt "
                      f"{cfg.temporal.year_start}-{cfg.temporal.year_end}")
    plt.tight_layout(rect=(0, 0.02, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def sample_velocity_at(nodes_df: pd.DataFrame, vel_path: Path, cfg: Config):
    """
    Interpola |v| do mosaico MEaSUREs nas posições dos nós.

    Devolve (vx, vy, speed) alinhados às linhas de `nodes_df`; NaN fora da
    cobertura do mosaico (nunca extrapola).
    """
    from netCDF4 import Dataset
    from scipy.interpolate import RegularGridInterpolator

    with Dataset(vel_path) as d:
        vx_name = cfg.velocity.vx_var if cfg.velocity.vx_var in d.variables else "VX"
        vy_name = cfg.velocity.vy_var if cfg.velocity.vy_var in d.variables else "VY"
        vxg = np.ma.filled(np.asarray(d[vx_name][:], dtype=float), np.nan)
        vyg = np.ma.filled(np.asarray(d[vy_name][:], dtype=float), np.nan)
        gx = np.asarray(d["x"][:], dtype=float)
        gy = np.asarray(d["y"][:], dtype=float)

    # RegularGridInterpolator exige eixos estritamente crescentes; o MEaSUREs
    # vem com y decrescente (convenção de raster, norte no topo).
    if gy[0] > gy[-1]:
        gy, vxg, vyg = gy[::-1], vxg[::-1], vyg[::-1]
    if gx[0] > gx[-1]:
        gx, vxg, vyg = gx[::-1], vxg[:, ::-1], vyg[:, ::-1]

    pts = np.c_[nodes_df["y"].to_numpy(), nodes_df["x"].to_numpy()]
    kw = dict(bounds_error=False, fill_value=np.nan, method="linear")
    vx = RegularGridInterpolator((gy, gx), vxg, **kw)(pts)
    vy = RegularGridInterpolator((gy, gx), vyg, **kw)(pts)
    return vx, vy, np.hypot(vx, vy)


def fig_dhdt_vs_velocity(nodes_df: pd.DataFrame, vel_path: Path, cfg: Config,
                         out: Path, fast_m_yr: float = 300.0,
                         stable_m_yr: float = 0.1) -> Path:
    """
    dh/dt por nó contra a velocidade superficial do gelo.

    Atende ao item 2 do backlog: nós classificados como "estáveis" (|dh/dt|
    pequeno) mas situados em gelo rápido são candidatos a mudança dinâmica que o
    sinal de elevação ainda não revela — o segundo painel os localiza no mapa.

    Descasamento temporal declarado: o mosaico de velocidade é de 1996-2018 e o
    dh/dt de 2019-2025. A leitura é de contexto espacial, não de correlação
    temporal — daí os limiares serem parâmetros explícitos, não constantes.
    """
    _, _, speed = sample_velocity_at(nodes_df, vel_path, cfg)
    dhdt = nodes_df["dhdt"].to_numpy()
    m = np.isfinite(speed) & np.isfinite(dhdt) & (speed > 0)
    if m.sum() < 50:
        raise ValueError(f"apenas {m.sum()} nós com velocidade válida — "
                         f"o recorte de velocidade cobre a ROI?")
    sp, dh = speed[m], dhdt[m]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.8), facecolor="white",
                                  gridspec_kw={"width_ratios": [1.35, 1]})
    ax.scatter(sp, dh, s=6, alpha=0.28, c=dh, cmap=_CMAP, norm=_dhdt_norm(),
               edgecolor="none", rasterized=True)
    ax.set_xscale("log")
    ax.axhline(0, color="k", lw=1, ls="--", alpha=0.5)

    # Mediana por faixa de velocidade: é o padrão que a nuvem de pontos esconde.
    edges = np.logspace(np.log10(max(sp.min(), 1.0)), np.log10(sp.max()), 16)
    cx, cmed, clo, chi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        s = (sp >= a) & (sp < b)
        if s.sum() >= 20:
            cx.append(np.sqrt(a * b)); cmed.append(np.median(dh[s]))
            clo.append(np.percentile(dh[s], 25)); chi.append(np.percentile(dh[s], 75))
    if cx:
        ax.plot(cx, cmed, "-", color="#111111", lw=2, label="mediana por faixa")
        ax.fill_between(cx, clo, chi, color="#111111", alpha=0.12, label="IQR")
    ax.axvline(fast_m_yr, color="#B71C1C", lw=1.2, ls=":")
    ax.set_xlabel("velocidade superficial |v| (m/ano, escala log)")
    ax.set_ylabel("dh/dt (m/ano)")
    ax.set_ylim(np.percentile(dh, 0.5), np.percentile(dh, 99.5))
    ax.set_title(f"dh/dt x velocidade do gelo - {_region(cfg)}",
                 fontweight="bold", loc="left", fontsize=11)
    ax.legend(loc="lower left", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    watch = (np.abs(dh) < stable_m_yr) & (sp > fast_m_yr)
    xs = nodes_df["x"].to_numpy()[m] / 1000
    ys = nodes_df["y"].to_numpy()[m] / 1000
    ax2.scatter(xs, ys, s=4, c="#DDDDDD", edgecolor="none", rasterized=True)
    ax2.scatter(xs[watch], ys[watch], s=16, c="#B71C1C", edgecolor="none",
                label=f"|dh/dt|<{stable_m_yr} e |v|>{fast_m_yr:.0f} m/ano: "
                      f"{int(watch.sum())} nos")
    ax2.set_aspect("equal")
    ax2.set_xlabel("X polar (km)"); ax2.set_ylabel("Y polar (km)")
    ax2.set_title("Nos a vigiar: estaveis em elevacao, porem em gelo rapido",
                  fontsize=10, loc="left")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.2, lw=0.4, ls="--")

    _footer(fig, cfg, f"| n={int(m.sum()):,} nos | velocidade "
                      f"{cfg.velocity.short_name} (1996-2018) vs dh/dt "
                      f"{cfg.temporal.year_start}-{cfg.temporal.year_end}: "
                      f"sem correspondencia temporal")
    plt.tight_layout(rect=(0, 0.02, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def fig_mass_budget(summaries: dict, cfg: Config, out: Path) -> Path:
    """
    Orçamento de massa: dh/dt observado decomposto nas parcelas conhecidas (ar
    no firn, SMB) e o resultado integrado em Gt/ano e nível do mar.

    `summaries` reúne os JSON gravados pelas etapas (chaves "mass_balance",
    "firn", "flux"). Parcelas ausentes aparecem como LACUNA DECLARADA — nunca
    como zero, que seria afirmar uma física que não medimos.
    """
    mb = summaries.get("mass_balance") or {}
    firn = summaries.get("firn") or {}
    flux = summaries.get("flux") or {}

    def g(d, *keys):
        for k in keys:
            if isinstance(d, dict) and d.get(k) is not None:
                return d[k]
        return None

    # Nomes conforme gravados por run_mass_balance / run_flux / run_firn.
    dhdt_obs = g(mb, "dhdt_mean_m_yr", "dhdt_median_m_yr")
    dfac = g(firn, "dfac_dt_median_m_yr", "fac_rate_median_m_yr",
             "dFAC_dt_median_m_yr", "dfac_dt_mean_m_yr")
    smb = g(flux, "smb_median_m_ice_yr")
    gt = g(mb, "dMdt_Gt_yr", "mass_gt_yr")
    # Duas barras de erro são gravadas: a "independent" trata os nós como
    # independentes e é otimista; a "correlated" usa o comprimento de correlação
    # espacial. Usamos a CORRELACIONADA — reportar a outra subestimaria a
    # incerteza do balanço, que é justamente a limitação conhecida do projeto.
    gt_err = g(mb, "sigma_dMdt_Gt_yr_correlated", "sigma_dMdt_Gt_yr_independent",
               "mass_gt_yr_err")
    err_kind = ("correlacionado"
                if mb.get("sigma_dMdt_Gt_yr_correlated") is not None
                else "independente (OTIMISTA)")
    sle = g(mb, "sle_mm_yr", "sea_level_mm_yr")
    sle_err = g(mb, "sigma_sle_mm_yr_correlated", "sigma_sle_mm_yr_independent")
    melt = g(flux, "melt_median_m_yr")
    area = g(mb, "area_total_km2")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4), facecolor="white",
                                  gridspec_kw={"width_ratios": [1.15, 1]})

    items, vals, cols = [], [], []
    if dhdt_obs is not None:
        items.append("dh/dt\nobservado"); vals.append(dhdt_obs); cols.append("#37474F")
    if dfac is not None:
        items.append("dFAC/dt\n(ar no firn)"); vals.append(dfac); cols.append("#1976D2")
    if smb is not None:
        items.append("SMB\n(m gelo/ano)"); vals.append(smb); cols.append("#43A047")
    if dhdt_obs is not None and dfac is not None:
        items.append("dh_gelo/dt\n(obs - dFAC)")
        vals.append(dhdt_obs - dfac); cols.append("#6A1B9A")
    if items:
        b = ax.bar(items, vals, color=cols, width=0.62)
        ax.bar_label(b, fmt="%+.3f", fontsize=9, padding=2)
        ax.axhline(0, color="k", lw=1)
        ax.set_ylabel("m/ano")
        lim = (max(abs(v) for v in vals) or 1.0) * 1.45
        ax.set_ylim(-lim, lim)
    else:
        ax.text(0.5, 0.5, "nenhuma parcela disponivel", ha="center",
                transform=ax.transAxes)
    ax.set_title(f"Parcelas do balanco - {_region(cfg)}", fontweight="bold",
                 loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    ax2.axis("off")
    rows = []
    if gt is not None:
        e = f" +/- {abs(gt_err):.1f}" if gt_err is not None else ""
        rows.append(("dM/dt", f"{gt:+.1f}{e} Gt/ano"))
    if sle is not None:
        e = f" +/- {abs(sle_err):.3f}" if sle_err is not None else ""
        rows.append(("contribuicao ao nivel do mar", f"{sle:+.3f}{e} mm/ano"))
    if melt is not None:
        rows.append(("derretimento basal (mediana)", f"{melt:+.2f} m/ano"))
    if dhdt_obs is not None:
        rows.append(("dh/dt medio", f"{dhdt_obs:+.3f} m/ano"))
    if area is not None:
        rows.append(("area integrada", f"{area:,.0f} km2"))

    missing = [n for n, v in (("dFAC/dt (firn)", dfac), ("SMB", smb),
                              ("derretimento basal", melt)) if v is None]
    y = 0.94
    ax2.text(0.0, y, "Balanco de massa integrado", fontsize=12,
             fontweight="bold", va="top")
    y -= 0.13
    for k, v in rows:
        ax2.text(0.0, y, k, fontsize=10, va="top", color="#444444")
        ax2.text(1.0, y, v, fontsize=11, va="top", ha="right", fontweight="bold")
        y -= 0.105
    if missing:
        # Lacuna declarada: sem isto o leitor somaria as barras como se o
        # orcamento estivesse fechado.
        ax2.text(0.0, y - 0.04,
                 "Parcelas NAO incluidas (lacuna, nao zero):\n  - " +
                 "\n  - ".join(missing), fontsize=8.5, va="top", color="#B71C1C")
    ax2.text(0.0, 0.02,
             "Sinal: negativo = perda de massa / afinamento.\n"
             f"Barra de erro: sigma {err_kind}, propagado do dh/dt por no\n"
             "(jackknife sobre anos + comprimento de correlacao espacial).",
             fontsize=8, va="bottom", color="#666666")

    _footer(fig, cfg)
    plt.tight_layout(rect=(0, 0.02, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out
