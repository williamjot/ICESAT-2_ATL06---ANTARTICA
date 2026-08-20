"""
pipelines/run_atl15_validation.py
=================================
Validação EXTERNA do nosso dh/dt contra o ATL15 — produto oficial de mudança de
elevação em grade da própria missão ICESat-2.

    data/ATL15_*_01km_*.nc + data/dhdt/dhdt_nodes_qc.parquet
        -> outputs/tables/atl15_validation.json
        -> outputs/figures/atl15_validation.png

Por que isto importa
--------------------
É a única validação verdadeiramente externa disponível ao projeto. Os
crossovers internos partem do MESMO ATL06, das mesmas correções e
da mesma máscara — medem consistência interna de método, não acurácia. O ATL15
é processado de forma independente pela equipe da missão (ajuste espaço-temporal
próprio, correções próprias, grade própria).

Diferenças metodológicas que NÃO são erro — declarar ao interpretar
-------------------------------------------------------------------
1. JANELA SAZONAL: nosso produto usa apenas JJA (inverno austral); o ATL15 usa
   o ano inteiro. Se houver ciclo sazonal de elevação, os dois divergem por
   razão física, não por erro de um deles.
2. MÁSCARA: restringimos a gelo aterrado com buffers; o ATL15 cobre todo gelo.
   A comparação é feita apenas onde o NOSSO domínio é válido.
3. CORREÇÕES: aplicamos CATS2008 e slope por REMA; o ATL15 tem o seu próprio
   tratamento.
4. JANELA TEMPORAL: usa-se o lag mais longo do ATL15 (24 trimestres = 6 anos) na
   época cujo centro mais se aproxima do centro do nosso período.

Concordância aqui é evidência de acurácia; discordância localiza onde investigar.

Uso: python pipelines/run_atl15_validation.py [--lag dhdt_lag24]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thwaites import load_config
from thwaites.logging import setup_logging


def load_atl15(path: Path, group: str, target_year: float, log):
    """
    Lê dh/dt e sua incerteza do ATL15 na época mais próxima de `target_year`.

    ATENÇÃO — NÃO usar `dhdt_lag24`: verificação empírica mostrou que ele
    difere de (h[k+24]-h[k])/6anos calculado a partir de `delta_h` do MESMO
    arquivo por um fator EXATAMENTE 3, com correlação 1,0000. A convenção de
    lag foi confirmada em `dhdt_lag8`, que reproduz (h[k+8]-h[k])/2anos como
    esperado; o lag24 não reconcilia com nenhum par de épocas. Sem entender a
    definição, o campo não é utilizável como referência.

    A função `atl15_trend` abaixo é a via preferida: ajusta a tendência de
    `delta_h` sobre exatamente o mesmo intervalo do nosso produto, o que torna
    a comparação transparente e reprodutível.
    """
    import h5py

    with h5py.File(path, "r") as f:
        if group not in f:
            raise KeyError(f"grupo {group} ausente. Disponíveis: {list(f.keys())}")
        g = f[group]
        t_days = np.asarray(g["time"][:], dtype=float)
        t_year = 2018.0 + t_days / 365.25
        k = int(np.argmin(np.abs(t_year - target_year)))
        log.info(f"{group}: {len(t_year)} épocas ({t_year.min():.2f}–"
                 f"{t_year.max():.2f}) -> escolhida {t_year[k]:.2f} "
                 f"(alvo {target_year:.2f})")

        x = np.asarray(g["x"][:], dtype=float)
        y = np.asarray(g["y"][:], dtype=float)
        dhdt = np.asarray(g["dhdt"][k, :, :], dtype=np.float64)
        sig = np.asarray(g["dhdt_sigma"][k, :, :], dtype=np.float64)

    # _FillValue do produto é ~3.4e38; qualquer coisa dessa ordem é inválida
    for a in (dhdt, sig):
        a[np.abs(a) > 1e30] = np.nan
    return x, y, dhdt, sig, float(t_year[k])


def atl15_trend(path: Path, px, py, t0: float, t1: float, log,
                winter_only: bool = False):
    """
    Tendência linear de `delta_h` do ATL15 nas posições dadas, no intervalo
    [t0, t1].

    É a referência de validação: ajusta a MESMA grandeza que estimamos (uma
    taxa por regressão sobre épocas), no MESMO período, a partir da série
    trimestral de altura do produto oficial.

    `winter_only` restringe às épocas de inverno austral, permitindo medir
    dentro do próprio ATL15 quanto a janela sazonal JJA enviesa a taxa — sem
    precisar reprocessar nosso pipeline com dados que não temos em disco.
    """
    import h5py

    with h5py.File(path, "r") as f:
        g = f["delta_h"]
        ty = 2018.0 + np.asarray(g["time"][:], dtype=float) / 365.25
        x = np.asarray(g["x"][:], dtype=float)
        y = np.asarray(g["y"][:], dtype=float)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        j = np.rint((np.asarray(px) - x[0]) / dx).astype(np.int64)
        i = np.rint((np.asarray(py) - y[0]) / dy).astype(np.int64)
        ok = (j >= 0) & (j < len(x)) & (i >= 0) & (i < len(y))
        ii, jj = i[ok], j[ok]

        sel = (ty >= t0) & (ty <= t1)
        if winter_only:
            # inverno austral: fração do ano entre ~jun e ~ago
            frac = ty % 1.0
            sel &= (frac >= 0.40) & (frac <= 0.70)
        idx = np.where(sel)[0]
        log.info(f"delta_h: {len(idx)} épocas em [{t0:.2f},{t1:.2f}]"
                 f"{' (só inverno)' if winter_only else ''}")

        H = np.full((len(idx), len(ii)), np.nan)
        for e, k in enumerate(idx):
            a = np.asarray(g["delta_h"][k], dtype=float)
            a[np.abs(a) > 1e30] = np.nan
            H[e, :] = a[ii, jj]

    t = ty[idx]
    out = np.full(len(px), np.nan)
    vals = np.full(len(ii), np.nan)
    for c in range(H.shape[1]):
        v = H[:, c]
        m = np.isfinite(v)
        if m.sum() >= 4:
            vals[c] = np.polyfit(t[m], v[m], 1)[0]
    out[ok] = vals
    return out


def sample_grid(x, y, field, px, py):
    """Amostra o campo do ATL15 nas posições (px, py) por vizinho mais próximo."""
    if y[0] > y[-1]:
        y = y[::-1]
        field = field[::-1, :]
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    j = np.rint((np.asarray(px) - x[0]) / dx).astype(np.int64)
    i = np.rint((np.asarray(py) - y[0]) / dy).astype(np.int64)
    ok = (j >= 0) & (j < len(x)) & (i >= 0) & (i < len(y))
    out = np.full(len(px), np.nan)
    out[ok] = field[i[ok], j[ok]]
    return out


def main():
    ap = argparse.ArgumentParser(description="Valida dh/dt contra o ATL15.")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--lag", default="dhdt_lag24",
                    help="grupo do ATL15 (lag em trimestres; 24 = 6 anos)")
    ap.add_argument("--nodes", default="dhdt_nodes_qc.parquet")
    args = ap.parse_args()

    cfg = load_config(args.profile)
    log = setup_logging(cfg.paths.logs, level=cfg.logging.level, run_name="atl15")

    cands = sorted(cfg.paths.data_dir.glob("ATL15_*_01km_*.nc"))
    if not cands:
        raise FileNotFoundError("ATL15 de 1 km não encontrado em data/.")
    atl15_path = cands[0]

    nodes = pd.read_parquet(cfg.paths.dhdt_dir / args.nodes)
    t_center = 0.5 * (cfg.temporal.year_start + cfg.temporal.year_end + 1)
    log.info(f"nós: {len(nodes):,} | centro do nosso período: {t_center:.2f}")

    # Intervalo REAL coberto pelos nossos dados, para casar a janela.
    t0, t1 = 2019.4, 2025.7
    px = nodes["x"].to_numpy()
    py = nodes["y"].to_numpy()

    ours = nodes["dhdt"].to_numpy(dtype=float)
    ours_sig = nodes["dhdt_err"].to_numpy(dtype=float)
    theirs = atl15_trend(atl15_path, px, py, t0, t1, log)
    theirs_winter = atl15_trend(atl15_path, px, py, t0, t1, log, winter_only=True)
    theirs_sig = np.full(len(ours), np.nan)   # sem sigma direto para a tendência
    epoch = 0.5 * (t0 + t1)

    m = np.isfinite(ours) & np.isfinite(theirs)
    n = int(m.sum())
    if n < 50:
        raise SystemExit(f"apenas {n} nós pareados — verifique a extensão do ATL15.")
    a, b = ours[m], theirs[m]
    diff = a - b

    # z combinando as duas incertezas: se ambas estiverem corretas, |z| ~ 1
    with np.errstate(invalid="ignore", divide="ignore"):
        z = diff / np.sqrt(ours_sig[m] ** 2 + theirs_sig[m] ** 2)
    zf = z[np.isfinite(z)]

    r = float(np.corrcoef(a, b)[0, 1])
    slope, icept = np.polyfit(b, a, 1)

    rep = {
        "atl15_file": atl15_path.name,
        "atl15_group": args.lag,
        "atl15_epoch_year": epoch,
        "nosso_periodo": [cfg.temporal.year_start, cfg.temporal.year_end],
        "n_nos_pareados": n,
        "nosso_dhdt_mediana": float(np.median(a)),
        "atl15_dhdt_mediana": float(np.median(b)),
        "diferenca_mediana": float(np.median(diff)),
        "diferenca_media": float(np.mean(diff)),
        "diferenca_rms": float(np.sqrt(np.mean(diff ** 2))),
        "diferenca_mad": float(np.median(np.abs(diff - np.median(diff)))),
        "correlacao_r": r,
        "regressao_nosso_vs_atl15": {"slope": float(slope), "intercept": float(icept)},
        "z_mediano_abs": float(np.median(np.abs(zf))) if zf.size else None,
        "frac_dentro_2sigma": float(np.mean(np.abs(zf) < 2)) if zf.size else None,
        "atl15_winter_only_mediana": float(np.nanmedian(theirs_winter)),
        "vies_sazonal_JJA_sobre_anual": float(
            np.nanmedian(theirs_winter) / np.nanmedian(theirs)),
        "nota_dhdt_lag24": (
            "NAO utilizado: difere de (h[k+24]-h[k])/6anos derivado de delta_h "
            "do mesmo arquivo por fator exatamente 3 (correlacao 1.0000). "
            "A convencao de lag foi confirmada em dhdt_lag8. Sem definicao "
            "verificavel, o campo nao serve de referencia."),
        "diferencas_metodologicas_declaradas": [
            f"janela sazonal: nosso = {cfg.season.name.upper()} apenas; "
            f"ATL15 = ano inteiro",
            "máscara: nosso = gelo aterrado + buffers; ATL15 = todo gelo",
            "correções: nosso aplica CATS2008 e slope por REMA; ATL15 usa "
            "tratamento próprio",
            f"janela temporal: ATL15 {args.lag} (lag em trimestres) centrado em "
            f"{epoch:.2f}",
        ],
    }
    cfg.paths.tables.mkdir(parents=True, exist_ok=True)
    (cfg.paths.tables / "atl15_validation.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info(f"pareados {n:,} nós | nosso {np.median(a):+.4f} vs "
             f"ATL15 {np.median(b):+.4f} m/ano")
    log.info(f"diferença: mediana {np.median(diff):+.4f} | RMS "
             f"{np.sqrt(np.mean(diff**2)):.4f} m/ano | r={r:.4f} | slope={slope:.3f}")
    if zf.size:
        log.info(f"|z| mediano {np.median(np.abs(zf)):.2f} | dentro de 2σ: "
                 f"{100*np.mean(np.abs(zf)<2):.1f}%")

    # ------------------------------------------------------------- figura
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from thwaites.viz.figures import _CMAP, _DPI, _dhdt_norm, _footer, _region

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), facecolor="white")
    ax = axes[0]
    lim = np.nanpercentile(np.abs(np.r_[a, b]), 99)
    ax.scatter(b, a, s=4, alpha=0.25, c="#2171b5", edgecolor="none", rasterized=True)
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="1:1")
    xs = np.linspace(-lim, lim, 10)
    ax.plot(xs, slope * xs + icept, "-", color="#d94801", lw=1.5,
            label=f"ajuste: {slope:.2f}x{icept:+.3f}")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel("ATL15 dh/dt (m/ano)"); ax.set_ylabel("nosso dh/dt (m/ano)")
    ax.set_title(f"Nosso vs ATL15 — n={n:,}, r={r:.3f}", fontweight="bold",
                 loc="left", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.2, lw=0.4, ls="--")

    ax = axes[1]
    ax.hist(diff, bins=80, color="#6a51a3")
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.axvline(np.median(diff), color="#d94801", lw=2,
               label=f"mediana {np.median(diff):+.3f}")
    ax.set_xlabel("nosso − ATL15 (m/ano)"); ax.set_ylabel("nº de nós")
    ax.set_title(f"Diferença — RMS {np.sqrt(np.mean(diff**2)):.3f} m/ano",
                 fontweight="bold", loc="left", fontsize=11)
    ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    sc = ax.scatter(nodes["x"].to_numpy()[m] / 1000, nodes["y"].to_numpy()[m] / 1000,
                    c=diff, s=6, cmap="RdBu_r",
                    vmin=-np.nanpercentile(np.abs(diff), 95),
                    vmax=np.nanpercentile(np.abs(diff), 95),
                    edgecolor="none", rasterized=True)
    ax.set_aspect("equal")
    ax.set_xlabel("X polar (km)"); ax.set_ylabel("Y polar (km)")
    ax.set_title("Onde discordam", fontweight="bold", loc="left", fontsize=11)
    cb = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("nosso − ATL15 (m/ano)")
    ax.grid(alpha=0.2, lw=0.4, ls="--")

    _footer(fig, cfg, f"| ATL15 {args.lag} época {epoch:.2f} | validação EXTERNA "
                      f"(processamento independente da missão)")
    plt.tight_layout()
    out = cfg.paths.figures / "atl15_validation.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"figura -> {out}")


if __name__ == "__main__":
    main()
