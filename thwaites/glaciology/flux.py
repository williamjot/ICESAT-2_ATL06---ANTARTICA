"""
thwaites.glaciology.flux
========================
Divergência de fluxo e derretimento basal — equivalentes ao `cubediv.py` e
`cubemelt.py` do captoolkit. É a etapa que transforma "a superfície baixou
X m/ano" em "o oceano está derretendo Y m/ano por baixo".

FÍSICA (conservação de massa de uma coluna de gelo, em m de gelo equivalente):

    ∂H/∂t = SMB − ṁ_b − ∇·(H·v)

    H   espessura do gelo            [m]
    v   velocidade da superfície     [m/ano]   (assume-se fluxo por deslizamento,
                                                v_coluna ≈ v_superfície)
    SMB balanço de massa superficial [m gelo/ano]
    ṁ_b derretimento basal (positivo = derretendo) [m gelo/ano]

Logo:
    ṁ_b = SMB − ∂H/∂t − ∇·(H·v)

Para gelo FLUTUANTE em equilíbrio hidrostático, a mudança de espessura é
amplificada em relação à mudança de altura de superfície:

    ∂H/∂t = (dh/dt) · ρ_w / (ρ_w − ρ_i)        (≈ 9,3× para 1027/917)

PREMISSAS QUE PRECISAM SER DECLARADAS AO PUBLICAR:
  1. v_coluna ≈ v_superfície (válido onde o fluxo é dominado por deslizamento
     basal — caso do tronco da Thwaites; falha onde há cisalhamento interno);
  2. equilíbrio hidrostático (só vale sobre gelo FLUTUANTE — não use este
     cálculo sobre gelo aterrado);
  3. sem correção de firn: dh/dt inclui compactação de firn, que NÃO é mudança
     de massa. Sem um FDM, parte do sinal é atribuída erroneamente a derretimento;
  4. descasamento temporal velocidade × dh/dt (ver velocity.epoch_note);
  5. sem SMB, o resultado é (ṁ_b − SMB), não ṁ_b.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from thwaites.config import Config
from thwaites.logging import get_logger


# --------------------------------------------------------------- insumos
def load_velocity(cfg: Config, path: str | Path | None = None):
    """
    Carrega o recorte de velocidade (fetch_velocity.py) como (x, y, vx, vy).

    Retorna arrays 1D x, y (EPSG:3031) e 2D vx, vy (m/ano), com nodata -> NaN.
    """
    import xarray as xr

    p = Path(path) if path else (cfg.paths.data_dir / cfg.velocity.path)
    if not p.exists():
        raise FileNotFoundError(
            f"Velocidade não encontrada: {p}\nRode pipelines/fetch_velocity.py.")
    ds = xr.open_dataset(p)
    vx = np.asarray(ds[cfg.velocity.vx_var].values, dtype=np.float64)
    vy = np.asarray(ds[cfg.velocity.vy_var].values, dtype=np.float64)
    x = np.asarray(ds["x"].values, dtype=np.float64)
    y = np.asarray(ds["y"].values, dtype=np.float64)
    # normaliza para y crescente (facilita o gradiente)
    if y[0] > y[-1]:
        y = y[::-1]
        vx = vx[::-1, :]
        vy = vy[::-1, :]
    for a in (vx, vy):
        a[np.abs(a) > 1e9] = np.nan       # fill values típicos
    ds.close()
    return x, y, vx, vy


def sample_bedmachine(cfg: Config, variable: str, x, y, nc_path: str | Path | None = None,
                      method: str = "linear"):
    """
    Amostra uma variável do BedMachine (ex.: 'thickness') numa grade x/y
    (EPSG:3031).

    O BedMachine .nc é o baixado por fetch_bedmachine.py (tem thickness, bed,
    surface, errbed além do mask).

    `method`: "linear" para campos contínuos (thickness, bed, surface) e
    "nearest" para campos CATEGÓRICOS. Interpolar 'mask' bilinearmente
    produziria valores intermediários (ex.: 2,4 entre aterrado=2 e flutuante=3)
    que não correspondem a nenhuma classe real.
    """
    import xarray as xr
    from scipy.interpolate import RegularGridInterpolator

    if nc_path is None:
        cands = sorted(cfg.paths.data_dir.glob("*BedMachineAntarctica*.nc"))
        if not cands:
            raise FileNotFoundError(
                "NetCDF do BedMachine não encontrado em data/ "
                "(o fetch_bedmachine.py o mantém após gerar o .tif).")
        nc_path = cands[0]

    ds = xr.open_dataset(nc_path)
    if variable not in ds:
        raise ValueError(f"'{variable}' ausente. Disponíveis: {list(ds.data_vars)}")
    bx = np.asarray(ds["x"].values, dtype=np.float64)
    by = np.asarray(ds["y"].values, dtype=np.float64)
    # recorta à área pedida (+buffer) antes de materializar — economia de memória
    pad = 5000.0
    ix = np.where((bx >= np.min(x) - pad) & (bx <= np.max(x) + pad))[0]
    iy = np.where((by >= np.min(y) - pad) & (by <= np.max(y) + pad))[0]
    if ix.size == 0 or iy.size == 0:
        raise ValueError("a grade pedida não intersecta o BedMachine.")
    sub = ds[variable].isel(x=ix, y=iy)
    vals = np.asarray(sub.values, dtype=np.float64)
    sbx, sby = bx[ix], by[iy]
    if sby[0] > sby[-1]:
        sby = sby[::-1]
        vals = vals[::-1, :]
    ds.close()

    interp = RegularGridInterpolator((sby, sbx), vals, method=method,
                                     bounds_error=False, fill_value=np.nan)
    X, Y = np.meshgrid(x, y)
    return interp(np.c_[Y.ravel(), X.ravel()]).reshape(Y.shape)


# ------------------------------------------------------------ cálculo
def _smooth(a, sigma_cells):
    """Suavização gaussiana preservando NaN (derivar dado ruidoso amplifica ruído)."""
    from scipy.ndimage import gaussian_filter
    if sigma_cells <= 0:
        return a
    v = np.isfinite(a)
    filled = np.where(v, a, 0.0)
    num = gaussian_filter(filled, sigma=sigma_cells)
    den = gaussian_filter(v.astype(float), sigma=sigma_cells)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 1e-6, num / den, np.nan)
    out[~v & (den <= 1e-6)] = np.nan
    return out


def flux_divergence(x, y, H, vx, vy, cfg: Config):
    """
    ∇·(H·v) = ∂(H·vx)/∂x + ∂(H·vy)/∂y   [m gelo/ano]

    H, vx, vy são grades 2D sobre (y, x). Suaviza H e v antes de derivar,
    porque a derivada amplifica ruído.
    """
    dx = float(np.mean(np.diff(x)))
    dy = float(np.mean(np.diff(y)))
    sigma_cells = (cfg.flux.smooth_km * 1000.0) / max(abs(dx), 1.0)

    Hs = _smooth(H, sigma_cells)
    vxs = _smooth(vx, sigma_cells)
    vys = _smooth(vy, sigma_cells)

    qx = Hs * vxs           # fluxo em x [m²/ano]
    qy = Hs * vys
    # np.gradient devolve (d/dy, d/dx) para arrays (y, x)
    dqx_dy, dqx_dx = np.gradient(qx, dy, dx)
    dqy_dy, dqy_dx = np.gradient(qy, dy, dx)
    div = dqx_dx + dqy_dy

    thin = ~np.isfinite(Hs) | (Hs < cfg.flux.min_thickness_m)
    div[thin] = np.nan
    return div


def hydrostatic_amplification(cfg: Config) -> float:
    """Fator ρ_w/(ρ_w − ρ_i) da relação hidrostática (~9,3 com os padrões)."""
    rho_i = cfg.mass_balance.ice_density
    rho_w = cfg.flux.water_density
    return rho_w / (rho_w - rho_i)


def hydrostatic_thickness_rate(dhdt, cfg: Config, floating=None):
    """
    ∂H/∂t a partir de dh/dt.

        gelo FLUTUANTE:  ∂H/∂t = dh/dt · ρ_w/(ρ_w − ρ_i)   (~9,3×)
        gelo ATERRADO:   ∂H/∂t = dh/dt                     (fator 1)

    `floating` é uma máscara booleana da mesma forma de `dhdt`. **Obrigatória na
    prática**: sem ela a amplificação hidrostática é aplicada à grade inteira, o
    que superestima ∂H/∂t em ~9,3× sobre todo o gelo aterrado — a maior parte de
    qualquer domínio continental — e contamina o derretimento basal derivado.

    Passar `floating=None` aplica a amplificação a toda a grade e emite aviso.
    """
    amp = hydrostatic_amplification(cfg)
    dhdt = np.asarray(dhdt, dtype=np.float64)
    if floating is None:
        import warnings
        warnings.warn(
            "hydrostatic_thickness_rate sem máscara `floating`: a amplificação "
            f"({amp:.2f}×) será aplicada também a gelo aterrado, onde o fator "
            "correto é 1,0. Passe a máscara de flutuação.",
            RuntimeWarning, stacklevel=2)
        return dhdt * amp
    floating = np.asarray(floating, dtype=bool)
    return np.where(floating, dhdt * amp, dhdt)


def basal_melt_rate(dHdt, div, smb=None, floating=None):
    """
    ṁ_b = SMB − ∂H/∂t − ∇·(H·v)      [m gelo/ano, positivo = derretendo]

    Sem `smb`, o retorno é (ṁ_b − SMB): a taxa de derretimento MENOS o balanço
    de massa superficial. Interprete como tal — não como ṁ_b puro.

    `floating`: onde a máscara é falsa o resultado vira NaN. O balanço só fecha
    em derretimento basal sob gelo flutuante; sobre gelo aterrado o mesmo
    resíduo mistura deformação interna, derretimento basal subglacial e erro do
    dado — reportá-lo como "derretimento basal" seria uma afirmação física
    incorreta. Sem a máscara, a estatística de ṁ_b passa a ser dominada por
    células onde a grandeza não é definida.
    """
    dHdt = np.asarray(dHdt, dtype=np.float64)
    div = np.asarray(div, dtype=np.float64)
    base = -dHdt - div
    out = base if smb is None else np.asarray(smb, dtype=np.float64) + base
    if floating is not None:
        out = np.where(np.asarray(floating, dtype=bool), out, np.nan)
    return out
