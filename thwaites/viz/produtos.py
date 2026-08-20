"""
thwaites.viz.produtos
=====================
Leitura e derivação dos campos usados nas figuras de produto (mapas de altura
de flutuação, fluxo por portão, dh/dt lagrangiano e perfis transversais).

Separado dos scripts de figura de propósito: a derivação física — em especial a
altura de flutuação — precisa ser testável sem passar por matplotlib.

Memória
-------
BedMachine tem 13.333 x 13.333 a 500 m e o ITS_LIVE anual tem 7 x 5.833 x 5.833
a 120 m (1,6 GB). Nenhum dos dois é lido inteiro: as funções abaixo recortam a
janela da ROI antes de materializar qualquer array.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

RHO_ICE = 917.0
RHO_SEA = 1027.0


def _janela(eixo: np.ndarray, lo: float, hi: float, folga: int = 4):
    """Índices de `eixo` que cobrem [lo, hi], com folga. Aceita eixo decrescente."""
    crescente = eixo[1] > eixo[0]
    e = eixo if crescente else eixo[::-1]
    i0 = max(np.searchsorted(e, lo) - folga, 0)
    i1 = min(np.searchsorted(e, hi) + folga, e.size)
    if crescente:
        return slice(i0, i1), False
    return slice(eixo.size - i1, eixo.size - i0), True


def carregar_bedmachine(path, x0, x1, y0, y1, vars=("bed", "thickness", "mask")):
    """
    Recorta BedMachine à janela pedida. Devolve (x, y crescente, {var: array}).

    O eixo `y` do arquivo é DECRESCENTE; aqui é invertido para crescente, junto
    com os campos, para que todo o resto do código trate os dois eixos igual.
    """
    from netCDF4 import Dataset

    out = {}
    with Dataset(path) as d:
        bx = np.asarray(d["x"][:], float)
        by = np.asarray(d["y"][:], float)
        sx, _ = _janela(bx, x0, x1)
        sy, inverteu = _janela(by, y0, y1)
        xs = bx[sx]
        ys = by[sy]
        for v in vars:
            a = np.asarray(d[v][sy, sx], float)
            out[v] = a[::-1, :] if inverteu else a
        if inverteu:
            ys = ys[::-1]
    return xs, ys, out


def altura_de_flutuacao(bed, thickness, rho_ice=RHO_ICE, rho_sea=RHO_SEA):
    """
    Altura acima da flutuação (m de gelo).

        H_flot = (ρ_mar / ρ_gelo) · max(−leito, 0)
        HAF    = H − H_flot

    HAF é a espessura EXCEDENTE que ainda prende o gelo ao leito. HAF = 0 é o
    limiar de flutuação: ali a coluna passa a ser sustentada pela água e a
    linha de aterramento migra.

    Onde o leito está acima do nível do mar, `H_flot` é zero e HAF = H — a
    coluna inteira é excedente, porque não há água para flutuar. O `max(·, 0)`
    é o que garante isso; sem ele, leito positivo produziria H_flot negativo e
    um HAF inflado.
    """
    bed = np.asarray(bed, float)
    H = np.asarray(thickness, float)
    h_flot = (rho_sea / rho_ice) * np.maximum(-bed, 0.0)
    return H - h_flot


def tempo_ate_desaterrar(haf, dhdt_ice, min_taxa=0.02):
    """
    Anos até HAF chegar a zero, mantida a taxa atual de perda de espessura.

        t = HAF / (−dH/dt),  para dH/dt < −min_taxa

    Devolve NaN onde o gelo espessa ou a taxa é pequena demais: extrapolar uma
    taxa quase nula produziria milhares de anos, um número sem conteúdo que
    dominaria a escala de cor e esconderia as zonas que importam.

    É extrapolação LINEAR de uma taxa observada em ~7 anos. Não é previsão: a
    taxa muda quando a geometria muda, e num leito retrógrado ela tende a
    acelerar. O valor deve ser lido como "quanto tempo restaria se nada mudasse",
    que é um limite SUPERIOR onde o leito aprofunda em direção ao interior.
    """
    haf = np.asarray(haf, float)
    taxa = np.asarray(dhdt_ice, float)
    perda = -taxa
    t = np.where(perda > min_taxa, haf / np.where(perda > min_taxa, perda, np.nan),
                 np.nan)
    return np.where(haf > 0, t, 0.0)


def carregar_velocidade_anual(path, x0, x1, y0, y1, decimar=1):
    """
    Recorta o ITS_LIVE anual. Devolve (x, y crescente, t_anos, vx, vy, v).

    Arquivo de 1,6 GB: a janela é aplicada na leitura, nunca depois.
    """
    from netCDF4 import Dataset, num2date

    with Dataset(path) as d:
        gx = np.asarray(d["x"][:], float)
        gy = np.asarray(d["y"][:], float)
        sx, _ = _janela(gx, x0, x1)
        sy, inverteu = _janela(gy, y0, y1)
        xs = gx[sx][::decimar]
        ys = gy[sy][::decimar]
        tt = np.asarray(d["time"][:], float)
        anos = np.array([c.year + (c.timetuple().tm_yday - 1) / 365.25
                         for c in num2date(tt, d["time"].units)])
        vx = np.ma.filled(d["vx"][:, sy, sx].astype(float), np.nan)[:, ::decimar, ::decimar]
        vy = np.ma.filled(d["vy"][:, sy, sx].astype(float), np.nan)[:, ::decimar, ::decimar]
    if inverteu:
        vx, vy, ys = vx[:, ::-1, :], vy[:, ::-1, :], ys[::-1]
    return xs, ys, anos, vx, vy, np.hypot(vx, vy)


def amostrar(campo, xs, ys, px, py):
    """Vizinho mais próximo de `campo` (ys, xs) nas posições (px, py)."""
    j = np.clip(np.rint((px - xs[0]) / (xs[1] - xs[0])).astype(np.int64), 0, xs.size - 1)
    i = np.clip(np.rint((py - ys[0]) / (ys[1] - ys[0])).astype(np.int64), 0, ys.size - 1)
    return campo[i, j]


def para_grade(df, coluna, res=5000.0):
    """
    Converte a tabela de células (x, y, valor) numa matriz regular.

    A grade do projeto é regular por construção, mas vem em formato longo; o
    mapa precisa da matriz. Células ausentes ficam NaN, e não zero — zero seria
    lido como "sem mudança" num campo cujo zero tem significado físico.
    """
    x = df["x"].to_numpy(float)
    y = df["y"].to_numpy(float)
    xs = np.arange(x.min(), x.max() + res / 2, res)
    ys = np.arange(y.min(), y.max() + res / 2, res)
    M = np.full((ys.size, xs.size), np.nan)
    j = np.rint((x - xs[0]) / res).astype(int)
    i = np.rint((y - ys[0]) / res).astype(int)
    M[i, j] = df[coluna].to_numpy(float)
    return xs, ys, M


def fluxo_por_portao(x0, y0, x1, y1, xs, ys, vx, vy, H, n=120):
    """
    Descarga através de um segmento, em Gt/ano.

        Q = ∫ H · (v · n̂) dl · ρ_gelo

    `n̂` é a normal unitária ao segmento. Só a componente PERPENDICULAR conta —
    o fluxo paralelo ao portão não o atravessa. Confundir |v| com v·n̂ é o erro
    clássico deste cálculo e superestima a descarga sempre que o gelo cruza o
    portão em ângulo.
    """
    t = np.linspace(0, 1, n)
    px = x0 + (x1 - x0) * t
    py = y0 + (y1 - y0) * t
    dx, dy = x1 - x0, y1 - y0
    L = float(np.hypot(dx, dy))
    nx, ny = dy / L, -dx / L                       # normal unitária
    u = amostrar(vx, xs, ys, px, py)
    v = amostrar(vy, xs, ys, px, py)
    h = amostrar(H, xs, ys, px, py)
    vn = u * nx + v * ny

    # ORIENTAÇÃO DA NORMAL — a convenção tem de vir do fluxo, não da ordem em
    # que os extremos do portão foram escritos. `(dy, −dx)/L` é uma das duas
    # normais possíveis; qual delas sai depende de o segmento ter sido definido
    # de sul para norte ou o contrário. Com a orientação errada a descarga
    # aparece NEGATIVA, como se a geleira estivesse recebendo massa pelo portão.
    # Aqui a normal é virada para o lado de jusante, definido pelo próprio campo
    # de velocidade, de modo que descarga positiva = massa atravessando o portão
    # no sentido do escoamento.
    if np.nansum(h * vn) < 0:
        vn = -vn

    integ = np.nansum(h * vn) * (L / n)            # m³/ano
    return integ * RHO_ICE / 1e12                  # Gt/ano
