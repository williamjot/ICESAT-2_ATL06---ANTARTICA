"""
thwaites.glaciology.trajectory
==============================
Integração de trajetórias Lagrangianas de parcelas de gelo, com velocidade
variável no espaço E no tempo (ITS_LIVE anual).

Por que rastrear ANTES de estimar a taxa
----------------------------------------
Na plataforma da Thwaites a velocidade é de 1–4 km/ano: em 7 anos uma parcela
percorre 7–28 km, muito mais que o espaçamento de nós. Uma tendência estimada
num nó FIXO mistura gelo com geometria diferente passando pelo mesmo ponto — o
resultado não é a mudança de espessura de nenhuma coluna de gelo real.

Isto é diferente de `advection.py`, que aplica a correção algébrica
`Dh/Dt = ∂h/∂t + v·∇h` DEPOIS de um ajuste Euleriano. Aquela correção é de
primeira ordem e supõe que o campo Euleriano é estimável; aqui as observações
são agrupadas por parcela antes de qualquer ajuste temporal.

Formulação
----------
A posição da parcela evolui por dx/dt = v(x, t), integrada por Runge-Kutta de
4ª ordem com passo de ~10–30 dias. A velocidade é interpolada bilinearmente no
espaço e linearmente no tempo entre as épocas anuais.

ATENÇÃO ao uso a jusante: com `DH/Dt` Lagrangiano a equação de balanço é
    ṁ_b = a_s − DH/Dt − H·∇·v
usando **H·∇·v**, NÃO a divergência completa ∇·(H·v). Misturar `DH/Dt` com
`∇·(H·v)` conta a advecção duas vezes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from thwaites.logging import get_logger


class VelocityField:
    """
    Campo de velocidade anual interpolável em espaço e tempo.

    Carrega o ITS_LIVE DECIMADO: a grade nativa é 5833×5833×7, o que dá ~950 MB
    por variável se lida inteira. A velocidade varia suavemente na escala de
    quilômetros, então decimar para ~500 m preserva o que importa para
    trajetórias e mantém a memória em centenas de MB.
    """

    def __init__(self, nc_path: str | Path, decimate: int = 4,
                 vx_var: str = "vx", vy_var: str = "vy"):
        from netCDF4 import Dataset

        logger = get_logger()
        with Dataset(nc_path) as d:
            D = max(1, int(decimate))
            self.x = np.asarray(d["x"][::D], dtype=np.float64)
            self.y = np.asarray(d["y"][::D], dtype=np.float64)
            self.vx = np.ma.filled(
                np.asarray(d[vx_var][:, ::D, ::D], dtype=np.float32), np.nan)
            self.vy = np.ma.filled(
                np.asarray(d[vy_var][:, ::D, ::D], dtype=np.float32), np.nan)
            t = d["time"]
            # `time` vem como datas; converte para ano decimal
            try:
                import cftime
                dts = cftime.num2date(t[:], t.units,
                                      only_use_cftime_datetimes=False)
                self.t = np.array([dd.year + (dd.timetuple().tm_yday - 1) / 365.25
                                   for dd in np.atleast_1d(dts)], dtype=float)
            except Exception:
                self.t = np.asarray(t[:], dtype=float)

        # eixos crescentes para indexação previsível
        if self.y[0] > self.y[-1]:
            self.y = self.y[::-1]
            self.vx = self.vx[:, ::-1, :]
            self.vy = self.vy[:, ::-1, :]
        self.dx = float(self.x[1] - self.x[0])
        self.dy = float(self.y[1] - self.y[0])
        logger.info(f"campo de velocidade: {self.vx.shape} @ {abs(self.dx):.0f} m "
                    f"| épocas {self.t.min():.2f}–{self.t.max():.2f} "
                    f"| {self.vx.nbytes/1024**2:.0f} MB")

    def _bilinear(self, field2d, px, py):
        """Interpolação bilinear; NaN fora do domínio ou sobre células sem dado."""
        fx = (px - self.x[0]) / self.dx
        fy = (py - self.y[0]) / self.dy
        j0 = np.floor(fx).astype(np.int64)
        i0 = np.floor(fy).astype(np.int64)
        ok = (j0 >= 0) & (j0 < len(self.x) - 1) & (i0 >= 0) & (i0 < len(self.y) - 1)
        out = np.full(len(px), np.nan)
        if not ok.any():
            return out
        j0o, i0o = j0[ok], i0[ok]
        wx = fx[ok] - j0o
        wy = fy[ok] - i0o
        f00 = field2d[i0o, j0o]
        f10 = field2d[i0o, j0o + 1]
        f01 = field2d[i0o + 1, j0o]
        f11 = field2d[i0o + 1, j0o + 1]
        out[ok] = ((1 - wx) * (1 - wy) * f00 + wx * (1 - wy) * f10 +
                   (1 - wx) * wy * f01 + wx * wy * f11)
        return out

    def at(self, px, py, t_year):
        """
        Velocidade (vx, vy) em m/ano nas posições dadas, no instante `t_year`.

        Interpola linearmente entre as duas épocas anuais que cercam `t_year`.
        Fora do intervalo coberto, usa a época extrema — e isso é uma
        EXTRAPOLAÇÃO que o chamador deve declarar (ex.: 2025 quando a série
        termina antes).
        """
        px = np.asarray(px, dtype=np.float64)
        py = np.asarray(py, dtype=np.float64)
        k = np.searchsorted(self.t, t_year) - 1
        k = int(np.clip(k, 0, len(self.t) - 2))
        t0, t1 = self.t[k], self.t[k + 1]
        w = 0.0 if t1 == t0 else np.clip((t_year - t0) / (t1 - t0), 0.0, 1.0)

        vx = ((1 - w) * self._bilinear(self.vx[k], px, py) +
              w * self._bilinear(self.vx[k + 1], px, py))
        vy = ((1 - w) * self._bilinear(self.vy[k], px, py) +
              w * self._bilinear(self.vy[k + 1], px, py))
        return vx, vy


def integrate_trajectory(vel: VelocityField, x0, y0, t0: float, t1: float,
                         dt_days: float = 20.0):
    """
    Integra parcelas de `t0` até `t1` por Runge-Kutta de 4ª ordem.

    Devolve `(x, y, valid)`. `valid` é falso onde a trajetória saiu do domínio
    de velocidade em algum passo — nesses casos a posição final não tem
    significado e NÃO deve ser usada, em vez de ser silenciosamente truncada.

    RK4 e não Euler: com passos de 20 dias e velocidade de até 4 km/ano, o erro
    de Euler acumula ao longo de 7 anos numa fração não desprezível do
    deslocamento; RK4 mantém o erro de truncamento irrelevante frente à
    incerteza da própria velocidade.
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    y = np.asarray(y0, dtype=np.float64).copy()
    valid = np.ones(len(x), dtype=bool)

    if t1 == t0:
        return x, y, valid
    step = (dt_days / 365.25) * np.sign(t1 - t0)
    n = int(np.ceil(abs(t1 - t0) / abs(step)))
    t = t0
    for _ in range(n):
        h = step if abs(t + step - t0) <= abs(t1 - t0) else (t1 - t)
        if h == 0:
            break
        k1x, k1y = vel.at(x, y, t)
        k2x, k2y = vel.at(x + 0.5 * h * k1x, y + 0.5 * h * k1y, t + 0.5 * h)
        k3x, k3y = vel.at(x + 0.5 * h * k2x, y + 0.5 * h * k2y, t + 0.5 * h)
        k4x, k4y = vel.at(x + h * k3x, y + h * k3y, t + h)
        dx = (h / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
        dy = (h / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)
        bad = ~np.isfinite(dx) | ~np.isfinite(dy)
        valid &= ~bad
        x = np.where(bad, x, x + dx)
        y = np.where(bad, y, y + dy)
        t += h
    return x, y, valid


def track_parcels(vel: VelocityField, x_ref, y_ref, t_ref: float, epochs,
                  dt_days: float = 20.0):
    """
    Posição de cada parcela em cada época pedida, partindo de (x_ref, y_ref) em
    `t_ref`.

    Devolve `(X, Y, V)` de forma (n_epocas, n_parcelas). Integra separadamente
    para trás e para frente a partir da época de referência, o que evita
    acumular erro atravessando `t_ref` duas vezes.
    """
    epochs = np.asarray(epochs, dtype=float)
    n = len(np.asarray(x_ref))
    X = np.full((len(epochs), n), np.nan)
    Y = np.full((len(epochs), n), np.nan)
    V = np.zeros((len(epochs), n), dtype=bool)
    for i, te in enumerate(epochs):
        xe, ye, ok = integrate_trajectory(vel, x_ref, y_ref, t_ref, float(te),
                                          dt_days=dt_days)
        X[i], Y[i], V[i] = xe, ye, ok
    return X, Y, V


def displacement_summary(X, Y) -> dict:
    """Deslocamento total das parcelas — diagnóstico de quanto o Lagrangiano importa."""
    d = np.sqrt((X[-1] - X[0]) ** 2 + (Y[-1] - Y[0]) ** 2)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {}
    return {"n": int(d.size),
            "desloc_mediano_km": float(np.median(d) / 1000),
            "desloc_p90_km": float(np.percentile(d, 90) / 1000),
            "desloc_max_km": float(d.max() / 1000)}
