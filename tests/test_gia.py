"""
Correção GIA: convenção de sinal, interpolação e tratamento da incerteza.

O sinal é o risco central deste módulo. Trocar `−` por `+` em
`correct_elevation_rate` não quebra nada, não gera aviso, e move o resultado de
massa na direção errada por um valor plausível — exatamente o tipo de erro que
sobrevive a uma revisão desatenta e contamina uma conclusão publicada.
"""

import numpy as np
import pytest

from thwaites.corrections.gia import (GIAField, correct_elevation_rate,
                                      systematic_mass_uncertainty, MM_TO_M)


@pytest.fixture
def caron_like(tmp_path):
    """Tabela no formato do Caron: grade de 2° com VLM = colatitude/100 mm/ano."""
    colats = np.arange(0.0, 180.0, 2.0)
    lons = np.arange(0.0, 360.0, 2.0)
    rows = []
    for c in colats:
        for l in lons:
            rows.append([c, l, c / 100.0, 0.5, 0.0, 0.0, 0.0, 0.0])
    p = tmp_path / "GIA_maps_fake"
    with open(p, "w") as f:
        f.write("% cabecalho de comentario\n% segunda linha\n")
        for r in rows:
            f.write(" ".join(f"{v:.7e}" for v in r) + "\n")
    return p


def test_le_tabela_e_converte_unidade(caron_like):
    fld = GIAField.from_caron_table(caron_like)
    assert fld.colat.size == 90 and fld.lon.size == 180
    # VLM em mm/ano na tabela -> m/ano no campo
    i = int(np.argmin(np.abs(fld.colat - 100.0)))
    assert np.isclose(fld.vlm[i, 0], 1.0 * MM_TO_M)


def test_amostragem_em_no_exato(caron_like):
    """Sobre um nó da grade, a bilinear tem de devolver o próprio valor."""
    fld = GIAField.from_caron_table(caron_like)
    lat = 90.0 - 160.0          # colatitude 160
    vlm, sig = fld.sample(np.array([10.0]), np.array([lat]))
    assert np.isclose(vlm[0], 1.60 * MM_TO_M, rtol=1e-9)
    assert np.isclose(sig[0], 0.5 * MM_TO_M, rtol=1e-9)


def test_interpolacao_e_linear_no_meio(caron_like):
    """Entre colatitudes 160 e 162 o valor tem de ser a média."""
    fld = GIAField.from_caron_table(caron_like)
    vlm, _ = fld.sample(np.array([10.0]), np.array([90.0 - 161.0]))
    assert np.isclose(vlm[0], 1.61 * MM_TO_M, rtol=1e-9)


def test_longitude_e_ciclica(caron_like):
    """
    O setor entre 358° e 0° tem de fechar pelo outro lado.

    Sem o fechamento cíclico, 359° cairia no valor de borda e toda a faixa
    entre o último nó e o meridiano de Greenwich ficaria congelada — um erro
    que só aparece em ROIs que cruzam 0°, e que passa despercebido nas que não
    cruzam.
    """
    fld = GIAField.from_caron_table(caron_like)
    lat = 90.0 - 160.0
    a, _ = fld.sample(np.array([359.0]), np.array([lat]))
    b, _ = fld.sample(np.array([-1.0]), np.array([lat]))   # mesma posição
    assert np.isclose(a[0], b[0], rtol=1e-9)
    # campo constante em longitude: o valor não pode saltar na costura
    ref, _ = fld.sample(np.array([180.0]), np.array([lat]))
    assert np.isclose(a[0], ref[0], rtol=1e-9)


def test_sinal_soerguimento_aumenta_a_perda():
    """
    Soerguimento POSITIVO mascara adelgaçamento: corrigir tem de tornar o
    dh/dt MAIS negativo.
    """
    dhdt = np.array([-0.30, -0.30])
    vlm = np.array([0.005, 0.0])          # 5 mm/ano e nada
    out = correct_elevation_rate(dhdt, vlm)
    assert out[0] < dhdt[0], "correção não aumentou a perda sob soerguimento"
    assert np.isclose(out[0], -0.305)
    assert np.isclose(out[1], -0.30), "sem VLM o valor não pode mudar"


def test_subsidencia_reduz_a_perda():
    """E o inverso: subsidência tem de reduzir a perda estimada."""
    out = correct_elevation_rate(np.array([-0.30]), np.array([-0.005]))
    assert np.isclose(out[0], -0.295)


def test_incerteza_gia_nao_e_reduzida_por_sqrt_n():
    """
    O σ do GIA é sistemático: dobrar o nº de células com o MESMO σ não pode
    reduzir a incerteza de massa.

    Se alguém 'otimizar' esta função dividindo por sqrt(N), este teste falha.
    """
    rho, area = 917.0, 2.0e11
    s1 = systematic_mass_uncertainty(np.full(100, 0.001), area, rho)
    s2 = systematic_mass_uncertainty(np.full(10000, 0.001), area, rho)
    assert np.isclose(s1, s2), "incerteza sistemática encolheu com N"
    # e escala linearmente com sigma e area
    assert np.isclose(systematic_mass_uncertainty(np.full(10, 0.002), area, rho),
                      2 * s1)


def test_tabela_irregular_e_rejeitada(tmp_path):
    """Grade incompleta tem de dar erro, não interpolação silenciosa."""
    p = tmp_path / "ruim"
    p.write_text("% h\n0.0 0.0 1.0 0.1 0 0 0 0\n1.0 0.0 1.0 0.1 0 0 0 0\n"
                 "1.0 1.0 1.0 0.1 0 0 0 0\n")
    with pytest.raises(ValueError, match="não formam grade"):
        GIAField.from_caron_table(p)
