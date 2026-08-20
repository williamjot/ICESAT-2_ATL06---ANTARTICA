"""
thwaites-icesat2
================
Pipeline de produção para monitoramento da Geleira Thwaites (Antártica
Ocidental) com dados altimétricos ICESat-2 / ATL06.

Princípio metodológico: o artigo publicado é referência histórica, não âncora.
Métodos e parâmetros são justificados por mérito
próprio e, quando possível, escolhidos por avaliação empírica.
"""

__version__ = "0.1.0"

from thwaites.config import Config, load_config

__all__ = ["Config", "load_config", "__version__"]
