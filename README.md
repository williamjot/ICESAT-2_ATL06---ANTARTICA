# ICESat-2 ATL06 — Geleira Thwaites

Pipeline Python reprodutível para processar altimetria ICESat-2/ATL06 na Geleira Thwaites e no Amundsen Sea Embayment, desde a extração seletiva dos grânulos e o controle de qualidade até a estimação de dh/dt, interpolação validada, propagação de incerteza, balanço de massa e produtos científicos derivados.

## Instalação

Requer Python 3.11 ou superior. Em um ambiente virtual:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[analysis,dev]"
pytest -q
```

O extra `analysis` instala as dependências das análises opcionais, incluindo a integração com o modelo regional de maré. O extra `dev` instala o ambiente de testes.

## Arquitetura

```text
├── thwaites/                  # pacote Python
│   ├── corrections/          # correções geofísicas e referenciais
│   ├── diagnostics/          # diagnósticos de vulnerabilidade
│   ├── experiments/          # infraestrutura de sensibilidade e manifestos
│   ├── glaciology/           # fluxo, advecção e trajetórias
│   ├── grid/                 # reprojeção e particionamento espacial
│   ├── interp/               # interpolação e seleção por validação cruzada
│   ├── io/                   # download, extração e armazenamento
│   ├── ocean/                # forçantes e diagnósticos oceânicos
│   ├── qc/                   # máscaras, filtros e crossovers
│   ├── timeseries/           # dh/dt, tendências e aceleração
│   ├── uncertainty/          # propagação de incerteza e balanço de massa
│   ├── validate/             # validações glaciológicas específicas
│   ├── validation/           # folds, métricas e concordância
│   └── viz/                  # funções reutilizáveis de visualização
├── pipelines/                # scripts executáveis do fluxo principal
│   ├── figures/              # figuras, mapas, animações, histogramas e perfis
│   ├── products/             # séries contínuas, nível do mar e previsão
│   └── experiments/          # testes de parâmetros e sensibilidade
├── tests/                    # testes unitários com pytest
├── config/                   # perfis YAML
│   ├── default.yaml
│   ├── jja.yaml
│   ├── djf.yaml
│   ├── anual.yaml
│   └── jja_spacetime.yaml
├── pyproject.toml            # pacote e dependências
└── README.md
```

Dados, resultados, logs e arquivos temporários não fazem parte do repositório.

## Ordem do pipeline principal

Os caminhos são definidos pelos perfis em `config/`; os exemplos de saída abaixo mostram o encadeamento lógico.

| # | Execução | Produto principal |
|---:|---|---|
| 1 | `python pipelines/fetch_bedmachine.py` | máscara e espessura BedMachine |
| 2 | `python pipelines/fetch_rema.py` | mosaico REMA recortado à área de estudo |
| 3 | `python pipelines/run_download.py` | segmentos ATL06 extraídos em arquivos leves |
| 4 | `python pipelines/run_consolidate.py` | observações consolidadas |
| 5 | `python pipelines/run_mask.py` | observações classificadas pela máscara de gelo |
| 6 | `python pipelines/run_cats_tide.py` *(opcional)* | correção regional de maré CATS2008 |
| 7 | `python pipelines/run_corrections.py` | elevação com correções geofísicas selecionadas |
| 8 | `python pipelines/run_slope.py` | elevação referenciada ao relevo local |
| 9 | `python pipelines/run_filttrack.py` | observações filtradas ao longo das trilhas |
| 10 | `python pipelines/run_tiles.py` | partições espaciais para processamento limitado em memória |
| 11 | `python pipelines/run_dhdt.py` | estimativas nodais de dh/dt |
| 12 | `python pipelines/run_timeseries.py` | séries, tendências e diagnósticos temporais |
| 13 | `python pipelines/run_interpolation.py` | grade selecionada por validação cruzada |
| 14 | `python pipelines/run_mass_balance.py` | balanço de massa e incerteza propagada |
| 15 | `python pipelines/figures/run_figures.py` | figuras e tabelas finais |

Análises complementares, como crossovers, velocidade, dinâmica do gelo, firn, GIA e mecanismos oceânicos, possuem scripts próprios em `pipelines/` e devem ser ativadas conforme a pergunta científica e a disponibilidade dos dados externos.

## Dados externos manuais — CATS2008

O modelo regional de maré `CATS2008_v2023.nc` exige download manual por causa do reCAPTCHA do repositório [USAP-DC 601772](https://www.usap-dc.org/view/dataset/601772). Salve o arquivo em `data/tide_models/CATS2008_v2023/CATS2008_v2023.nc` e habilite `cats.enabled: true` no perfil de configuração usado. O arquivo NetCDF não deve ser versionado.

## Premissas científicas

- Cada grânulo ICESat-2 é baixado individualmente, tem somente as variáveis necessárias extraídas e é removido em um bloco `finally`; arquivos HDF5 brutos não são preservados.
- O domínio é trabalhado em EPSG:3031, apropriado à Antártica, após a seleção geográfica das observações.
- A janela sazonal, o método de interpolação, a resolução da grade, o variograma e os limiares de filtragem são decisões abertas e devem ser sustentados por avaliação empírica, sensibilidade e validação cruzada.
- Uma análise restrita a JJA não equivale ao ciclo anual; sua interpretação como estimativa conservadora precisa ser quantificada, não presumida.
- Sem correção de firn, parte de dh/dt pode representar compactação ou variabilidade da coluna de neve, e não mudança de massa de gelo.
- A conversão de dh/dt para massa exige densidade, área efetivamente observada e propagação formal das incertezas espacialmente correlacionadas.
- A aproximação de velocidade integrada na coluna pela velocidade superficial e a hipótese hidrostática só são aplicáveis nos domínios físicos correspondentes.
- Produtos baseados em máscara estática, referência geoidal ou cobertura incompleta devem declarar essas limitações; regiões aparentemente estáveis precisam ser confrontadas com velocidade do gelo antes de uma interpretação dinâmica.
