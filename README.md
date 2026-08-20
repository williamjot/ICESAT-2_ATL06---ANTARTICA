# ICESat-2 ATL06 — Geleira Thwaites

Pipeline Python reprodutível para processar altimetria ICESat-2/ATL06 na Geleira Thwaites e no Amundsen Sea Embayment, desde a extração seletiva dos grânulos e o controle de qualidade até a estimação de dh/dt, interpolação validada, propagação de incerteza, balanço de massa e produtos científicos derivados.

## Sumário

- [Instalação](#instalação)
- [Arquitetura](#arquitetura)
- [Ordem do pipeline principal](#ordem-do-pipeline-principal)
- [Guia dos scripts em pipelines](#guia-dos-scripts-em-pipelines)
- [Guia dos módulos em thwaites](#guia-dos-módulos-em-thwaites)
- [Dados externos manuais — CATS2008](#dados-externos-manuais--cats2008)
- [Premissas científicas](#premissas-científicas)

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

Os caminhos são definidos pelos perfis em `config/`. A tabela resume o fluxo em
15 fases; algumas fases agrupam mais de um executável. A ordem operacional
completa está codificada em `pipelines/run_chain.py`.

| # | Fase e execução | Entrada principal | Produto ou decisão |
|---:|---|---|---|
| 1 | Referências: `fetch_bedmachine.py` e `fetch_rema.py` | metadados da ROI | máscara/espessura BedMachine e DEM REMA recortados |
| 2 | Aquisição: `run_download.py` | busca Earthdata ATL06 | Parquet leve por grânulo; o HDF5 temporário é removido |
| 3 | Consolidação: `run_consolidate.py` | Parquets por grânulo | tabela ATL06 consolidada e filtrada pelos flags nativos |
| 4 | Máscara e correções: `run_mask.py`, `run_cats_tide.py`, `run_corrections.py` | pontos, BedMachine e maré | domínio classificado e elevação `h_corr` |
| 5 | Referência local e trilhas: `run_slope.py`, `run_filttrack.py` | `h_corr`, REMA e geometria orbital | `h_res`, `track_id` e rejeição de blunders along-track |
| 6 | Domínio aterrado e tiles: `run_grounded_mask.py`, `run_tiles.py` | pontos filtrados | recorte com buffers físicos e tiles com halo |
| 7 | Taxa de elevação: `run_dhdt.py` | tiles ATL06 | dh/dt e aceleração formal por nó |
| 8 | Incerteza e QC nodal: `run_uncertainty.py`, `run_qc_report.py` | estimativas nodais | incerteza por jackknife e `dhdt_nodes_qc.parquet` |
| 9 | Interpolação: `run_interpolation.py` | nós validados | método escolhido por CV espacial e grade de dh/dt |
| 10 | Série e tendência: `run_timeseries.py` | tiles e nós | série nó × ano, Mann–Kendall, Sen e FDR |
| 11 | Firn: `run_firn.py` | dh/dt e modelo de firn | separação entre mudança de superfície e de gelo |
| 12 | Massa: `run_mass_balance.py` | grade explícita e incertezas | Gt/ano e equivalente de nível do mar |
| 13 | Verificações independentes: `run_xover.py`, `run_validation.py`, `run_dynamics.py` | observações, crossovers e velocidade | viés, métricas sem vazamento e diagnóstico dinâmico |
| 14 | Plataforma flutuante: `run_shelf_mask.py` até `run_basal_melt.py` | frentes datadas, ITS_LIVE e pontos flutuantes | DH/Dt lagrangiano, divergência e derretimento basal |
| 15 | Comunicação: `pipelines/figures/run_figures.py` | produtos validados | mapas, histogramas e diagramas finais |

### Execução e perfis

Cada executável pode ser inspecionado com `--help`. O orquestrador executa as
etapas em subprocessos separados, registra um log por etapa e interrompe a cadeia
se uma delas falhar:

```bash
python pipelines/run_chain.py --profile jja
python pipelines/run_chain.py --profile djf
python pipelines/run_chain.py --profile anual
python pipelines/run_interpolation.py --help
```

Os perfis alteram configuração e diretórios de derivados. Eles não representam
uma afirmação de que JJA, DJF ou o ciclo anual sejam metodologicamente superiores.

## Guia dos scripts em `pipelines/`

Os arquivos desta pasta são pontos de entrada. Eles interpretam argumentos,
carregam configuração, registram logs e chamam a lógica reutilizável de
`thwaites/`. Os scripts `fetch_*` obtêm dados auxiliares; os `run_*` executam
processamento ou diagnóstico. Nenhum deles deve ser importado como biblioteca.

### Aquisição de dados e referências

| Script | Explicação |
|---|---|
| `pipelines/fetch_atl21.py` | Busca ATL21 mensal, baixa um grânulo por vez, extrai a anomalia de altura da superfície do mar na parte marinha da ROI e remove o HDF5 temporário. É um ramo oceanográfico, não uma substituição do ATL06. |
| `pipelines/fetch_bedmachine.py` | Obtém o BedMachine Antarctica e prepara a máscara em EPSG:3031 usada para separar oceano, gelo aterrado, plataforma e rocha; também fornece campos de leito, espessura e geoide a módulos derivados. |
| `pipelines/fetch_firn.py` | Baixa e recorta o GSFC-FDM à ROI para estimar variações de firn/FAC e SMB. A cobertura temporal do modelo deve ser verificada antes de extrapolar a correção. |
| `pipelines/fetch_glorys12v1_jja.py` | Solicita subconjuntos anuais JJA do GLORYS12V1 para o ASE, valida os arquivos recebidos e grava manifesto. Requer credenciais e não define, por si só, que JJA seja a janela correta. |
| `pipelines/fetch_grounding_products.py` | Baixa e converte produtos observacionais de linha e zona de aterramento, incluindo informação temporal usada na classificação grounded/GZ/floating. |
| `pipelines/fetch_icelines.py` | Prepara frentes de calving datadas do IceLines/Sentinel-1 por plataforma; essas geometrias evitam tratar a frente como estática durante todo o período. |
| `pipelines/fetch_itslive.py` | Recorta compósitos anuais ITS_LIVE de velocidade para a ROI e os organiza para rastreamento lagrangiano e diagnósticos de dinâmica. |
| `pipelines/fetch_ocean_melt.py` | Obtém e valida o pequeno conjunto observacional BAS/ITGC MELT usado como evidência oceanográfica independente na interface gelo–oceano. |
| `pipelines/fetch_qc_flags.py` | Reabre somente os grânulos ATL06 que contribuíram para a ROI e extrai variáveis adicionais de qualidade, mantendo pareamento estrito com os Parquets processados. |
| `pipelines/fetch_rema.py` | Determina a extensão diretamente da configuração, identifica tiles REMA v2 de 32 m e gera o mosaico recortado usado na referência topográfica e em declividades. |
| `pipelines/fetch_velocity.py` | Obtém o mosaico MEaSUREs de velocidade e o recorta à ROI para verificar zonas de estabilidade altimétrica aparente e calcular produtos de fluxo. |

### Cadeia principal e controle de qualidade

| Script | Explicação |
|---|---|
| `pipelines/run_chain.py` | Orquestra em sequência o ramo de gelo aterrado e o ramo de plataforma. Resolve caminhos dependentes do perfil, executa cada etapa em processo isolado, registra duração e preserva a cauda do log em caso de falha. |
| `pipelines/run_download.py` | Adaptador fino para `thwaites.io.download.run_download`: autentica uma vez, busca metadados, aplica período/estação e processa um grânulo ATL06 por vez. |
| `pipelines/run_consolidate.py` | Une os Parquets leves por grânulo em uma tabela intermediária, sem depender dos HDF5 brutos, e mantém o esquema necessário às etapas seguintes. |
| `pipelines/run_mask.py` | Amostra a máscara BedMachine por janela e adiciona `mask_class`, removendo classes fora do domínio amplo configurado antes dos recortes científicos específicos. |
| `pipelines/run_cats_tide.py` | Prediz maré CATS2008 nos pontos, preserva a maré global original para comparação e substitui `tide_ocean` somente conforme a configuração e o domínio aplicável. |
| `pipelines/run_corrections.py` | Subtrai maré oceânica e DAC de forma configurável para produzir `h_corr`; verifica obsolescência da entrada CATS e processa Parquet em lotes limitados em memória. |
| `pipelines/run_slope.py` | Amostra o REMA por blocos e produz `h_res = h_corr − REMA`, removendo a topografia estática local do ajuste em nós fixos. Esse residual não deve ser usado indiscriminadamente em trajetórias móveis. |
| `pipelines/run_filttrack.py` | Identifica trilhas e rejeita blunders ao longo do traço por estatística local robusta, com duas passagens para evitar materializar toda a tabela. |
| `pipelines/run_grounded_mask.py` | Constrói o recorte de gelo aterrado, aplica buffers de linha de aterramento e costa, calcula campos de distância e produz relatório auditável do que foi mantido. |
| `pipelines/run_tiles.py` | Divide os pontos em tiles EPSG:3031 com halo, permitindo ajustes locais contínuos nas bordas e processamento com memória limitada. |
| `pipelines/run_dhdt.py` | Executa o ajuste espaço-temporal local em todos os tiles e consolida dh/dt, aceleração formal, número de observações e diagnósticos de condicionamento por nó. |
| `pipelines/run_uncertainty.py` | Reestima a incerteza da taxa por jackknife entre anos. Deve preceder o QC nodal para que a incerteza defensável seja propagada ao produto validado. |
| `pipelines/run_qc_report.py` | Filtra nós pela classe física, buffers e suporte espacial, gera `dhdt_nodes_qc.parquet` e cria mapas/tabelas que documentam retenções e rejeições. |
| `pipelines/run_interpolation.py` | Compara interpoladores por validação cruzada em blocos espaciais, ajusta variograma quando aplicável e gera a grade somente com o método vencedor segundo métricas declaradas. |
| `pipelines/run_timeseries.py` | Constrói elevação anual por nó e calcula testes formais de tendência, incluindo Mann–Kendall, Sen e correção FDR; não converte automaticamente tendência sazonal em anual. |
| `pipelines/run_firn.py` | Amostra a taxa de FAC/SMB, calcula `dhdt_ice` e registra sensibilidades; mantém separados o produto bruto de superfície e o produto corrigido de firn. |
| `pipelines/run_mass_balance.py` | Integra uma grade explícita de taxa usando área, densidade, cobertura e correlação do erro para produzir Gt/ano e contribuição eustática com incerteza. |

### Diagnósticos, validações e ramos científicos

| Script | Explicação |
|---|---|
| `pipelines/run_acceleration.py` | Testa se o termo quadrático representa aceleração estatisticamente sustentada, usando critérios mínimos, bootstrap e checkpoints por tile. |
| `pipelines/run_advection.py` | Calcula `v·∇h` e converte dh/dt Euleriano em taxa lagrangiana para quantificar o efeito de advecção; não deve ser somado novamente a um orçamento que já contém divergência de fluxo. |
| `pipelines/run_agreement.py` | Compara espacialmente fitsec e crossovers pareados, estima viés/regressão robusta, localiza hotspots de discordância e avalia sensibilidade à subamostragem. |
| `pipelines/run_aliasing.py` | Injeta harmônicos nas épocas reais e mede vazamento sazonal para a tendência; também compara JJA e ano inteiro na componente representada pelo modelo de firn. |
| `pipelines/run_ase_glorys_jja_mechanism.py` | Processa os subconjuntos GLORYS anuais, calcula métricas JJA na camada oceânica de interesse e organiza a série regional para o diagnóstico do mecanismo. |
| `pipelines/run_atl15_validation.py` | Faz validação externa contra ATL15 em grade, estima a tendência ATL15 no período compatível e amostra o campo nos locais do produto do projeto. |
| `pipelines/run_basal_ocean.py` | Encadeia incrementalmente a estimativa de derretimento basal e o diagnóstico oceânico, pulando etapas já válidas e registrando dependências. |
| `pipelines/run_dhdt_janelas.py` | Recalcula dh/dt em janelas móveis ou de início fixo para mostrar como o padrão espacial da taxa evolui ao longo do registro. |
| `pipelines/run_dynamics.py` | Integra dh/dt, velocidade e posição relativa à linha de aterramento para classificar comportamento dinâmico e testar correlações com autocorrelação espacial. |
| `pipelines/run_flux.py` | Calcula `∇·(H·v)`, aplica a conversão hidrostática somente onde fisicamente válida e deriva um diagnóstico de derretimento basal com suas premissas. |
| `pipelines/run_gia.py` | Aplica movimento vertical do embasamento à taxa de elevação, recalcula massa e mantém comparação com/sem GIA e uma incerteza sistemática separada. |
| `pipelines/run_grounding_qa.py` | Produz inspeções visuais e temporais dos produtos de grounding, permitindo detectar geometrias, épocas ou larguras de flexão inconsistentes. |
| `pipelines/run_ocean_mechanism.py` | Relaciona forçantes observacionais junto à TEIS com o diagnóstico basal e declara limitações de tamanho amostral e autocorrelação temporal. |
| `pipelines/run_sigma_corr.py` | Quantifica erro de altura induzido por geolocalização horizontal e declividade, calibra sua influência temporal e o incorpora à incerteza correlacionada. |
| `pipelines/run_space_time_grounding_mask.py` | Classifica cada observação no tempo como grounded, zona de aterramento, floating ou desconhecida usando produtos observacionais e larguras de flexão. |
| `pipelines/run_validation.py` | Recalcula nós dentro de folds espaciais, orbitais e temporais sem compartilhar observações entre treino e teste; reporta erro, viés e cobertura por método. |
| `pipelines/run_velocity_check.py` | Cruza dh/dt com MEaSUREs para identificar locais onde taxa próxima de zero pode coexistir com fluxo rápido, evitando interpretar automaticamente esses nós como estáveis. |
| `pipelines/run_xover.py` | Encontra cruzamentos entre trilhas, estima uma taxa independente do fitsec e mede viés inter-feixe e sua sensibilidade. |

### Ramo de plataforma flutuante

| Script | Explicação |
|---|---|
| `pipelines/run_shelf_mask.py` | Seleciona gelo flutuante com máscara BedMachine e frentes IceLines dependentes de época, aplicando buffers próprios da grounding zone e da frente de calving. |
| `pipelines/run_shelf_lagrangian.py` | Rastreia parcelas com velocidade ITS_LIVE variável no espaço e tempo, harmoniza o datum quando possível e ajusta DH/Dt ao longo de cada trajetória. |
| `pipelines/run_shelf_windows.py` | Repete o ajuste lagrangiano por plataforma em janelas sazonais móveis, produzindo evolução temporal comparável entre JJA e DJF. |
| `pipelines/run_shelf_divergence.py` | Amostra espessura e velocidade, suaviza antes de derivar e calcula `H·∇·v` nas parcelas, com diagnóstico de cobertura e ruído. |
| `pipelines/run_basal_melt.py` | Combina acumulação superficial, DH/Dt lagrangiano e divergência para estimar derretimento basal por parcela; o resultado depende da máscara, do firn/SMB e das hipóteses declaradas. |

### Experimentos de parâmetros

| Script | Explicação |
|---|---|
| `pipelines/experiments/run_sensitivity.py` | Varre configurações de filtros e buffers em sub-regiões representativas, compara cada cenário com uma linha de base pré-declarada e grava manifesto de aceitação. |
| `pipelines/experiments/run_ase_jja_dhdt_interpolation_tests.py` | Compara, no ASE, a representação nodal sem interpolação com IDW selecionado por folds espaciais e produz figuras comuns para avaliação A/B. |
| `pipelines/experiments/run_ase_jja_dhdt_test_b_suave.py` | Gera a variante cartográfica suavizada do teste B em grade mais fina; serve para avaliar representação visual e não substitui validação quantitativa. |

### Produtos derivados

| Script | Explicação |
|---|---|
| `pipelines/products/run_previsao.py` | Faz hindcast por nó, treina somente antes de uma data de corte e avalia previsões nas posições/épocas observadas para determinar horizonte defensável de extrapolação. |
| `pipelines/products/run_sealevel.py` | Calcula tendência espacial de SSHA a partir do ATL21 na parte marinha da ROI e quantifica risco de viés por amostragem; não converte diretamente perda de gelo em nível do mar. |
| `pipelines/products/run_serie_massa.py` | Agrega massa por nó e ano, produz série acumulada e campo em metros de água equivalente dentro da cobertura efetivamente observada. |
| `pipelines/products/run_ase_regional_architecture.py` | Harmoniza IBCSO e BedMachine numa grade regional para mapear batimetria, leito, cavidades e arquitetura oceânica estática do ASE. |
| `pipelines/products/run_ase_jja_atl06_continuous.py` | Constrói um campo contínuo exploratório JJA sobre plataformas, selecionando parâmetros IDW por folds e preservando a máscara flutuante de alta resolução. |
| `pipelines/products/run_ase_jja_basal_dhdt_products.py` | Integra campos JJA de dh/dt e derretimento basal, inclusive janelas móveis, mapas conjuntos e animações na resolução de análise. |
| `pipelines/products/run_ase_jja_basal_dhdt_products_500m.py` | Reamostra os campos integrados para a grade nativa de 500 m e gera mapas/animações, sem alegar que a resolução visual adiciona informação observacional. |
| `pipelines/products/run_ase_seasonal_continuous_products.py` | Produz campos contínuos JJA/DJF, mapas conjuntos e animações comparáveis, mantendo diretórios e escalas sazonais explicitamente controlados. |

### Figuras, mapas e animações

| Script | Explicação |
|---|---|
| `pipelines/figures/run_figures.py` | Gerador geral de figuras a partir de produtos validados; recusa resumos obsoletos e não usa nós crus como fallback silencioso. |
| `pipelines/figures/run_maps.py` | Cria mapas finais de dh/dt, velocidade, derretimento basal e orçamento sobre relevo sombreado, contornos físicos e frentes datadas. |
| `pipelines/figures/run_basal_diagnostic_maps.py` | Produz mapas integrados de derretimento basal e diagnósticos da Thwaites, com camadas auxiliares e incerteza. |
| `pipelines/figures/run_ase_jja_diagnostic_maps.py` | Recorta rigorosamente a ROI do ASE e gera mapas diagnósticos JJA, inclusive incerteza basal e subconjunto de nós confiáveis. |
| `pipelines/figures/run_figuras_comparativas.py` | Gera histogramas e curvas de massa em painéis independentes voltados a comparação científica. |
| `pipelines/figures/run_figuras_massa.py` | Produz série temporal e mapa de massa em linguagem visual comparável à de produtos gravimétricos, mas restrita ao método e à cobertura ICESat-2. |
| `pipelines/figures/run_histogramas_dhdt_individuais.py` | Cria histogramas separados de JJA e DJF, evitando que uma distribuição agregada oculte diferenças sazonais. |
| `pipelines/figures/run_perfil_bruto.py` | Calcula perfis latitudinais diretamente dos segmentos ATL06 por mediana e balanceamento, sem ajuste nodal ou interpolação. |
| `pipelines/figures/run_perfil_trilha.py` | Constrói corredor curvo de repetição orbital, agrega elevação por ano e mostra rebaixamento ao longo da mesma geometria de trilha. |
| `pipelines/figures/run_previsao_mapa.py` | Mapeia projeções somente até o horizonte aceito pelo hindcast e apresenta a incerteza correspondente. |
| `pipelines/figures/run_produto_figuras.py` | Reúne figuras de produto JJA/DJF: altura de flutuação, portões de fluxo, DH/Dt lagrangiano, basal, transectos e janelas móveis. |
| `pipelines/figures/run_animacao_janelas.py` | Anima a evolução de dh/dt em janelas equivalentes para JJA e DJF. |
| `pipelines/figures/run_animacao_massa.py` | Anima a evolução da massa acumulada em formato 16:9 a partir do produto temporal já calculado. |
| `pipelines/figures/run_ase_jja_mass_animation.py` | Gera mapa e animação JJA de perda de massa no padrão cartográfico regional do ASE. |
| `pipelines/figures/run_ase_seasonal_side_by_side_animations.py` | Renderiza JJA e DJF lado a lado no mesmo quadro, com extensão, linha do tempo e escalas coordenadas. |

## Guia dos módulos em `thwaites/`

`thwaites/` é a biblioteca reutilizável. Os módulos não devem depender da forma
de apresentação final: recebem arrays, tabelas, grades ou configuração e retornam
produtos testáveis. Os arquivos `__init__.py` abaixo definem os namespaces dos
subpacotes; a implementação científica está nos demais módulos.

### Núcleo e namespaces

| Módulo | Responsabilidade |
|---|---|
| `thwaites/__init__.py` | Expõe a versão do pacote e `load_config`, além do princípio de que decisões metodológicas precisam de mérito próprio. |
| `thwaites/config.py` | Define modelos Pydantic para área, período, produto, QC, correções, interpolação, tendências, massa e caminhos; faz merge profundo de perfis YAML e rejeita chaves desconhecidas. |
| `thwaites/logging.py` | Configura logs estruturados no terminal e em arquivo rotativo por execução. |
| `thwaites/corrections/__init__.py` | Namespace das correções geofísicas e de referencial vertical. |
| `thwaites/diagnostics/__init__.py` | Namespace dos diagnósticos de vulnerabilidade. |
| `thwaites/experiments/__init__.py` | Namespace da infraestrutura de experimentos reproduzíveis. |
| `thwaites/glaciology/__init__.py` | Namespace de fluxo, advecção e trajetórias. |
| `thwaites/grid/__init__.py` | Namespace de reprojeção e tiling EPSG:3031. |
| `thwaites/interp/__init__.py` | Namespace de interpolação e variogramas. |
| `thwaites/io/__init__.py` | Namespace de download, extração e armazenamento. |
| `thwaites/ocean/__init__.py` | Namespace dos diagnósticos oceânicos. |
| `thwaites/qc/__init__.py` | Namespace dos filtros e máscaras de qualidade. |
| `thwaites/timeseries/__init__.py` | Namespace de dh/dt, séries, tendência, aceleração e aliasing. |
| `thwaites/uncertainty/__init__.py` | Namespace da propagação de incerteza até massa e nível do mar. |
| `thwaites/validate/__init__.py` | Namespace das validações com dados independentes específicos. |
| `thwaites/validation/__init__.py` | Namespace da infraestrutura geral de validação sem vazamento. |
| `thwaites/viz/__init__.py` | Namespace das funções reutilizáveis de visualização. |

### Correções geofísicas e referenciais

| Módulo | Responsabilidade |
|---|---|
| `thwaites/corrections/apply.py` | Aplica, com gating configurável, maré oceânica e DAC às colunas extraídas e produz `h_corr` sem esconder quais termos foram usados. |
| `thwaites/corrections/cats_tide.py` | Resolve o modelo CATS2008, prediz maré com pyTMD e oferece caminhos em memória ou streaming, preservando comparação com a maré ATL06. |
| `thwaites/corrections/datum.py` | Amostra o geoide EIGEN-6C4/BedMachine, converte altura elipsoidal em ortométrica e estima erro de taxa devido ao gradiente geoidal em trajetórias. |
| `thwaites/corrections/firn.py` | Lê campos de FAC/SMB, estima taxas no período, amostra nos nós, calcula `dhdt_ice` e executa análise de sensibilidade à cobertura do modelo. |
| `thwaites/corrections/gia.py` | Representa o campo de movimento vertical do embasamento, corrige dh/dt e mantém a incerteza sistemática de GIA separada. |
| `thwaites/corrections/slope.py` | Localiza o REMA, amostra por blocos com interpolação bilinear e calcula `h_res` sem materializar o mosaico completo. |

### Experimentos reproduzíveis

| Módulo | Responsabilidade |
|---|---|
| `thwaites/experiments/manifest.py` | Calcula hashes de entradas, configuração e árvore de código; registra commit, parâmetros, sementes e saídas; e impede sobrescrita acidental de um experimento. |
| `thwaites/experiments/sensitivity.py` | Define a grade de parâmetros, aplica sobrescritas controladas, seleciona regiões, executa cenários e avalia cada resultado contra critérios de aceitação pré-declarados. |

### Entrada, saída, grade e memória

| Módulo | Responsabilidade |
|---|---|
| `thwaites/io/download.py` | Autentica no Earthdata, busca e filtra grânulos, baixa um por vez, chama a extração e remove o HDF5 dentro de `finally`. |
| `thwaites/io/extract.py` | Lê os caminhos HDF5 configurados por feixe, converte tempo, aplica flags essenciais e retorna somente as variáveis necessárias em DataFrame. |
| `thwaites/io/gridded.py` | Identifica e valida eixos 1D/2D de produtos NetCDF e recorta grades regulares à ROI polar. |
| `thwaites/io/memory.py` | Estima custo em RAM, seleciona colunas, reduz tipos e fornece leitura/escrita em lotes com orçamento explícito. |
| `thwaites/io/store.py` | Define esquema de pontos, grava/lê Parquet e consolida arquivos com metadados de proveniência. |
| `thwaites/grid/reproject.py` | Mantém transformadores em cache para conversões EPSG:4326 ↔ EPSG:3031. |
| `thwaites/grid/tiles.py` | Atribui coordenadas polares, cria tiles com núcleo e halo em memória ou streaming e carrega o manifesto espacial. |

### Controle de qualidade e máscaras

| Módulo | Responsabilidade |
|---|---|
| `thwaites/qc/atl06_flags.py` | Constrói máscara booleana a partir dos flags nativos ATL06 e resume quantos pontos cada critério rejeitou. |
| `thwaites/qc/filtst.py` | Detecta outliers em células espaço-temporais, respeitando suporte mínimo e escala configurada. |
| `thwaites/qc/filttrack.py` | Gera identificadores de trilha e aplica filtro robusto along-track por arrays, preservando a ordem original. |
| `thwaites/qc/front_mask.py` | Densifica frentes IceLines, escolhe a época aplicável e classifica a posição do ponto em relação à frente de calving. |
| `thwaites/qc/grounded_mask.py` | Lê a ROI BedMachine, calcula distâncias a oceano, gelo flutuante e gelo aterrado e implementa máscaras independentes para domínio aterrado e plataforma. |
| `thwaites/qc/grounding_zone.py` | Normaliza nomes de geleira, deriva largura de flexão e combina campos para classificação espaço-temporal grounded/GZ/floating/unknown. |
| `thwaites/qc/mask.py` | Amostra `mask_class` por leitura de janela determinística e remove classes fora dos valores aceitos sem carregar o raster inteiro. |
| `thwaites/qc/reliability.py` | Classifica nós em níveis de confiabilidade a partir de suporte temporal, espacial, incerteza e condições físicas, gerando relatório agregado. |
| `thwaites/qc/xover.py` | Classifica trilhas, encontra crossovers, estima taxa entre passagens e calcula viés inter-feixe com sensibilidade. |

### Estimação temporal

| Módulo | Responsabilidade |
|---|---|
| `thwaites/timeseries/dhdt.py` | Monta e resolve o ajuste local espaço-temporal, faz rejeição iterativa por MAD, calcula taxa/aceleração e suporta janelas móveis por tile. |
| `thwaites/timeseries/build.py` | Ajusta um plano local por nó e ano para produzir a série temporal de elevação com incerteza e suporte observacional. |
| `thwaites/timeseries/trend.py` | Implementa Mann–Kendall, Sen, variante sazonal e correção FDR entre nós. |
| `thwaites/timeseries/model.py` | Constrói modelos temporais candidatos, verifica identificabilidade sazonal, seleciona por evidência, mede autocorrelação e faz validação leave-one-year-out. |
| `thwaites/timeseries/acceleration.py` | Avalia a evidência do termo quadrático por critérios de magnitude, significância, estabilidade e bootstrap. |
| `thwaites/timeseries/aliasing.py` | Calcula a resposta do estimador a harmônicos sintéticos nas épocas reais e compara tendências JJA/anuais na componente observável pelo FDM. |
| `thwaites/timeseries/uncertainty.py` | Reajusta a taxa omitindo um ano por vez e substitui o erro formal por incerteza jackknife quando há suporte suficiente. |

### Interpolação e validação espacial

| Módulo | Responsabilidade |
|---|---|
| `thwaites/interp/methods.py` | Implementa IDW, OI/Markov, krigagem ordinária local, kernel gaussiano e mediana por vizinhos, retornando predição e variância. |
| `thwaites/interp/variogram.py` | Calcula variograma empírico, ajusta modelos candidatos e fornece a função de semivariância usada pelos métodos geoestatísticos. |
| `thwaites/interp/select.py` | Cria folds em blocos, calcula RMSE/viés/calibração, seleciona o interpolador e propaga também a incerteza de entrada para a grade. |
| `thwaites/validation/folds.py` | Gera folds espaciais com buffer, por trilha ou por tempo, e verifica que não há compartilhamento indevido de observações. |
| `thwaites/validation/evaluate.py` | Reajusta nós usando somente treino, prevê observações retidas e resume métricas por fold e método. |
| `thwaites/validation/agreement.py` | Pareia crossovers e nós, estima diferenças e regressão robusta, analisa estrutura espacial e localiza áreas de discordância. |

### Glaciologia, oceano e diagnósticos

| Módulo | Responsabilidade |
|---|---|
| `thwaites/glaciology/advection.py` | Amostra declividade superficial, calcula `v·∇h`, converte taxa Euleriana em lagrangiana e testa sensibilidade à escala de suavização. |
| `thwaites/glaciology/flux.py` | Lê velocidade/espessura, calcula `∇·(H·v)`, trata amplificação hidrostática no domínio flutuante e deriva derretimento basal. |
| `thwaites/glaciology/trajectory.py` | Interpola velocidade anual no espaço/tempo e integra trajetórias de parcelas, retornando deslocamento e cobertura temporal. |
| `thwaites/ocean/bas_melt.py` | Valida o conjunto BAS/ITGC MELT, converte propriedades hidrográficas, calcula temperatura de congelamento e resume forçantes/harmônicos. |
| `thwaites/ocean/glorys.py` | Calcula espessuras de camada e métricas JJA de temperatura, salinidade e transporte a partir do GLORYS. |
| `thwaites/diagnostics/vulnerability.py` | Agrega basal, contraste sazonal, tendência de velocidade, declive do leito ao longo do fluxo e consenso espacial de afinamento. |
| `thwaites/validate/velocity.py` | Amostra e agrega velocidade nos nós, estima aceleração de fluxo, distância à grounding line, tamanho amostral efetivo e classes conjuntas dh/dt–velocidade. |

### Propagação de incerteza

| Módulo | Responsabilidade |
|---|---|
| `thwaites/uncertainty/geolocation.py` | Modela erro vertical devido a erro horizontal sobre declive, calibra com QC e estima sua contribuição à incerteza de dh/dt. |
| `thwaites/uncertainty/error_correlation.py` | Estima comprimento de correlação dos resíduos de CV ou crossovers e compara essa escala com a do sinal, evitando usar o variograma de dh/dt como erro. |
| `thwaites/uncertainty/mass_balance.py` | Aplica máscara de cobertura, combina componentes branca e correlacionada, integra massa e converte Gt/ano para equivalente de nível do mar. |

### Visualização reutilizável

| Módulo | Responsabilidade |
|---|---|
| `thwaites/viz/basemap.py` | Carrega hillshade, desenha costa/grounding line/frentes e adiciona escala cartográfica em EPSG:3031. |
| `thwaites/viz/figures.py` | Gera mapas e histogramas de dh/dt, validação por crossover, mapas de incerteza e significância e painéis de confiança. |
| `thwaites/viz/glaciology.py` | Gera mapas de basal, relação dh/dt–velocidade e diagrama de orçamento de massa. |
| `thwaites/viz/produtos.py` | Carrega BedMachine/velocidade e deriva altura de flutuação, tempo até desaterramento, grades, fluxo por portão e perfis. |
| `thwaites/viz/qc_maps.py` | Produz mapas de máscara, pontos mantidos/removidos, número de observações, distribuição temporal e confiabilidade. |

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
