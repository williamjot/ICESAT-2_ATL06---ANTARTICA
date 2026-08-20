"""
thwaites.config
===============
Carregamento e validação da configuração do pipeline.

- Lê `config/default.yaml` e opcionalmente um perfil (`jja`, `anual`, ...)
  que sobrescreve seções via merge profundo.
- Valida com pydantic (chaves desconhecidas são ERRO — pega typo no YAML).
- Resolve todos os caminhos a partir da raiz do projeto, sem caminhos absolutos
  embutidos no código.

Uso:
    from thwaites import load_config
    cfg = load_config()            # base
    cfg = load_config("anual")     # base + perfil anual
    cfg.paths.processed            # -> .../data/processed
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

# Raiz do projeto = pasta que contém `thwaites/` e `config/`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


# ---------------------------------------------------------------------------
# Modelos (extra="forbid" => YAML com chave estranha falha na validação)
# ---------------------------------------------------------------------------
class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProjectCfg(_Base):
    name: str
    experiment: str = "default"


class AreaCfg(_Base):
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    epsg_lonlat: int = 4326
    epsg_polar: int = 3031
    # Nome legível da região, usado em títulos e rodapés de figura. Existe porque
    # os rótulos estavam escritos "Thwaites" no código: ao expandir a ROI para o
    # Amundsen Sea Embayment as figuras passariam a afirmar uma região errada
    # (o domínio inclui Pine Island e Smith/Kohler/Pope), sem nenhum erro visível.
    label: Optional[str] = None

    @field_validator("lon_max")
    @classmethod
    def _lon_order(cls, v, info):
        if "lon_min" in info.data and v <= info.data["lon_min"]:
            raise ValueError("lon_max deve ser > lon_min")
        return v

    @field_validator("lat_max")
    @classmethod
    def _lat_order(cls, v, info):
        if "lat_min" in info.data and v <= info.data["lat_min"]:
            raise ValueError("lat_max deve ser > lat_min")
        return v

    @property
    def bounding_box(self) -> tuple[float, float, float, float]:
        """(lon_oeste, lat_sul, lon_leste, lat_norte) — ordem do earthaccess."""
        return (self.lon_min, self.lat_min, self.lon_max, self.lat_max)


class TemporalCfg(_Base):
    year_start: int
    year_end: int

    @field_validator("year_end")
    @classmethod
    def _year_order(cls, v, info):
        if "year_start" in info.data and v < info.data["year_start"]:
            raise ValueError("year_end deve ser >= year_start")
        return v

    @property
    def temporal_range(self) -> tuple[str, str]:
        return (f"{self.year_start}-01-01", f"{self.year_end}-12-31")


class SeasonCfg(_Base):
    name: str
    months: list[int]

    @field_validator("months")
    @classmethod
    def _valid_months(cls, v):
        if not v:
            raise ValueError("season.months não pode ser vazio")
        if any(m < 1 or m > 12 for m in v):
            raise ValueError("season.months deve conter valores 1..12")
        return v


class ProductCfg(_Base):
    # Decisão metodológica do projeto: nenhum outro produto ICESat-2 pode
    # alimentar os produtos científicos. Produtos auxiliares não-ICESat-2
    # continuam permitidos e são configurados nas seções próprias.
    short_name: Literal["ATL06"]
    version: str
    beams: list[str]
    segments_group: str
    geophysical_group: str
    variables: dict[str, str]
    correction_vars: dict[str, str]
    atlas_epoch_year: float
    seconds_per_year: float


class QCCfg(_Base):
    quality_max: int
    fill_value: float
    h_min_valid: float


class MaskCfg(_Base):
    bedmachine_path: str
    mask_variable: str
    keep_values: list[int]
    floating_class: int


class GroundedCfg(_Base):
    """Recorte científico para gelo aterrado (ver bloco `grounded` na config)."""
    enabled: bool = True
    buffer_grounding_line_m: float = 5000.0
    buffer_coast_m: float = 2000.0
    filter_nodes: bool = True
    min_grounded_fraction: float = 0.5


class Atl06QcCfg(_Base):
    """
    Filtro de qualidade pelos flags nativos do ATL06 (ver bloco `atl06_qc` na
    config e thwaites/qc/atl06_flags.py). Limiar None = critério DESLIGADO.
    """
    enabled: bool = False
    flags_dir: str = "qc_flags"
    require_quality_summary_zero: bool = True
    max_h_robust_sprd_m: Optional[float] = None
    max_snr_significance: Optional[float] = None
    min_n_fit_photons: Optional[int] = None
    max_msw_flag: Optional[int] = None
    max_cloud_flg_asr: Optional[int] = None
    max_bsnow_conf: Optional[int] = None
    max_dem_diff_m: Optional[float] = None


class ShelfCfg(_Base):
    """
    Domínio de PLATAFORMA (gelo flutuante) — produto separado do aterrado.

    Máscara própria, não negação da aterrada. Buffers com justificativa
    distinta: `buffer_grounding_zone_m` afasta da zona onde a flutuação não é
    plena (hipótese hidrostática falha); `buffer_front_m` afasta da frente de
    calving e margens laterais.
    """
    enabled: bool = False
    buffer_grounding_zone_m: float = 5000.0
    buffer_front_m: float = 5000.0
    dynamic_front_enabled: bool = True
    fronts_path: str = "calving_fronts.parquet"
    front_densify_step_m: float = 500.0
    front_assignment_max_m: float = 150_000.0
    front_max_search_m: float = 100_000.0
    # Alvo temporal e corte absoluto separados. O primeiro rotula a melhor
    # evidência (aprox. 90 dias); o segundo impede extrapolação maior que
    # aproximadamente um semestre.
    front_preferred_epoch_gap_years: float = 0.25
    front_max_epoch_gap_years: float = 0.50
    front_fail_closed: bool = True
    # Rótulo obrigatório: com máscara estática o produto é EXPLORATÓRIO.
    exploratory: bool = True


class CorrectionsCfg(_Base):
    apply: list[str]
    gate_to_floating: bool
    fill_threshold: float
    # `tide_equilibrium` é separado do `tide_ocean` no ATL06. A escolha precisa
    # ser explícita para impedir tanto omissão silenciosa quanto dupla correção
    # após substituir GOT por um modelo regional.
    equilibrium_tide_mode: Literal[
        "excluded_unverified", "apply_atl06", "included_in_replacement",
    ] = "excluded_unverified"


class SlopeCfg(_Base):
    enabled: bool = True
    rema_path: str = "REMA_thwaites_32m.tif"   # GeoTIFF EPSG:3031 (fetch_rema.py)
    max_slope_ref_m: float = 200.0             # |h - REMA| acima disso = blunder -> sem correção


class FilttrackCfg(_Base):
    enabled: bool = True
    window: int = 21
    n_sigma: float = 5.0
    gap_year: float = 1.0e-4
    mad_floor_m: float = 0.1
    mad_window_factor: int = 5    # janela do MAD local = window * fator


class CatsCfg(_Base):
    """Maré CATS2008 via pyTMD (substitui o GOT4.8 embutido no ATL06)."""
    enabled: bool = False          # ativar após baixar o modelo manualmente
    model_dir: str = "tide_models"  # pasta com CATS2008_v2023.nc (sob data/)
    model_file: str = "CATS2008_v2023.nc"
    model_name: str = "CATS2008-v2023"
    method: str = "linear"         # interpolação das constantes harmônicas ("linear"|"nearest")
    apply_flexure: bool = False    # escala a maré na zona de aterramento (flexão da plataforma)
    extrapolate: bool = True       # extrapola até `cutoff_km` da grade do modelo
    cutoff_km: float = 10.0
    # memória: o modelo é recortado à ROI e os pontos processados em chunks
    chunk_size: int = 100_000      # pontos por chunk de predição
    row_batch: int = 500_000       # linhas por row group na leitura/escrita streaming
    buffer_km: float = 50.0        # buffer do recorte do modelo à volta da ROI
    dask_chunks: Optional[int] = None  # chunking dask na leitura (None = eager)
    # Declarado pelo usuário/modelo após auditoria de constituintes. `None`
    # significa que não se sabe se a previsão de substituição já inclui LPE.
    equilibrium_tide_included: Optional[bool] = None


class SensitivityCfg(_Base):
    """
    Prioridade 2 (§3): sensibilidade dos filtros.

    Os limiares de aceite são PRÉ-DEFINIDOS aqui e registrados no manifesto
    ANTES de qualquer comparação (§3.5) — não podem ser escolhidos depois de
    ver os resultados.
    """
    n_regions: int = 6            # sub-regiões representativas (tratabilidade)
    region_km: float = 50.0       # lado de cada sub-região
    seed: int = 0
    bootstrap_iters: int = 1000   # IC bootstrap das diferenças pareadas
    # --- critérios de aceite PRÉ-DEFINIDOS (§3.5) ---------------------------
    max_median_dhdt_shift: float = 0.05   # |Δ mediana de dh/dt| tolerada (m/ano)
    max_xover_dispersion_increase: float = 0.10  # +10% na dispersão dos crossovers
    min_coverage_gain: float = 0.02       # ganho de cobertura p/ "material" (2%)
    max_residual_increase: float = 0.10   # +10% no RMSE do ajuste


class VelocityCfg(_Base):
    """Velocidade do gelo MEaSUREs (backlog #2 + insumo da divergência de fluxo)."""
    short_name: str = "NSIDC-0754"     # Phase-Based Antarctica Ice Velocity Map v1
    version: str = "1"
    vx_var: str = "VX"
    vy_var: str = "VY"
    path: str = "velocity_thwaites.nc"  # recorte gerado por fetch_velocity.py
    # descasamento temporal a declarar: mosaico 1996–2018 vs dh/dt 2019–2025
    epoch_note: str = "MEaSUREs NSIDC-0754 (1996-2018); dh/dt 2019-2025"


class FirnCfg(_Base):
    """
    Correção de firn (GSFC-FDM v1.2.1, Zenodo 7221954).

    Separa mudança de altura de mudança de massa: a densificação do firn baixa
    a superfície sem perder massa. Sem esta correção, ρ=917 sobre o dh/dt bruto
    atribui compactação de firn a perda de gelo.
    """
    enabled: bool = False           # ativar após rodar fetch_firn.py
    path: str = "firn_thwaites.nc"  # recorte gerado por fetch_firn.py
    fac_var: Optional[str] = None   # None => detecta (FAC/fac/firn_air_content)
    smb_var: str = "SMB_a"          # ANOMALIA cumulativa de SMB (não SMB total!)
    smb_total_var: str = "SMB"      # SMB TOTAL, já em m gelo/ANO (arquivo de componentes)
    smb_path: str = "smb_thwaites.nc"
    # [DECLARAR] o FDM termina em 30/06/2022 e o dh/dt vai a 2025 -> a taxa de
    # FAC é estimada na sobreposição e extrapolada.
    epoch_note: str = "GSFC-FDM v1.2.1 (1980-06/2022); dh/dt 2019-2025"


class FluxCfg(_Base):
    """Divergência de fluxo e derretimento basal (cubediv/cubemelt)."""
    enabled: bool = False
    grid_res_m: float = 1000.0     # resolução do cálculo de ∇·(H·v)
    smooth_km: float = 2.0         # suavização de H e v antes de derivar (ruído)
    water_density: float = 1027.0  # kg/m³ — água do mar (flutuação hidrostática)
    smb_path: Optional[str] = None  # [OPCIONAL] balanço de massa superficial (m gelo/ano)
    min_thickness_m: float = 10.0  # abaixo disso o cálculo é instável


class FiltstCfg(_Base):
    """Filtro espaço-temporal por binagem (ver thwaites/qc/filtst.py)."""
    enabled: bool = False     # complementar ao filttrack; ativar se a cauda persistir
    cell_km: float = 5.0      # lado da célula espacial
    dt_years: float = 1.0     # janela temporal da célula
    n_sigma: float = 5.0
    min_count: int = 20       # mínimo de pontos na célula p/ estatística robusta
    mad_floor_m: float = 0.1
    passes: int = 2           # 2ª passagem com grade deslocada (reduz efeito de borda)


class XoverCfg(_Base):
    enabled: bool = True
    subsample: int = 10           # subamostragem dos pontos p/ a geometria da trilha
    min_track_points: int = 50    # trilhas curtas demais não geram cruzamento útil
    fit_radius_m: float = 200.0   # raio do ajuste de plano local no cruzamento
    min_points: int = 4           # mínimo de pontos por trilha no ajuste local
    min_dt_years: float = 0.5     # |Δt| mínimo p/ dh/dt (evita dividir por ~0)
    # Custo: o nº de cruzamentos cresce com (n_asc × n_desc). Usar TODAS as
    # trilhas dá ~1,25 M cruzamentos (~86 min). Para VALIDAÇÃO isso é
    # desperdício — uma amostra aleatória (espacialmente não-enviesada) de
    # trilhas já determina a mediana com precisão muito além do necessário.
    # null = usar todas.
    max_tracks_per_direction: Optional[int] = 400
    seed: int = 0


class TilesCfg(_Base):
    tile_km: float
    halo_km: float
    chunk: int

    @property
    def tile_m(self) -> float:
        return self.tile_km * 1000.0

    @property
    def halo_m(self) -> float:
        return self.halo_km * 1000.0


class DhdtCfg(_Base):
    search_radius_m: float
    node_spacing_m: float                # resolução dos nós do ajuste (computacional)
    min_points: int
    dt_min_years: float
    dt_min_years_accel: float
    t_ref: float
    rate_limit: float
    accel_limit: float
    poly_order: int
    temp_order: int
    use_weights: bool
    max_iter: int
    n_sigma: float
    resid_limit: float


class TimeseriesCfg(_Base):
    node_spacing_m: float
    search_radius_m: float
    min_points_per_epoch: int
    min_epochs: int


class CVCfg(_Base):
    strategy: str
    block_km: float
    n_folds: int
    seed: int = 0


class VariogramCfg(_Base):
    lag_m: Optional[float]               # [DATA-DRIVEN]
    model: Optional[str]                 # [DATA-DRIVEN]
    max_lag_m: float
    n_lags: int = 15


class InterpolationCfg(_Base):
    candidates: list[str]
    cv: CVCfg
    variogram: VariogramCfg
    neighbors: int = 32
    idw_power: float = 2.0
    grid_res_m: float = 5000.0
    # [DATA-DRIVEN] se null, derivados do alcance do variograma (range/3, range/2)
    gauss_sigma_m: Optional[float] = None
    median_radius_m: Optional[float] = None


class TrendCfg(_Base):
    alpha: float
    fdr_method: str
    mk_variant: str


class SeaLevelCfg(_Base):
    """
    SSHA do oceano coberto por gelo marinho (ATL21 gridado / ATL07 along-track).

    Produto SEPARADO do balanço de massa: mede topografia dinâmica do oceano,
    não contribuição da geleira ao nível do mar (essa é `mass_balance`, via
    dM/dt -> mm SLE). Misturar os dois seria erro de interpretação, não de
    implementação.
    """
    short_name: str = "ATL21"
    version: str = "004"
    path: str = "atl21_ase_ssha.parquet"
    # Célula-mês com poucas superfícies de referência não é uma média mensal —
    # é uma medida isolada com o rótulo de média. Medido em ago/2022 (pico de
    # gelo marinho): das 447 células da ROI, só 21 tinham SSHA, com mediana de
    # 2 refsurfs. O limiar reflete esse regime de amostragem.
    min_refsurfs: int = 2
    # Meses distintos exigidos por célula para testar tendência. Com 84 meses
    # possíveis (2019-2025), 24 é ~29% de preenchimento.
    min_months: int = 24
    # Anos distintos exigidos (o MK sazonal compara mesmo mês entre anos).
    min_years: int = 4
    grid_res_m: float = 25000.0


class MassBalanceCfg(_Base):
    ice_density: float
    ocean_area_m2: float
    correlation_length_m: Optional[float] = None
    coverage_dist_m: float = 7500.0


class LoggingCfg(_Base):
    level: str = "INFO"
    rotation: str = "10 MB"
    retention: str = "30 days"


class Paths(_Base):
    """Caminhos resolvidos a partir da raiz do projeto."""
    base_dir: Path
    data_dir: Path
    raw_temp: Path          # download temporário (um .h5 por vez, deletado)
    processed: Path         # CSVs/Parquets leves por grânulo
    interim: Path           # intermediários (merged, masked, ...)
    tiles_dir: Path
    dhdt_dir: Path
    qc_flags: Path
    timeseries_dir: Path
    outputs_dir: Path
    figures: Path
    tables: Path
    qgis: Path
    logs: Path
    season: Optional[str] = None

    @classmethod
    def from_base(cls, base_dir: Path, season: str | None = None) -> "Paths":
        """
        Caminhos do projeto, com os DERIVADOS isolados por estação.

        Por que a estação entra no caminho
        ----------------------------------
        `processed`, `interim`, `tiles_dir` e `dhdt_dir` dependem da estação.
        Como os pipelines aceitam `--profile`, caminhos fixos fariam perfis
        diferentes sobrescreverem silenciosamente produtos intermediários,
        inclusive `atl06_merged.parquet` e `dhdt_nodes.parquet`.

        O que NÃO é isolado
        -------------------
        `data_dir` continua compartilhado, e é ali que ficam os dados de
        REFERÊNCIA: REMA (640 MB), BedMachine, CATS2008, GSFC-FDM, ITS_LIVE,
        Caron. Nenhum deles depende da janela sazonal, e duplicá-los custaria
        dezenas de GB para guardar bytes idênticos.

        `raw_temp` também é compartilhado de propósito: é a pasta do download
        temporário, que por regra do projeto tem de estar VAZIA entre
        execuções. Dar a ela um caminho por estação criaria vários lugares
        onde um `.h5` órfão pode se esconder.
        """
        data = base_dir / "data"
        out = base_dir / "outputs"
        # derivados por estação; sem estação, mantém o layout plano
        d = data / season if season else data
        o = out / season if season else out
        return cls(
            base_dir=base_dir,
            data_dir=data,
            raw_temp=data / "raw_temp",
            processed=d / "processed",
            interim=d / "interim",
            tiles_dir=d / "tiles",
            dhdt_dir=d / "dhdt",
            qc_flags=d / "qc_flags",
            timeseries_dir=d / "timeseries",
            outputs_dir=o,
            figures=o / "figures",
            tables=o / "tables",
            qgis=o / "qgis",
            logs=base_dir / "logs",
            season=season,
        )

    def ensure(self) -> None:
        """Cria as pastas de saída (idempotente)."""
        for p in (self.raw_temp, self.processed, self.interim, self.tiles_dir,
                  self.dhdt_dir, self.qc_flags, self.timeseries_dir, self.figures,
                  self.tables, self.qgis, self.logs):
            p.mkdir(parents=True, exist_ok=True)


class Config(_Base):
    project: ProjectCfg
    area: AreaCfg
    roi: Optional[AreaCfg] = None        # recorte de processamento (ex.: Thwaites)
    temporal: TemporalCfg
    season: SeasonCfg
    product: ProductCfg
    qc: QCCfg
    mask: MaskCfg
    grounded: GroundedCfg = GroundedCfg()
    atl06_qc: Atl06QcCfg = Atl06QcCfg()
    shelf: ShelfCfg = ShelfCfg()
    corrections: CorrectionsCfg
    slope: SlopeCfg = SlopeCfg()
    cats: CatsCfg = CatsCfg()
    filttrack: FilttrackCfg = FilttrackCfg()
    filtst: FiltstCfg = FiltstCfg()
    sensitivity: SensitivityCfg = SensitivityCfg()
    firn: FirnCfg = FirnCfg()
    velocity: VelocityCfg = VelocityCfg()
    flux: FluxCfg = FluxCfg()
    xover: XoverCfg = XoverCfg()
    tiles: TilesCfg
    dhdt: DhdtCfg
    timeseries: TimeseriesCfg
    interpolation: InterpolationCfg
    trend: TrendCfg
    sealevel: SeaLevelCfg = SeaLevelCfg()
    mass_balance: MassBalanceCfg
    logging: LoggingCfg
    paths: Paths


# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------
def _deep_merge(base: dict, override: dict) -> dict:
    """Merge recursivo: valores do override vencem; dicts são fundidos."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(
    profile: Optional[str] = None,
    base_dir: Optional[Path] = None,
    config_dir: Optional[Path] = None,
) -> Config:
    """
    Carrega a configuração base e, opcionalmente, um perfil que a sobrescreve.

    Parâmetros
    ----------
    profile : str | None
        Nome de um perfil em config/ (ex.: "jja", "anual"). Sem extensão.
    base_dir : Path | None
        Raiz do projeto (para resolver caminhos). Default: raiz detectada.
    config_dir : Path | None
        Pasta dos YAMLs. Default: <raiz>/config.

    Retorna
    -------
    Config validado, com `paths` resolvidos.
    """
    base_dir = Path(base_dir) if base_dir else PROJECT_ROOT
    config_dir = Path(config_dir) if config_dir else (base_dir / "config")

    default_path = config_dir / "default.yaml"
    if not default_path.exists():
        raise FileNotFoundError(f"config base não encontrada: {default_path}")

    with open(default_path, "r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp)

    if profile:
        profile_path = config_dir / f"{profile}.yaml"
        if not profile_path.exists():
            raise FileNotFoundError(f"perfil não encontrado: {profile_path}")
        with open(profile_path, "r", encoding="utf-8") as fp:
            override = yaml.safe_load(fp) or {}
        data = _deep_merge(data, override)

    # A estação sai da config JÁ MESCLADA com o perfil — é o `season.name`
    # efetivo que define onde os derivados são escritos, não o nome do arquivo
    # de perfil. Assim `--profile jja` e a default (que também é JJA) apontam
    # para a MESMA pasta, em vez de criarem dois produtos silenciosamente
    # divergentes com o mesmo conteúdo.
    season = (data.get("season") or {}).get("name")
    data["paths"] = Paths.from_base(base_dir, season=season).model_dump()
    return Config(**data)
