"""
thwaites.experiments.manifest
=============================
Manifesto de reprodutibilidade (§8 do PLANO_PRIORIDADES_CIENTIFICAS).

Registra tudo que é necessário para reproduzir um experimento:
data/hora, commit do código, hash da configuração, produto e versão dos dados,
período e ROI, checksums das entradas, colunas usadas, parâmetros de filtro,
correções aplicadas, versões de dependências, semente, partições e checksums
das saídas.

Princípio: um experimento grava em `outputs/experiments/<nome>/` e NUNCA
sobrescreve outro. Se o diretório existir, é erro (a não ser com overwrite
explícito) — evita perder silenciosamente uma execução anterior.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from thwaites.config import Config

# dependências cujas versões afetam resultados numéricos
_TRACKED_PACKAGES = (
    "numpy", "scipy", "pandas", "pyarrow", "h5py", "xarray",
    "statsmodels", "shapely", "rasterio", "pyproj", "pyTMD",
)


def file_checksum(path: str | Path, algo: str = "sha256",
                  max_bytes: int | None = 256 * 1024 * 1024) -> dict:
    """
    Checksum de um arquivo. Para arquivos grandes, amostra início e fim, além
    de tamanho e mtime_ns. Hash só do início deixava alterações em row groups
    finais de Parquet invisíveis ao manifesto.
    """
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    h = hashlib.new(algo)
    size = path.stat().st_size
    partial = bool(max_bytes and size > max_bytes)
    read = 0
    with open(path, "rb") as fp:
        if not partial:
            while True:
                chunk = fp.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
                read += len(chunk)
        else:
            # Metade inicial + metade final: mudanças tardias nos Parquets são
            # detectadas sem exigir hash integral de arquivos multi-GB.
            sample = max(1, int(max_bytes // 2))
            for _ in range(2):
                left = sample
                while left > 0:
                    chunk = fp.read(min(1024 * 1024, left))
                    if not chunk:
                        break
                    h.update(chunk)
                    read += len(chunk)
                    left -= len(chunk)
                if _ == 0:
                    fp.seek(max(size - sample, 0))
    return {"path": str(path), "exists": True, "size_bytes": size,
            "algo": algo, "digest": h.hexdigest(), "partial": partial,
            "hashed_bytes": read, "sample_strategy": (
                "full" if not partial else "head_tail_equal") ,
            "mtime_ns": path.stat().st_mtime_ns}


def source_tree_hash(root: str | Path) -> str:
    """Hash do código/configuração usado, inclusive fora de um repositório Git."""
    root = Path(root)
    files = [root / "pyproject.toml"]
    for rel in ("thwaites", "pipelines", "config"):
        d = root / rel
        if d.exists():
            files.extend(sorted(p for p in d.rglob("*")
                                if p.is_file() and p.suffix in {".py", ".yaml", ".yml"}))
    h = hashlib.sha256()
    for p in sorted(set(files)):
        if not p.exists():
            continue
        h.update(p.relative_to(root).as_posix().encode("utf-8"))
        h.update(b"\0")
        with open(p, "rb") as fp:
            for chunk in iter(lambda: fp.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def config_hash(cfg: Config) -> str:
    """Hash estável do conteúdo da configuração (ordenado, sem caminhos absolutos)."""
    d = cfg.model_dump(mode="json")
    d.pop("paths", None)          # caminhos variam por máquina
    blob = json.dumps(d, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def git_commit(repo_dir: str | Path | None = None) -> str | None:
    """Commit atual, se o projeto estiver sob git (retorna None se não estiver)."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_dir or "."),
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _dependency_versions() -> dict:
    import importlib.metadata as md
    vers = {}
    for p in _TRACKED_PACKAGES:
        try:
            vers[p] = md.version(p)
        except Exception:
            vers[p] = None
    vers["python"] = platform.python_version()
    return vers


def experiment_dir(cfg: Config, name: str, overwrite: bool = False) -> Path:
    """
    Diretório do experimento (`outputs/experiments/<name>/`), criado.

    Protege contra sobrescrever silenciosamente os produtos de outra execução
    (§8), mas apenas de execuções CONCLUÍDAS: o marcador de conclusão é o
    `manifest.json`, escrito por último.

    Uma pasta que só tem o `config_snapshot.json` é resto de execução que morreu
    no meio (o construtor do Manifest cria a pasta antes de processar). Bloquear
    a retentativa nesse caso seria só atrapalhar — e forçaria o usuário a usar
    `overwrite` num caso em que não há nada a preservar.
    """
    d = cfg.paths.outputs_dir / "experiments" / name
    completed = (d / "manifest.json").exists()
    if completed and not overwrite:
        raise FileExistsError(
            f"O experimento '{name}' já foi CONCLUÍDO em {d} (manifest.json presente).\n"
            f"Use outro nome ou overwrite=True (isso descarta a execução anterior)."
        )
    if d.exists() and (completed or any(d.iterdir())):
        # `overwrite` é explícito; uma tentativa incompleta pode ser reiniciada.
        # Remove a árvore inteira para impedir saídas antigas não sobrescritas de
        # parecerem produto da execução nova.
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    return d


class Manifest:
    """
    Acumula os metadados de um experimento e grava `manifest.json`.

    Uso:
        man = Manifest(cfg, "sens_baseline", purpose="configuração-base")
        man.add_input(path)                     # checksum da entrada
        man.set("acceptance", {...})            # critérios PRÉ-definidos
        man.add_output(path)
        man.write()
    """

    def __init__(self, cfg: Config, name: str, purpose: str = "",
                 overwrite: bool = False, seed: int | None = None):
        self.cfg = cfg
        self.name = name
        self.dir = experiment_dir(cfg, name, overwrite=overwrite)
        roi = cfg.roi or cfg.area
        self.data: dict[str, Any] = {
            "experiment": name,
            "purpose": purpose,
            "created_utc": datetime.utcnow().isoformat() + "Z",
            "git_commit": git_commit(cfg.paths.base_dir),
            "source_tree_sha256": source_tree_hash(cfg.paths.base_dir),
            "config_hash": config_hash(cfg),
            "config": cfg.model_dump(mode="json"),
            "product": {"short_name": cfg.product.short_name,
                        "version": cfg.product.version},
            "period": {"year_start": cfg.temporal.year_start,
                       "year_end": cfg.temporal.year_end,
                       "season": cfg.season.name,
                       "months": cfg.season.months},
            "roi": {"lon_min": roi.lon_min, "lon_max": roi.lon_max,
                    "lat_min": roi.lat_min, "lat_max": roi.lat_max,
                    "epsg": cfg.area.epsg_polar},
            "corrections_applied": list(cfg.corrections.apply),
            "slope_reference": (cfg.slope.rema_path if cfg.slope.enabled else None),
            "tide_model": (cfg.cats.model_name if cfg.cats.enabled
                           else "ATL06 embutido (GOT4.8)"),
            "mask": {"path": cfg.mask.bedmachine_path,
                     "keep_values": cfg.mask.keep_values,
                     "floating_class": cfg.mask.floating_class},
            "seed": seed,
            "dependencies": _dependency_versions(),
            "inputs": [],
            "outputs": [],
            "columns_used": [],
        }
        self.data["run_id"] = (
            f"{self.data['created_utc'].replace(':', '').replace('-', '')}_"
            f"{self.data['config_hash'][:12]}_{self.data['source_tree_sha256'][:12]}")
        # a config em vigor, salva à parte para diff fácil entre experimentos
        (self.dir / "config_snapshot.json").write_text(
            json.dumps(self.data["config"], indent=2, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------------ API
    def set(self, key: str, value: Any) -> "Manifest":
        self.data[key] = value
        return self

    def add_input(self, path, columns: list[str] | None = None) -> "Manifest":
        self.data["inputs"].append(file_checksum(path))
        if columns:
            for c in columns:
                if c not in self.data["columns_used"]:
                    self.data["columns_used"].append(c)
        return self

    def add_output(self, path) -> "Manifest":
        self.data["outputs"].append(file_checksum(path))
        return self

    def path_for(self, filename: str) -> Path:
        """Caminho de uma saída dentro do diretório do experimento."""
        return self.dir / filename

    def write(self) -> Path:
        self.data["written_utc"] = datetime.utcnow().isoformat() + "Z"
        p = self.dir / "manifest.json"
        p.write_text(json.dumps(self.data, indent=2, ensure_ascii=False, default=str),
                     encoding="utf-8")
        return p
