import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from src.config.settings import settings
from src.mlops.config.mlops_settings import PROJECT_ROOT, mlops_settings
from src.mlops.registry.model_factory import create_model
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.models.base import BaseMLModel

logger = get_logger(__name__)


class ModelStage(str, Enum):
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass
class ModelRecord:
    model_name: str
    version: int
    stage: ModelStage
    path: str  # relative path string, JSON-serializable
    metrics: dict[str, float]
    trained_at: str
    dataset_version: str
    mlflow_run_id: str | None = None
    git_commit: str | None = None


_STAGE_DIR_ATTR = {
    ModelStage.STAGING: "model_staging_dir_path",
    ModelStage.PRODUCTION: "model_production_dir_path",
    ModelStage.ARCHIVED: "model_archived_dir_path",
}


class ModelRegistry:
    """Repository pattern over a flat-file model registry:
    models/registry/<name>/<version>/model.pkl + models/registry/<name>/index.json,
    with flat "current pointer" copies at models/{staging,production,archived}/<name>.pkl.
    """

    def __init__(self, registry_dir: Path | None = None) -> None:
        self.registry_dir = registry_dir or mlops_settings.model_registry_dir_path

    # -- internal index helpers -------------------------------------------------

    def _model_dir(self, model_name: str) -> Path:
        return self.registry_dir / model_name

    def _index_path(self, model_name: str) -> Path:
        return self._model_dir(model_name) / "index.json"

    def _read_index(self, model_name: str) -> list[dict]:
        index_path = self._index_path(model_name)
        if not index_path.exists():
            return []
        return json.loads(index_path.read_text())

    def _write_index(self, model_name: str, records: list[dict]) -> None:
        index_path = self._index_path(model_name)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps(records, indent=2))

    def _record_from_dict(self, data: dict) -> ModelRecord:
        data = dict(data)
        data["stage"] = ModelStage(data["stage"])
        return ModelRecord(**data)

    def _stage_dir(self, stage: ModelStage) -> Path:
        return getattr(mlops_settings, _STAGE_DIR_ATTR[stage])

    def _flat_path(self, stage: ModelStage, model_name: str) -> Path:
        return self._stage_dir(stage) / f"{model_name}.pkl"

    # -- write operations ---------------------------------------------------

    def register_model(
        self,
        model_name: str,
        source_path: Path,
        metrics: dict[str, float],
        dataset_version: str,
        mlflow_run_id: str | None = None,
    ) -> ModelRecord:
        """Copy source_path into the registry as a new version, initial stage STAGING."""
        records = self._read_index(model_name)
        version = len(records) + 1

        dest_path = self._model_dir(model_name) / str(version) / "model.pkl"
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_path)

        try:
            path_str = str(dest_path.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            # Custom (e.g. temp-dir) registry_dir outside PROJECT_ROOT: store an absolute path.
            path_str = str(dest_path.resolve())

        record = ModelRecord(
            model_name=model_name,
            version=version,
            stage=ModelStage.STAGING,
            path=path_str,
            metrics=metrics,
            trained_at=datetime.now(timezone.utc).isoformat(),
            dataset_version=dataset_version,
            mlflow_run_id=mlflow_run_id,
            git_commit=None,
        )
        records.append(asdict(record))
        self._write_index(model_name, records)
        logger.info("Registered model=%s version=%d stage=%s", model_name, version, record.stage.value)
        return record

    def promote(self, model_name: str, version: int, to: ModelStage) -> ModelRecord:
        """Promote a specific version to `to` stage; promoting to PRODUCTION archives
        whatever was previously in PRODUCTION for this model_name.
        """
        records = self._read_index(model_name)
        target = None
        for rec in records:
            if rec["version"] == version:
                target = rec
                break
        if target is None:
            raise FileNotFoundError(f"No record for model_name={model_name!r} version={version}")

        if to == ModelStage.PRODUCTION:
            for rec in records:
                if rec["version"] != version and rec["stage"] == ModelStage.PRODUCTION.value:
                    rec["stage"] = ModelStage.ARCHIVED.value
                    old_prod_flat = self._flat_path(ModelStage.PRODUCTION, model_name)
                    archived_flat = self._flat_path(ModelStage.ARCHIVED, model_name)
                    if old_prod_flat.exists():
                        archived_flat.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(old_prod_flat), str(archived_flat))
                    logger.info(
                        "Archived previous production version=%d of model=%s",
                        rec["version"],
                        model_name,
                    )

        target["stage"] = to.value
        self._write_index(model_name, records)

        source_registry_path = self._model_dir(model_name) / str(version) / "model.pkl"
        dest_flat = self._flat_path(to, model_name)
        dest_flat.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_registry_path, dest_flat)

        logger.info("Promoted model=%s version=%d to stage=%s", model_name, version, to.value)
        return self.get_record(model_name, version)

    def rollback(self, model_name: str, to_version: int | None = None) -> ModelRecord:
        """Re-promote a previous (or explicit) version back to PRODUCTION."""
        if to_version is not None:
            return self.promote(model_name, to_version, ModelStage.PRODUCTION)

        records = self._read_index(model_name)
        archived_versions = sorted(
            (rec["version"] for rec in records if rec["stage"] == ModelStage.ARCHIVED.value),
            reverse=True,
        )
        if not archived_versions:
            raise ValueError(
                f"No archived version available to roll back to for model_name={model_name!r}"
            )
        return self.promote(model_name, archived_versions[0], ModelStage.PRODUCTION)

    def archive_model(self, model_name: str, version: int) -> ModelRecord:
        """Set a version's stage to ARCHIVED, moving its flat file if present."""
        records = self._read_index(model_name)
        target = None
        for rec in records:
            if rec["version"] == version:
                target = rec
                break
        if target is None:
            raise FileNotFoundError(f"No record for model_name={model_name!r} version={version}")

        previous_stage = ModelStage(target["stage"])
        target["stage"] = ModelStage.ARCHIVED.value
        self._write_index(model_name, records)

        if previous_stage in (ModelStage.STAGING, ModelStage.PRODUCTION):
            old_flat = self._flat_path(previous_stage, model_name)
            archived_flat = self._flat_path(ModelStage.ARCHIVED, model_name)
            if old_flat.exists():
                archived_flat.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(old_flat), str(archived_flat))
        else:
            source_registry_path = self._model_dir(model_name) / str(version) / "model.pkl"
            archived_flat = self._flat_path(ModelStage.ARCHIVED, model_name)
            archived_flat.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_registry_path, archived_flat)

        return self.get_record(model_name, version)

    # -- read operations ------------------------------------------------------

    def get_record(self, model_name: str, version: int | None = None) -> ModelRecord:
        records = self._read_index(model_name)
        if not records:
            raise FileNotFoundError(f"No registry records for model_name={model_name!r}")
        if version is None:
            best = max(records, key=lambda r: r["version"])
            return self._record_from_dict(best)
        for rec in records:
            if rec["version"] == version:
                return self._record_from_dict(rec)
        raise FileNotFoundError(f"No record for model_name={model_name!r} version={version}")

    def list_versions(self, model_name: str) -> list[ModelRecord]:
        return [self._record_from_dict(rec) for rec in self._read_index(model_name)]

    def _load_from_record(self, model_name: str, record: ModelRecord) -> "BaseMLModel":
        model = create_model(model_name)
        record_path = Path(record.path)
        path = record_path if record_path.is_absolute() else PROJECT_ROOT / record_path
        model.load(path)
        return model

    def load_latest(self, model_name: str) -> "BaseMLModel":
        record = self.get_record(model_name)
        return self._load_from_record(model_name, record)

    def load_staging(self, model_name: str) -> "BaseMLModel":
        flat_path = self._flat_path(ModelStage.STAGING, model_name)
        model = create_model(model_name)
        if flat_path.exists():
            model.load(flat_path)
            return model
        raise FileNotFoundError(f"No staging model file for model_name={model_name!r}")

    def load_production(self, model_name: str) -> "BaseMLModel":
        """Load the production model, falling back to the legacy flat
        models/<name>.pkl file if nothing has been registered/promoted yet
        (keeps Phase 1-5 fully working from day one).
        """
        flat_path = self._flat_path(ModelStage.PRODUCTION, model_name)
        has_production_record = False
        try:
            records = self._read_index(model_name)
            has_production_record = any(
                rec["stage"] == ModelStage.PRODUCTION.value for rec in records
            )
        except Exception:
            has_production_record = False

        model = create_model(model_name)
        if flat_path.exists() or has_production_record:
            if flat_path.exists():
                model.load(flat_path)
                return model

        legacy_path = settings.model_dir_path / f"{model_name}.pkl"
        if legacy_path.exists():
            logger.warning(
                "No production registry entry for model=%s; falling back to legacy path=%s",
                model_name,
                legacy_path,
            )
            model.load(legacy_path)
            return model

        raise FileNotFoundError(
            f"No production model found for model_name={model_name!r} "
            f"(checked registry production dir {flat_path} and legacy path {legacy_path})"
        )
