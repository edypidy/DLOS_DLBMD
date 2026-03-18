import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from .schema import AppConfig, DataConfig, InferConfig, ModelConfig, TrainConfig


def _read_config_file(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    if config_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required for yaml config files.") from exc
        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    if config_path.suffix.lower() == ".json":
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    raise ValueError(f"Unsupported config extension: {config_path.suffix}")


def _build_app_config(raw: Dict[str, Any], mode: str) -> AppConfig:
    if "data" not in raw or "model" not in raw:
        raise ValueError("Config must include 'data' and 'model' sections.")

    data = DataConfig(**raw["data"])
    model = ModelConfig(**raw["model"])
    train = TrainConfig(**raw["train"]) if "train" in raw else None
    infer = InferConfig(**raw["infer"]) if "infer" in raw else None

    app = AppConfig(data=data, model=model, train=train, infer=infer)
    validate_config(app, mode=mode)
    app.ensure_paths()
    return app


def validate_config(cfg: AppConfig, mode: str) -> None:
    if mode == "train":
        if cfg.train is None:
            raise ValueError("train mode requires [train] section.")
        if (
            cfg.model.inverse_attention
            and cfg.train.hu_threshold is None
            and not cfg.data.use_manifest_bone_mask
        ):
            raise ValueError(
                "train.hu_threshold is required when model.inverse_attention=true "
                "and data.use_manifest_bone_mask=false."
            )
        if cfg.train.epochs < 1:
            raise ValueError("train.epochs must be >= 1")
    elif mode == "infer":
        if cfg.infer is None:
            raise ValueError("infer mode requires [infer] section.")
        if not cfg.infer.checkpoint_path:
            raise ValueError("infer.checkpoint_path is required.")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    total_ratio = cfg.data.train_ratio + cfg.data.valid_ratio + cfg.data.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("data split ratios must sum to 1.0")


def load_app_config(config_path: str, mode: str) -> AppConfig:
    raw = _read_config_file(config_path)
    return _build_app_config(raw, mode=mode)


def dump_effective_config(cfg: AppConfig, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)
