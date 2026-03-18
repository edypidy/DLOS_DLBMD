import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch


def load_manifest(manifest_path: str) -> List[Dict]:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("manifest must be a JSON array")
    return data


def save_manifest(records: Sequence[Dict], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(list(records), f, indent=2, ensure_ascii=False)


def ensure_manifest_required_keys(
    records: Sequence[Dict],
    required_keys: Iterable[str] = ("image_path", "label", "patient_id"),
) -> None:
    required = set(required_keys)
    for idx, row in enumerate(records):
        missing = [k for k in required if k not in row]
        if missing:
            raise ValueError(f"manifest row {idx} missing keys: {missing}")


def resolve_path(base_dir: str, file_path: Optional[str]) -> Optional[str]:
    if not file_path:
        return None
    p = Path(file_path)
    if p.is_absolute():
        return str(p)
    if base_dir:
        return str((Path(base_dir) / p).resolve())
    return str(p.resolve())


def load_tensor_from_pt(path: str, field_name: str, index: int) -> torch.Tensor:
    value = torch.load(path)
    if not torch.is_tensor(value):
        raise TypeError(
            f"manifest row {index} field '{field_name}' must load to torch.Tensor, "
            f"got {type(value).__name__} from {path}."
        )
    return value
