from typing import Any, Callable, Dict, Optional, Sequence

import torch
from torch.utils.data import Dataset

from src.config.schema import DataConfig

from .manifest_io import load_tensor_from_pt, resolve_path


def apply_window_and_normalize(image_hu: torch.Tensor, ww: Optional[float], wc: Optional[float]) -> torch.Tensor:
    x = image_hu.clone().float()
    if ww is not None and wc is not None:
        lwin = wc - ww / 2.0
        rwin = wc + ww / 2.0
        x = torch.clamp(x, min=lwin, max=rwin)
    x = x - x.min()
    max_val = x.max().clamp_min(1e-6)
    x = x / max_val
    return x


class ManifestDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Dict],
        data_cfg: DataConfig,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        self.records = list(records)
        self.data_cfg = data_cfg
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.records[index]
        image_path = resolve_path(self.data_cfg.image_base_dir, row["image_path"])
        if image_path is None:
            raise ValueError("image_path is required")

        image_hu = load_tensor_from_pt(image_path, field_name="image_path", index=index).float()
        if image_hu.ndim == 3:
            image_hu = image_hu.unsqueeze(0)
        image = apply_window_and_normalize(image_hu, self.data_cfg.ww, self.data_cfg.wc)

        item: Dict[str, torch.Tensor] = {
            "image": image,
            "image_hu": image_hu,
            "label": torch.tensor(int(row[self.data_cfg.label_key]), dtype=torch.long),
            "index": torch.tensor(index, dtype=torch.long),
        }

        t_score = row.get(self.data_cfg.t_score_key)
        if t_score is not None:
            item["t_score"] = torch.tensor(float(t_score), dtype=torch.float32)

        bone_path = resolve_path(self.data_cfg.bone_base_dir, row.get("bone_path"))
        if bone_path:
            bone = load_tensor_from_pt(bone_path, field_name="bone_path", index=index).float()
            if bone.ndim == 3:
                bone = bone.unsqueeze(0)
            item["bone"] = bone

        nonbone_path = resolve_path(self.data_cfg.nonbone_base_dir, row.get("nonbone_path"))
        if nonbone_path:
            nonbone = load_tensor_from_pt(nonbone_path, field_name="nonbone_path", index=index).float()
            if nonbone.ndim == 3:
                nonbone = nonbone.unsqueeze(0)
            item["nonbone"] = nonbone

        if self.transform is not None:
            item = self.transform(item)

        item["patient_id"] = row[self.data_cfg.patient_id_key]
        return item
