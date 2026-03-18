import math
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from torch.utils.data import DataLoader

from src.config.schema import DataConfig, TrainConfig

from .dataset import ManifestDataset
from .split import filter_by_split, prepare_manifest_records

try:
    from monai import transforms as monai_transforms
except ImportError:  # pragma: no cover - optional dependency
    monai_transforms = None


def build_dataloader(
    records: Sequence[Dict],
    data_cfg: DataConfig,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> Tuple[DataLoader, ManifestDataset]:
    transform = _build_monai_transform(data_cfg=data_cfg, train=shuffle)
    dataset = ManifestDataset(records=records, data_cfg=data_cfg, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader, dataset


def _build_monai_transform(
    data_cfg: DataConfig,
    train: bool,
) -> Optional[Callable[[Dict[str, Any]], Dict[str, Any]]]:
    if not data_cfg.use_monai_transforms:
        return None
    if monai_transforms is None:
        raise ImportError(
            "MONAI is required when data.use_monai_transforms=true. "
            "Install monai or set data.use_monai_transforms=false."
        )

    keys = ["image", "image_hu", "bone", "nonbone"]
    if not train:
        return monai_transforms.Compose(
            [
                monai_transforms.EnsureTyped(keys=keys, allow_missing_keys=True),
            ]
        )

    return monai_transforms.Compose(
        [
            monai_transforms.RandFlipd(
                keys=keys,
                prob=data_cfg.rand_flip_prob,
                spatial_axis=0,
                allow_missing_keys=True,
            ),
            monai_transforms.RandFlipd(
                keys=keys,
                prob=data_cfg.rand_flip_prob,
                spatial_axis=1,
                allow_missing_keys=True,
            ),
            monai_transforms.RandFlipd(
                keys=keys,
                prob=data_cfg.rand_flip_prob,
                spatial_axis=2,
                allow_missing_keys=True,
            ),
            monai_transforms.RandRotate90d(
                keys=keys,
                prob=data_cfg.rand_rotate90_prob,
                max_k=3,
                allow_missing_keys=True,
            ),
            monai_transforms.RandAffined(
                keys=keys,
                prob=data_cfg.rand_affine_prob,
                mode=("bilinear", "bilinear", "nearest", "nearest"),
                rotate_range=(math.pi / 12, math.pi / 12, math.pi / 12),
                scale_range=(0.1, 0.1, 0.1),
                translate_range=(5, 5, 5),
                padding_mode="zeros",
                allow_missing_keys=True,
            ),
            monai_transforms.EnsureTyped(keys=keys, allow_missing_keys=True),
        ]
    )


def build_train_eval_loaders(
    records: Sequence[Dict], data_cfg: DataConfig, train_cfg: TrainConfig
) -> Dict[str, DataLoader]:
    train_records = filter_by_split(records, split="train", split_key=data_cfg.split_key)
    valid_records = filter_by_split(records, split="valid", split_key=data_cfg.split_key)
    test_records = filter_by_split(records, split="test", split_key=data_cfg.split_key)

    train_loader, _ = build_dataloader(
        train_records,
        data_cfg=data_cfg,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        shuffle=True,
    )
    valid_loader, _ = build_dataloader(
        valid_records,
        data_cfg=data_cfg,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        shuffle=False,
    )
    test_loader, _ = build_dataloader(
        test_records,
        data_cfg=data_cfg,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        shuffle=False,
    )
    return {"train": train_loader, "valid": valid_loader, "test": test_loader}
