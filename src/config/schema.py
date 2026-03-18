from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class DataConfig:
    manifest_path: str
    image_base_dir: str = ""
    bone_base_dir: str = ""
    nonbone_base_dir: str = "" # optional mask for non-bone region
    use_manifest_bone_mask: bool = True
    patient_id_key: str = "patient_id"
    label_key: str = "label"
    t_score_key: str = "t_score"
    # set the ratio fitting your dataset for train/valid/test split
    split_key: str = "split" # key for split in manifest
    generate_split: bool = True # generate split from patient_id if not provided in manifest
    train_ratio: float = 0.7
    valid_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    # windowing parameters
    ww: Optional[float] = None
    wc: Optional[float] = None
    use_monai_transforms: bool = True
    # monai transform parameters
    rand_flip_prob: float = 0.0
    rand_rotate90_prob: float = 0.0
    rand_affine_prob: float = 0.2


@dataclass
class ModelConfig:
    growth_rate: int = 32
    block_config: Tuple[int, int, int, int] = (6, 12, 24, 16)
    inverse_attention: bool = True
    attentive_regularization: bool = False
    split_denominator: int = 2
    num_classes: int = 3
    regression: bool = False


@dataclass
class TrainConfig:
    output_dir: str = "outputs/train"
    epochs: int = 100
    batch_size: int = 4
    num_workers: int = 0
    lr: float = 1e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 1
    save_every: int = 1
    seed: int = 42
    hu_threshold: Optional[float] = None
    inv_loss_weight: float = 1.0
    reg_loss_weight: float = 1.0
    use_gpu: bool = True


@dataclass
class InferConfig:
    output_dir: str = "outputs/infer"
    checkpoint_path: str = ""
    batch_size: int = 2
    num_workers: int = 0
    split: str = "test"
    use_gpu: bool = True


@dataclass
class AppConfig:
    data: DataConfig
    model: ModelConfig
    train: Optional[TrainConfig] = None
    infer: Optional[InferConfig] = None

    def ensure_paths(self) -> None:
        Path(self.data.manifest_path)
        if self.train is not None:
            Path(self.train.output_dir).mkdir(parents=True, exist_ok=True)
        if self.infer is not None:
            Path(self.infer.output_dir).mkdir(parents=True, exist_ok=True)
