import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config.schema import AppConfig
from src.data.loaders import build_train_eval_loaders, prepare_manifest_records
from src.model import DLBMD
from src.utils.bone_map import make_inverse_region, make_target_region_from_hu
from src.utils.metrics import aggregate_epoch_metrics, classification_accuracy, regression_mae
from src.utils.optim import build_optimizer, get_scheduler
from src.utils.seed import seed_everything


def _choose_device(use_gpu: bool) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _forward_model(model: DLBMD, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reg_out = None
    spatial_map = None
    output = model(image)
    if model.inverse_attention:
        if model.regression:
            logits, reg_out, spatial_map = output
        else:
            logits, spatial_map = output
    else:
        if model.regression:
            logits, reg_out = output
        else:
            logits = output
    return logits, reg_out, spatial_map


def _ensure_tensor(name: str, value: object) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"'{name}' must be a torch.Tensor, got {type(value).__name__}.")
    return value


def _to_5d_region(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 3:
        return mask.unsqueeze(0).unsqueeze(0)
    if mask.ndim == 4:
        return mask.unsqueeze(1) if mask.shape[0] > 1 else mask.unsqueeze(0)
    if mask.ndim == 5:
        return mask
    raise ValueError(f"Unsupported region ndim: {mask.ndim}")


def _build_target_region(
    batch: Dict[str, object],
    image_hu: torch.Tensor,
    spatial_map: torch.Tensor,
    hu_threshold: Optional[float],
    use_manifest_bone_mask: bool,
) -> torch.Tensor:
    batch_bone = batch.get("bone")
    if use_manifest_bone_mask:
        if batch_bone is None:
            raise ValueError(
                "data.use_manifest_bone_mask=true requires manifest 'bone_path' for every sample."
            )
        bone = _ensure_tensor("bone", batch_bone).to(spatial_map.device).float()
        bone = _to_5d_region(bone)
        if bone.shape[-3:] != spatial_map.shape[-3:]:
            bone = F.interpolate(bone, size=spatial_map.shape[-3:], mode="nearest")
        return (bone > 0.5).float().contiguous()

    if hu_threshold is None:
        raise ValueError("train.hu_threshold is required when data.use_manifest_bone_mask=false.")

    # HU thresholding expects raw HU-scale volumes, not normalized [0, 1] tensors.
    hu_min = float(image_hu.min().item())
    hu_max = float(image_hu.max().item())
    if hu_min >= -1e-3 and hu_max <= 1.0 + 1e-3:
        raise ValueError(
            "HU fallback received a normalized tensor in [0, 1]. "
            "Pass raw HU tensor as 'image_hu', not normalized 'image'."
        )

    return make_target_region_from_hu(
        image_hu,
        hu_threshold=float(hu_threshold),
        out_size=int(spatial_map.shape[-1]),
    )


def _run_epoch(
    loader: DataLoader,
    model: DLBMD,
    device: torch.device,
    hu_threshold: Optional[float],
    use_manifest_bone_mask: bool,
    optimizer: torch.optim.Optimizer,
    cls_criterion: nn.Module,
    reg_criterion: nn.Module,
    inv_loss_weight: float,
    reg_loss_weight: float,
    train: bool,
) -> Dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()

    batch_logs: List[Dict[str, float]] = []
    for batch in loader:
        image = _ensure_tensor("image", batch["image"]).to(device).float()
        image_hu = _ensure_tensor("image_hu", batch["image_hu"]).to(device).float()
        label = batch["label"].to(device).long()
        t_score = batch.get("t_score")
        if t_score is not None:
            t_score = t_score.to(device).float().view(-1, 1)

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            logits, reg_out, spatial_map = _forward_model(model, image)
            cls_loss = cls_criterion(logits, label)
            total_loss = cls_loss
            reg_loss = torch.tensor(0.0, device=device)
            inv_loss = torch.tensor(0.0, device=device)

            if model.regression and reg_out is not None and t_score is not None:
                reg_loss = reg_criterion(reg_out, t_score)
                total_loss = total_loss + (reg_loss_weight * reg_loss)

            if model.inverse_attention and spatial_map is not None:
                target_region = _build_target_region(
                    batch=batch,
                    image_hu=image_hu,
                    spatial_map=spatial_map,
                    hu_threshold=hu_threshold,
                    use_manifest_bone_mask=use_manifest_bone_mask,
                )
                inv_bone = make_inverse_region(target_region)
                inv_loss = (spatial_map * inv_bone).pow(2).mean()
                total_loss = total_loss + (inv_loss_weight * inv_loss)

            if train:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()

        log: Dict[str, float] = {
            "loss": float(total_loss.item()),
            "cls_loss": float(cls_loss.item()),
            "acc": classification_accuracy(logits.detach(), label.detach()),
            "reg_loss": float(reg_loss.item()),
            "inv_loss": float(inv_loss.item()),
        }
        if model.regression and reg_out is not None and t_score is not None:
            log["mae"] = regression_mae(reg_out.detach(), t_score.detach())
        batch_logs.append(log)
    return aggregate_epoch_metrics(batch_logs)


def _save_checkpoint(model: DLBMD, optimizer: torch.optim.Optimizer, epoch: int, out_dir: Path) -> Path:
    ckpt_path = out_dir / f"epoch_{epoch:03d}.pt"
    torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, ckpt_path)
    return ckpt_path


def run_training(cfg: AppConfig) -> None:
    if cfg.train is None:
        raise ValueError("train config is required")

    seed_everything(cfg.train.seed)
    device = _choose_device(cfg.train.use_gpu)
    out_dir = Path(cfg.train.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = prepare_manifest_records(cfg.data)
    loaders = build_train_eval_loaders(records=records, data_cfg=cfg.data, train_cfg=cfg.train)

    model = DLBMD(
        growth_rate=cfg.model.growth_rate,
        block_config=cfg.model.block_config,
        inverse_attention=cfg.model.inverse_attention,
        attentive_regularization=cfg.model.attentive_regularization,
        split_denominator=cfg.model.split_denominator,
        num_classes=cfg.model.num_classes,
        regression=cfg.model.regression,
    ).to(device)

    optimizer = build_optimizer(model, lr=cfg.train.lr, wd=cfg.train.weight_decay)
    scheduler = get_scheduler(
        optimizer=optimizer,
        warmup=cfg.train.warmup_epochs,
        total=cfg.train.epochs,
        unit="epoch",
    )

    cls_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.MSELoss()
    history: List[Dict] = []
    best_valid_loss = float("inf")
    best_ckpt = None

    for epoch in range(1, cfg.train.epochs + 1):
        train_metrics = _run_epoch(
            loaders["train"],
            model=model,
            device=device,
            hu_threshold=cfg.train.hu_threshold,
            use_manifest_bone_mask=cfg.data.use_manifest_bone_mask,
            optimizer=optimizer,
            cls_criterion=cls_criterion,
            reg_criterion=reg_criterion,
            inv_loss_weight=cfg.train.inv_loss_weight,
            reg_loss_weight=cfg.train.reg_loss_weight,
            train=True,
        )
        valid_metrics = _run_epoch(
            loaders["valid"],
            model=model,
            device=device,
            hu_threshold=cfg.train.hu_threshold,
            use_manifest_bone_mask=cfg.data.use_manifest_bone_mask,
            optimizer=optimizer,
            cls_criterion=cls_criterion,
            reg_criterion=reg_criterion,
            inv_loss_weight=cfg.train.inv_loss_weight,
            reg_loss_weight=cfg.train.reg_loss_weight,
            train=False,
        )
        scheduler.step()

        row = {"epoch": epoch, "train": train_metrics, "valid": valid_metrics}
        history.append(row)
        print(
            f"[epoch {epoch}] train_loss={train_metrics.get('loss', 0):.4f} "
            f"valid_loss={valid_metrics.get('loss', 0):.4f} "
            f"valid_acc={valid_metrics.get('acc', 0):.4f}"
        )

        if valid_metrics.get("loss", float("inf")) < best_valid_loss:
            best_valid_loss = valid_metrics["loss"]
            best_ckpt = _save_checkpoint(model, optimizer, epoch, out_dir)

        if cfg.train.save_every > 0 and (epoch % cfg.train.save_every == 0):
            _save_checkpoint(model, optimizer, epoch, out_dir)

    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    if best_ckpt is not None:
        print(f"best checkpoint: {best_ckpt}")
