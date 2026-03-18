import csv
from pathlib import Path
from typing import Dict, List

import torch

from src.config.schema import AppConfig, TrainConfig
from src.data.loaders import build_dataloader, filter_by_split, prepare_manifest_records
from src.model import DLBMD


def _choose_device(use_gpu: bool) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_checkpoint(model: DLBMD, checkpoint_path: str, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)


def run_inference(cfg: AppConfig) -> None:
    if cfg.infer is None:
        raise ValueError("infer config is required")

    device = _choose_device(cfg.infer.use_gpu)
    out_dir = Path(cfg.infer.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = prepare_manifest_records(cfg.data)
    split_records = filter_by_split(records, split=cfg.infer.split, split_key=cfg.data.split_key)
    infer_loader, _ = build_dataloader(
        split_records,
        data_cfg=cfg.data,
        batch_size=cfg.infer.batch_size,
        num_workers=cfg.infer.num_workers,
        shuffle=False,
    )

    model = DLBMD(
        growth_rate=cfg.model.growth_rate,
        block_config=cfg.model.block_config,
        inverse_attention=cfg.model.inverse_attention,
        attentive_regularization=cfg.model.attentive_regularization,
        split_denominator=cfg.model.split_denominator,
        num_classes=cfg.model.num_classes,
        regression=cfg.model.regression,
    ).to(device)
    _load_checkpoint(model, cfg.infer.checkpoint_path, device=device)
    model.eval()

    rows: List[Dict] = []
    with torch.no_grad():
        for batch in infer_loader:
            image = batch["image"].to(device).float()
            label = batch["label"].to(device).long()
            logits = model(image)
            reg_out = None
            if model.inverse_attention:
                if model.regression:
                    logits, reg_out, _ = logits
                else:
                    logits, _ = logits
            elif model.regression:
                logits, reg_out = logits

            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)
            bsz = pred.shape[0]
            for i in range(bsz):
                row = {
                    "true_label": int(label[i].item()),
                    "pred_label": int(pred[i].item()),
                }
                for c in range(probs.shape[1]):
                    row[f"prob_{c}"] = float(probs[i, c].item())
                if reg_out is not None:
                    row["pred_t_score"] = float(reg_out[i].view(-1)[0].item())
                rows.append(row)

    output_csv = out_dir / f"predictions_{cfg.infer.split}.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        else:
            f.write("")
    print(f"saved inference result: {output_csv}")


def run_inference_with_train_config(
    app_cfg: AppConfig, train_cfg_for_loader: TrainConfig, checkpoint_path: str
) -> None:
    app_cfg.infer.checkpoint_path = checkpoint_path
    app_cfg.infer.batch_size = train_cfg_for_loader.batch_size
    run_inference(app_cfg)
