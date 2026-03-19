# **Repository for** "Deep Learning–Based Opportunistic Screening for Osteoporosis on Low-Dose Chest CT: Development and Multicenter Validation."

This repository is codes for paper: "Deep Learning–Based Opportunistic Screening for Osteoporosis on Low-Dose Chest CT: Development and Multicenter Validation."

## Directory Overview

- `main.py`: CLI entry point (`train`, `infer`)
- `src/pipelines/train.py`: Training orchestration and checkpoint saving
- `src/pipelines/infer.py`: Checkpoint loading and prediction CSV export
- `src/data/manifest_io.py`: JSON manifest IO and key validation
- `src/data/split.py`: `patient_id` group split generation and split filtering
- `src/data/dataset.py`: `ManifestDataset` and tensor/window normalization
- `src/data/loaders.py`: DataLoader assembly and MONAI transform wiring
- `src/model/densenet3d.py`: `DLBMD` + inverse attention + regression head
- `src/data/patient_split.py`: `patient_id`-group split and leakage checks
- `src/utils/bone_map.py`: HU-threshold-based `target_region(32)` and inverse map generation
- `src/config/schema.py`, `src/config/loader.py`: Config schema and validation
- `configs/*.yaml`: Example run configurations

## Data Input Assumptions (JSON Manifest)

The manifest is a JSON array, and each row represents one sample.

Required keys:

- `image_path`: `.pt` path (3D tensor)
- `label`: Integer class index
- `patient_id`: Patient identifier

Optional keys:

- `split`: `train|valid|test`
- `t_score`: Regression target
- `bone_path`, `nonbone_path`: Additional input paths
- `meta`: Additional metadata

If `split` is not provided, set `data.generate_split=true` to generate a grouped split by `patient_id`.
Inverse-attention target policy is controlled strictly by `data.use_manifest_bone_mask`:

- `true`: `bone_path` must be provided for every sample used in training.
- `false`: manifest bone masks are ignored, and CT HU-threshold fallback is always used.
`build_dataloader` applies MONAI dict transforms directly (train-time augmentation), controlled by `data.use_monai_transforms`.

### Runtime Input Tensor Types

- `image_hu`:
  - Source: `image_path`
  - Type: `torch.Tensor` (`float32` in runtime)
  - Shape: `[C,D,H,W]` after loading (`C=1` for single-channel volumes)
  - Semantics: raw HU-domain CT volume (not min-max normalized)
- `image`:
  - Source: derived from `image_hu`
  - Type: `torch.Tensor` (`float32`)
  - Shape: same as `image_hu`
  - Semantics: model input after optional HU windowing (`ww/wc`) and min-max normalization to `[0,1]`
- `bone` / `nonbone` (optional):
  - Source: `bone_path` / `nonbone_path`
  - Type: `torch.Tensor` (`float32`)
  - Shape: `[C,D,H,W]` (or `[D,H,W]` before channel expansion)
  - Semantics: binary or soft region masks

Important: HU-threshold-based bone map generation must use `image_hu`, not `image`.

## Core Policies

- Patient leakage prevention: split must be performed at the `patient_id` level
- 32 bone_map:
  - `target_region = make_target_region_from_hu(image_hu, hu_threshold)` for CT fallback path
  - HU fallback must use raw HU tensor (`image_hu`), not normalized model input (`image`)
  - `target_region = manifest bone_mask` only when `data.use_manifest_bone_mask=true`
  - `inv_bone = 1 - target_region`

## Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

For a CUDA-enabled GPU, install `torch` that matches your driver using the [PyTorch install guide](https://pytorch.org/get-started/locally/) first, then run `pip install -r requirements.txt` so the rest of the dependencies are installed without forcing a CPU-only `torch` wheel.

## How To Run

Training:

```bash
python main.py train --config configs/train.example.yaml
```

Inference:

```bash
python main.py infer --config configs/infer.example.yaml
```

## CSV -> Manifest Conversion (Optional)

```bash
python scripts/convert_csv_to_manifest.py --input_csv data.csv --output_json data/manifest.json
```

## Patient Split (Optional)

```bash
python scripts/make_patient_split.py --input_manifest data/manifest.json --output_manifest data/manifest_split.json
```

