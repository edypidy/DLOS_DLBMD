from typing import Dict, List, Sequence

from src.config.schema import DataConfig

from .manifest_io import ensure_manifest_required_keys, load_manifest
from .patient_split import assert_no_patient_leakage, split_by_patient_id


def prepare_manifest_records(data_cfg: DataConfig) -> List[Dict]:
    records = load_manifest(data_cfg.manifest_path)
    ensure_manifest_required_keys(records)

    has_split = all(data_cfg.split_key in row for row in records)
    if data_cfg.generate_split and not has_split:
        records = split_by_patient_id(
            records,
            patient_id_key=data_cfg.patient_id_key,
            train_ratio=data_cfg.train_ratio,
            valid_ratio=data_cfg.valid_ratio,
            test_ratio=data_cfg.test_ratio,
            seed=data_cfg.seed,
            split_key=data_cfg.split_key,
        )
    ok, leaks = assert_no_patient_leakage(records, patient_id_key=data_cfg.patient_id_key, split_key=data_cfg.split_key)
    if not ok:
        raise ValueError(f"patient leakage detected: {leaks}")
    return records


def filter_by_split(records: Sequence[Dict], split: str, split_key: str = "split") -> List[Dict]:
    return [r for r in records if r.get(split_key) == split]
