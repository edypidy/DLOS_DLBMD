import random
from collections import defaultdict
from typing import Dict, List, Sequence, Set, Tuple


def split_by_patient_id(
    records: Sequence[Dict],
    patient_id_key: str = "patient_id",
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    split_key: str = "split",
) -> List[Dict]:
    if abs((train_ratio + valid_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train/valid/test ratio must sum to 1.0")

    patient_to_rows: Dict[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(records):
        pid = row.get(patient_id_key)
        if pid is None:
            raise ValueError(f"row {idx} missing patient id key: {patient_id_key}")
        patient_to_rows[str(pid)].append(idx)

    patients = list(patient_to_rows.keys())
    rng = random.Random(seed)
    rng.shuffle(patients)

    n_patients = len(patients)
    n_train = max(1, int(round(n_patients * train_ratio)))
    n_valid = max(1, int(round(n_patients * valid_ratio)))
    if n_train + n_valid >= n_patients:
        n_valid = max(1, n_patients - n_train - 1)
    n_test = n_patients - n_train - n_valid
    if n_test < 1:
        n_test = 1
        if n_train > n_valid:
            n_train -= 1
        else:
            n_valid -= 1

    train_patients = set(patients[:n_train])
    valid_patients = set(patients[n_train : n_train + n_valid])
    test_patients = set(patients[n_train + n_valid :])

    new_records = [dict(row) for row in records]
    for idx, row in enumerate(new_records):
        pid = str(row[patient_id_key])
        if pid in train_patients:
            row[split_key] = "train"
        elif pid in valid_patients:
            row[split_key] = "valid"
        elif pid in test_patients:
            row[split_key] = "test"
        else:
            raise RuntimeError(f"patient split assignment failed: {pid} (row={idx})")
    return new_records


def collect_split_patient_sets(
    records: Sequence[Dict], patient_id_key: str = "patient_id", split_key: str = "split"
) -> Dict[str, Set[str]]:
    split_sets: Dict[str, Set[str]] = defaultdict(set)
    for row in records:
        split = str(row.get(split_key, ""))
        if not split:
            continue
        split_sets[split].add(str(row[patient_id_key]))
    return split_sets


def assert_no_patient_leakage(
    records: Sequence[Dict], patient_id_key: str = "patient_id", split_key: str = "split"
) -> Tuple[bool, Dict[str, int]]:
    split_sets = collect_split_patient_sets(records=records, patient_id_key=patient_id_key, split_key=split_key)
    train_set = split_sets.get("train", set())
    valid_set = split_sets.get("valid", set())
    test_set = split_sets.get("test", set())

    leaks = {
        "train_valid": len(train_set.intersection(valid_set)),
        "train_test": len(train_set.intersection(test_set)),
        "valid_test": len(valid_set.intersection(test_set)),
    }
    ok = all(v == 0 for v in leaks.values())
    return ok, leaks
