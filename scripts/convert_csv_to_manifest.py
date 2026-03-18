import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CSV to JSON manifest")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--image_col", default="file_name")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--patient_id_col", default="patient_id")
    parser.add_argument("--t_score_col", default="t_score")
    parser.add_argument("--bone_col", default="bone_path")
    parser.add_argument("--nonbone_col", default="nonbone_path")
    parser.add_argument("--split_col", default="split")
    return parser.parse_args()


def _optional_value(row: pd.Series, col: str):
    if col in row and pd.notna(row[col]):
        return row[col]
    return None


def convert(df: pd.DataFrame, args: argparse.Namespace) -> List[Dict]:
    required = [args.image_col, args.label_col, args.patient_id_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    records: List[Dict] = []
    for _, row in df.iterrows():
        rec: Dict = {
            "image_path": str(row[args.image_col]),
            "label": int(row[args.label_col]),
            "patient_id": str(row[args.patient_id_col]),
        }
        t_score = _optional_value(row, args.t_score_col)
        if t_score is not None:
            rec["t_score"] = float(t_score)
        bone_path = _optional_value(row, args.bone_col)
        if bone_path is not None:
            rec["bone_path"] = str(bone_path)
        nonbone_path = _optional_value(row, args.nonbone_col)
        if nonbone_path is not None:
            rec["nonbone_path"] = str(nonbone_path)
        split = _optional_value(row, args.split_col)
        if split is not None:
            rec["split"] = str(split)
        records.append(rec)
    return records


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    records = convert(df, args)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"saved manifest: {output_path} ({len(records)} rows)")


if __name__ == "__main__":
    main()
