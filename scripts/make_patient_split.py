import argparse

from src.data.manifest_io import load_manifest, save_manifest
from src.data.patient_split import assert_no_patient_leakage, split_by_patient_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create patient_id group splits in manifest")
    parser.add_argument("--input_manifest", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--patient_id_key", default="patient_id")
    parser.add_argument("--split_key", default="split")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--valid_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_manifest(args.input_manifest)
    records = split_by_patient_id(
        records=records,
        patient_id_key=args.patient_id_key,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        split_key=args.split_key,
    )
    ok, leaks = assert_no_patient_leakage(records=records, patient_id_key=args.patient_id_key, split_key=args.split_key)
    if not ok:
        raise RuntimeError(f"split leakage detected: {leaks}")
    save_manifest(records, args.output_manifest)
    print(f"saved split manifest: {args.output_manifest}")


if __name__ == "__main__":
    main()
