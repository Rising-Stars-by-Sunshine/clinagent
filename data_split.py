import argparse
import json
import os
from typing import Tuple

import pandas as pd


DEFAULT_SPLIT_COUNTS = {
    "development": 128,
    "validation": 19,
    "test": 37,
}


def _resolve_columns(df: pd.DataFrame) -> Tuple[str, str]:
    id_candidates = ["Participant_ID", "participant_id", "session_id", "Session_ID"]
    score_candidates = ["PHQ8_Score", "phq8_score", "PHQ_Score", "score"]

    id_column = next((column for column in id_candidates if column in df.columns), None)
    score_column = next((column for column in score_candidates if column in df.columns), None)

    if id_column is None:
        raise ValueError(f"No participant or session identifier found. Available columns: {list(df.columns)}")
    if score_column is None:
        raise ValueError(f"No PHQ-8 score column found. Available columns: {list(df.columns)}")

    return id_column, score_column


def load_valid_sessions(index_csv: str, data_root: str = "", require_files: bool = False) -> pd.DataFrame:
    df = pd.read_csv(index_csv)
    id_column, score_column = _resolve_columns(df)

    valid_df = df.dropna(subset=[id_column, score_column]).copy()
    valid_df = valid_df.drop_duplicates(subset=[id_column], keep="first")
    valid_df[id_column] = valid_df[id_column].astype(int)

    if require_files:
        if not data_root:
            raise ValueError("--data-root is required when --require-files is enabled.")

        valid_df = valid_df[
            valid_df[id_column].apply(
                lambda participant_id: os.path.exists(
                    os.path.join(data_root, f"{int(participant_id)}_TRANSCRIPT.csv")
                )
                and os.path.exists(os.path.join(data_root, f"{int(participant_id)}_CLNF_AUs.csv"))
            )
        ].copy()

    valid_df = valid_df.sort_values(id_column).reset_index(drop=True)
    return valid_df


def split_sessions(
    df: pd.DataFrame,
    seed: int = 42,
    split_counts: dict = DEFAULT_SPLIT_COUNTS,
) -> dict:
    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    total_required = sum(split_counts.values())

    if len(shuffled) != total_required:
        development_count = round(len(shuffled) * 0.70)
        validation_count = round(len(shuffled) * 0.10)
        test_count = len(shuffled) - development_count - validation_count
        split_counts = {
            "development": development_count,
            "validation": validation_count,
            "test": test_count,
        }

    development_end = split_counts["development"]
    validation_end = development_end + split_counts["validation"]

    return {
        "development": shuffled.iloc[:development_end].reset_index(drop=True),
        "validation": shuffled.iloc[development_end:validation_end].reset_index(drop=True),
        "test": shuffled.iloc[validation_end:].reset_index(drop=True),
    }


def validate_session_level_split(splits: dict) -> None:
    id_column = "Participant_ID"
    for split in splits.values():
        if id_column not in split.columns:
            id_column = split.columns[0]
            break

    id_sets = {
        name: set(split[id_column].astype(int).tolist())
        for name, split in splits.items()
    }

    names = list(id_sets.keys())
    for index, left_name in enumerate(names):
        for right_name in names[index + 1:]:
            overlap = id_sets[left_name].intersection(id_sets[right_name])
            if overlap:
                raise ValueError(
                    f"Data leakage detected between {left_name} and {right_name}: {sorted(overlap)}"
                )


def write_splits(splits: dict, output_dir: str, seed: int, source_csv: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    validate_session_level_split(splits)

    manifest = {
        "source_csv": source_csv,
        "seed": seed,
        "counts": {name: len(split) for name, split in splits.items()},
        "files": {},
    }

    for name, split in splits.items():
        output_path = os.path.join(output_dir, f"{name}.csv")
        split.to_csv(output_path, index=False)
        manifest["files"][name] = output_path

    manifest_path = os.path.join(output_dir, "split_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create session-level data splits for PHQ-8 experiments.")
    parser.add_argument("--index-csv", required=True, help="Path to the validated session index CSV.")
    parser.add_argument("--data-root", default="", help="Directory containing transcript and AU files.")
    parser.add_argument("--output-dir", default="data_splits", help="Directory for split CSV files.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splitting.")
    parser.add_argument(
        "--require-files",
        action="store_true",
        help="Keep only sessions with both transcript and AU files present.",
    )
    args = parser.parse_args()

    valid_sessions = load_valid_sessions(args.index_csv, args.data_root, args.require_files)
    splits = split_sessions(valid_sessions, seed=args.seed)
    write_splits(splits, args.output_dir, args.seed, args.index_csv)

    counts = {name: len(split) for name, split in splits.items()}
    print(f"Created session-level splits: {counts}")


if __name__ == "__main__":
    main()
