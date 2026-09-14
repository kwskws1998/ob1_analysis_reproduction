"""Audit the official Provo CSVs and emit a machine-readable inventory."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


REQUIRED_EYE_COLUMNS = {
    "Participant_ID",
    "Word_Unique_ID",
    "Text_ID",
    "Word_Number",
    "Word",
    "IA_DWELL_TIME",
    "IA_FIXATION_COUNT",
    "IA_SKIP",
}
REQUIRED_PREDICTABILITY_COLUMNS = {
    "Word_Unique_ID",
    "Text_ID",
    "Text",
    "Word_Number",
    "Word",
    "Response_Proportion",
}


def integer_value(value: str) -> int:
    """Convert an integer-like CSV field such as 2 or 2.0 to int."""
    return int(float(value))


def corrected_ob1_eye_position(
    text_id: int,
    word_number: int,
    word: str,
) -> tuple[int, int]:
    """Apply the position corrections encoded in the published OB1 evaluator."""
    corrected_text_id = text_id - 1
    corrected_word_id = word_number - 1
    if corrected_text_id == 17 and corrected_word_id == 2 and word == "evolution":
        corrected_word_id = 50
    if corrected_text_id == 2 and 45 <= corrected_word_id <= 59:
        corrected_word_id -= 1
    if corrected_text_id == 12 and 19 <= corrected_word_id <= 54:
        corrected_word_id -= 1
    return corrected_text_id, corrected_word_id


def audit_eye_tracking(path: Path) -> dict:
    """Summarize the participant-level Provo eye-tracking table."""
    participants = set()
    texts = set()
    positions = set()
    corrected_positions = set()
    position_records = defaultdict(set)
    trt_zero = 0
    trt_positive = 0
    rows = 0
    missing_word_number_rows = 0

    with path.open(encoding="cp1252", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_EYE_COLUMNS.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            rows += 1
            participants.add(row["Participant_ID"])
            if not row["Word_Number"] or row["Word_Number"].upper() == "NA":
                missing_word_number_rows += 1
                continue
            text_id = integer_value(row["Text_ID"])
            word_number = integer_value(row["Word_Number"])
            texts.add(text_id)
            positions.add((text_id, word_number))
            corrected_positions.add(
                corrected_ob1_eye_position(text_id, word_number, row["Word"])
            )
            position_records[(text_id, word_number)].add(
                (row["Word_Unique_ID"], row["Word"])
            )
            trt = float(row["IA_DWELL_TIME"])
            trt_zero += trt == 0
            trt_positive += trt > 0

    duplicate_positions = {
        f"{text_id}:{word_number}": sorted(records)
        for (text_id, word_number), records in position_records.items()
        if len(records) > 1
    }
    return {
        "rows": rows,
        "participants": len(participants),
        "passages": len(texts),
        "unique_text_word_positions": len(positions),
        "trt_zero_rows": trt_zero,
        "trt_positive_rows": trt_positive,
        "missing_word_number_rows": missing_word_number_rows,
        "duplicate_positions": duplicate_positions,
        "_positions": positions,
        "_ob1_positions": corrected_positions,
    }


def audit_predictability(path: Path) -> dict:
    """Summarize the Provo predictability table and its passage texts."""
    texts = set()
    positions = set()
    position_records = defaultdict(set)
    passage_texts = {}
    rows = 0

    with path.open(encoding="cp1252", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_PREDICTABILITY_COLUMNS.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            rows += 1
            text_id = integer_value(row["Text_ID"])
            word_number = integer_value(row["Word_Number"])
            texts.add(text_id)
            positions.add((text_id, word_number))
            passage_texts.setdefault(text_id, row["Text"])
            position_records[(text_id, word_number)].add(
                (row["Word_Unique_ID"], row["Word"])
            )

    split_word_count = sum(len(text.split()) for text in passage_texts.values())
    corrected_passage_texts = {
        text_id: text.replace(" Ñ", "") if text_id == 36 else text
        for text_id, text in passage_texts.items()
    }
    ob1_positions = {
        (text_id - 1, word_id)
        for text_id, text in corrected_passage_texts.items()
        for word_id in range(1, len(text.split()))
        if (text_id - 1, word_id) not in {(54, 9), (54, 60), (54, 61)}
    }
    duplicate_positions = {
        f"{text_id}:{word_number}": sorted(records)
        for (text_id, word_number), records in position_records.items()
        if len(records) > 1
    }
    return {
        "rows": rows,
        "passages": len(texts),
        "unique_text_word_positions": len(positions),
        "passage_text_split_words": split_word_count,
        "duplicate_positions": duplicate_positions,
        "_positions": positions,
        "_ob1_positions": ob1_positions,
    }


def audit_dataset(raw_dir: Path) -> dict:
    """Combine both CSV audits and expose cross-file alignment differences."""
    eye = audit_eye_tracking(raw_dir / "Provo_Corpus-Eyetracking_Data.csv")
    predictability = audit_predictability(
        raw_dir / "Provo_Corpus-Predictability_Norms.csv"
    )
    eye_positions = eye.pop("_positions")
    predictability_positions = predictability.pop("_positions")
    corrected_eye_positions = eye.pop("_ob1_positions")
    ob1_positions = predictability.pop("_ob1_positions")
    return {
        "eye_tracking": eye,
        "predictability": predictability,
        "alignment": {
            "predictability_only_positions": sorted(
                [
                    list(position)
                    for position in predictability_positions - eye_positions
                ]
            ),
            "eye_tracking_only_positions": sorted(
                [
                    list(position)
                    for position in eye_positions - predictability_positions
                ]
            ),
            "shared_positions": len(eye_positions & predictability_positions),
        },
        "published_ob1_alignment": {
            "evaluable_positions": len(ob1_positions),
            "corrected_human_positions": len(corrected_eye_positions),
            "human_only_positions": sorted(
                [list(position) for position in corrected_eye_positions - ob1_positions]
            ),
            "model_only_positions": sorted(
                [list(position) for position in ob1_positions - corrected_eye_positions]
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    """Parse the raw-data directory."""
    default_raw = Path(__file__).resolve().parents[1] / "data" / "raw"
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=default_raw)
    return parser.parse_args()


def main() -> None:
    """Print the audit report as deterministic JSON."""
    args = parse_args()
    print(json.dumps(audit_dataset(args.raw_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
