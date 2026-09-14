"""Build the canonical Provo passage and human-TRT evaluation tables."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .audit_provo import corrected_ob1_eye_position


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data/raw"
PROCESSED_DIR = ROOT / "data/processed"
EYE_FILENAME = "Provo_Corpus-Eyetracking_Data.csv"
PREDICTABILITY_FILENAME = "Provo_Corpus-Predictability_Norms.csv"
KNOWN_MISSING_HUMAN_POSITIONS = {(54, 9), (54, 60), (54, 61)}
KNOWN_ALIGNMENT_EXCEPTIONS = {
    (24, 44): ("90", "09"),
    (26, 43): ("doesnõt", "doesnt"),
    (44, 27): ("womenõs", "womens"),
    (53, 26): ("bondsõ", "bonds"),
}


def normalize_ob1_word(value: str) -> str:
    """Apply the word cleanup used by the pinned OB1 implementation."""
    return re.sub(r"[^\w\s]", "", str(value)).lower().strip()


def load_passage_texts(predictability_path: Path) -> dict[int, str]:
    """Load one unique official passage string for every Provo text ID."""
    frame = pd.read_csv(
        predictability_path,
        encoding="cp1252",
        low_memory=False,
    )
    text_counts = frame.groupby("Text_ID")["Text"].nunique(dropna=False)
    if not bool((text_counts == 1).all()):
        invalid = {}
        for text_id in text_counts[text_counts != 1].index:
            variants = frame.loc[frame["Text_ID"] == text_id, "Text"].unique()
            normalized_variants = {str(item).replace("Õ", "'") for item in variants}
            if len(normalized_variants) != 1:
                invalid[int(text_id)] = variants.tolist()
        if invalid:
            raise ValueError(f"Passages with incompatible Text values: {invalid}")
    return {
        int(text_id): text
        for text_id, text in frame.groupby("Text_ID", sort=True)["Text"].first().items()
    }


def prepare_eye_rows(eye_path: Path) -> pd.DataFrame:
    """Load participant rows and add published OB1-aligned coordinates."""
    frame = pd.read_csv(eye_path, encoding="cp1252", low_memory=False)
    frame = frame.dropna(subset=["Word_Number"]).copy()
    corrected = [
        corrected_ob1_eye_position(
            int(text_id),
            int(word_number),
            str(word),
        )
        for text_id, word_number, word in zip(
            frame["Text_ID"],
            frame["Word_Number"],
            frame["Word"],
        )
    ]
    frame["passage_id_zero_based"] = [item[0] for item in corrected]
    frame["word_id_zero_based"] = [item[1] for item in corrected]
    return frame


def exclusion_record(
    passage_id_raw: int,
    word_id_zero_based: int,
    word_raw: str,
    reason: str,
) -> dict:
    """Create one passage-level exclusion audit record."""
    return {
        "passage_id_raw": passage_id_raw,
        "passage_id_zero_based": passage_id_raw - 1,
        "word_id_zero_based": word_id_zero_based,
        "word_number_one_based": word_id_zero_based + 1,
        "word_raw": word_raw,
        "word_clean": normalize_ob1_word(word_raw),
        "reason": reason,
    }


def build_canonical_tables(
    eye_path: Path,
    predictability_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Construct passage, evaluable-word, exclusion, and audit objects."""
    passage_texts = load_passage_texts(predictability_path)
    eye = prepare_eye_rows(eye_path)
    eye_groups = eye.groupby(
        ["passage_id_zero_based", "word_id_zero_based"],
        sort=True,
    )

    passage_records = []
    word_records = []
    exclusions = []
    for passage_id_raw, original_text in passage_texts.items():
        passage_id_zero_based = passage_id_raw - 1
        if passage_id_raw == 36:
            original_matches = list(re.finditer(r"\S+", original_text))
            for raw_word_id, match in enumerate(original_matches):
                if match.group() == "Ñ":
                    exclusions.append(
                        exclusion_record(
                            passage_id_raw,
                            raw_word_id,
                            match.group(),
                            "malformed_stimulus_token_removed_by_published_ob1",
                        )
                    )
            passage_text = original_text.replace(" Ñ", "")
        else:
            passage_text = original_text

        token_matches = list(re.finditer(r"\S+", passage_text))
        evaluable_count = 0
        for word_id_zero_based, match in enumerate(token_matches):
            word_raw = match.group()
            position = (passage_id_zero_based, word_id_zero_based)
            if word_id_zero_based == 0:
                exclusions.append(
                    exclusion_record(
                        passage_id_raw,
                        word_id_zero_based,
                        word_raw,
                        "first_word_not_evaluated_by_published_ob1",
                    )
                )
                continue
            if position in KNOWN_MISSING_HUMAN_POSITIONS:
                exclusions.append(
                    exclusion_record(
                        passage_id_raw,
                        word_id_zero_based,
                        word_raw,
                        "position_missing_from_human_eye_tracking",
                    )
                )
                continue

            try:
                participant_rows = eye_groups.get_group(position)
            except KeyError as error:
                raise ValueError(
                    f"No human rows for canonical position {position} {word_raw!r}"
                ) from error
            if participant_rows["Participant_ID"].nunique() != len(participant_rows):
                raise ValueError(f"Duplicate participant rows at {position}")

            eye_words = participant_rows["Word"].dropna().astype(str).unique()
            if len(eye_words) != 1:
                raise ValueError(
                    f"Multiple human words at {position}: {eye_words.tolist()}"
                )
            eye_word_raw = eye_words[0]
            word_clean = normalize_ob1_word(word_raw)
            eye_word_clean = normalize_ob1_word(eye_word_raw)
            if word_clean == eye_word_clean:
                alignment_status = "exact_after_ob1_normalization"
            elif KNOWN_ALIGNMENT_EXCEPTIONS.get(position) == (
                word_clean,
                eye_word_clean,
            ):
                alignment_status = "published_known_exception"
            else:
                raise ValueError(
                    f"Unrecognized word mismatch at {position}: "
                    f"{word_raw!r} versus {eye_word_raw!r}"
                )

            trt = participant_rows["IA_DWELL_TIME"].astype(float)
            positive_trt = trt[trt > 0]
            eye_word_numbers = sorted(
                participant_rows["Word_Number"].astype(int).unique().tolist()
            )
            word_records.append(
                {
                    "passage_id_raw": passage_id_raw,
                    "passage_id_zero_based": passage_id_zero_based,
                    "word_number_one_based": word_id_zero_based + 1,
                    "word_id_zero_based": word_id_zero_based,
                    "word_raw": word_raw,
                    "word_clean": word_clean,
                    "character_start": match.start(),
                    "character_end": match.end(),
                    "eye_word_number_raw": "|".join(map(str, eye_word_numbers)),
                    "eye_word_raw": eye_word_raw,
                    "eye_word_clean": eye_word_clean,
                    "human_reader_count": participant_rows["Participant_ID"].nunique(),
                    "human_total_skip_count": int((trt == 0).sum()),
                    "human_first_pass_skip_count": int(
                        participant_rows["IA_SKIP"].astype(int).sum()
                    ),
                    "human_trt_unconditional": float(trt.mean()),
                    "human_trt_conditional": (
                        float(positive_trt.mean()) if len(positive_trt) else np.nan
                    ),
                    "alignment_status": alignment_status,
                }
            )
            evaluable_count += 1

        passage_records.append(
            {
                "passage_id_raw": passage_id_raw,
                "passage_id_zero_based": passage_id_zero_based,
                "passage_text": passage_text,
                "character_count": len(passage_text),
                "word_count_after_text_correction": len(token_matches),
                "evaluable_word_count": evaluable_count,
            }
        )

    passages = pd.DataFrame(passage_records)
    words = pd.DataFrame(word_records)
    excluded = pd.DataFrame(exclusions)
    expected_positions = set(
        zip(words["passage_id_zero_based"], words["word_id_zero_based"])
    )
    human_positions = set(zip(eye["passage_id_zero_based"], eye["word_id_zero_based"]))
    audit = {
        "passages": len(passages),
        "canonical_words": len(words),
        "excluded_positions": len(excluded),
        "human_positions": len(human_positions),
        "canonical_only_positions": sorted(
            [list(item) for item in expected_positions - human_positions]
        ),
        "human_only_positions": sorted(
            [list(item) for item in human_positions - expected_positions]
        ),
        "reader_count_min": int(words["human_reader_count"].min()),
        "reader_count_max": int(words["human_reader_count"].max()),
        "alignment_status_counts": {
            str(key): int(value)
            for key, value in words["alignment_status"].value_counts().items()
        },
        "exclusion_reason_counts": {
            str(key): int(value)
            for key, value in excluded["reason"].value_counts().items()
        },
    }
    validate_canonical_tables(passages, words, excluded, audit)
    return passages, words, excluded, audit


def validate_canonical_tables(
    passages: pd.DataFrame,
    words: pd.DataFrame,
    excluded: pd.DataFrame,
    audit: dict,
) -> None:
    """Enforce the frozen Provo evaluation contract."""
    if len(passages) != 55:
        raise ValueError(f"Expected 55 passages, found {len(passages)}")
    if len(words) != 2686:
        raise ValueError(f"Expected 2686 evaluable words, found {len(words)}")
    if words.duplicated(["passage_id_zero_based", "word_id_zero_based"]).any():
        raise ValueError("Canonical word coordinates are not unique")
    if not bool((words["human_reader_count"] == 84).all()):
        raise ValueError("Every canonical word must have all 84 participant rows")
    if audit["canonical_only_positions"] or audit["human_only_positions"]:
        raise ValueError(
            "Canonical and corrected human position grids do not match: "
            f"{audit['canonical_only_positions']} / "
            f"{audit['human_only_positions']}"
        )
    if len(excluded) != 59:
        raise ValueError(f"Expected 59 exclusions, found {len(excluded)}")


def write_canonical_tables(
    output_dir: Path,
    passages: pd.DataFrame,
    words: pd.DataFrame,
    excluded: pd.DataFrame,
    audit: dict,
) -> None:
    """Write deterministic canonical Provo artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    passages.to_csv(output_dir / "provo_passages.csv", index=False)
    words.to_csv(output_dir / "provo_words.csv", index=False)
    excluded.to_csv(output_dir / "provo_excluded_positions.csv", index=False)
    with (output_dir / "provo_prepare_audit.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    """Parse raw input paths and processed output directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--eye", type=Path, default=RAW_DIR / EYE_FILENAME)
    parser.add_argument(
        "--predictability",
        type=Path,
        default=RAW_DIR / PREDICTABILITY_FILENAME,
    )
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    return parser.parse_args()


def main() -> None:
    """Build, validate, save, and summarize the canonical Provo tables."""
    args = parse_args()
    artifacts = build_canonical_tables(args.eye, args.predictability)
    write_canonical_tables(args.output_dir, *artifacts)
    print(json.dumps(artifacts[-1], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
