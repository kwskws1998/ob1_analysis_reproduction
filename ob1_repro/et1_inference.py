"""Run frozen ET1 and apply fixed learned or symmetric redistribution."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch


from .asym_gaussian_redistributor import AsymGaussianRedistributor
from .sigmas import (
    FIXED_SYMMETRIC_SIGMA,
    rms_scale_symmetric_sigma,
)


def passage_word_spans(text: str) -> list[dict]:
    """Return whitespace-token word IDs and exact character spans."""
    return [
        {
            "word_id_zero_based": word_id,
            "word_raw": match.group(),
            "character_start": match.start(),
            "character_end": match.end(),
        }
        for word_id, match in enumerate(re.finditer(r"\S+", text))
    ]


def assign_token_offsets(
    text: str,
    offsets: list[tuple[int, int]],
    special_tokens_mask: list[int],
) -> tuple[list[int | None], list[dict]]:
    """Assign each non-special ET1 token to exactly one passage word."""
    spans = passage_word_spans(text)
    assignments: list[int | None] = []
    for token_index, ((start, end), is_special) in enumerate(
        zip(offsets, special_tokens_mask)
    ):
        if is_special:
            assignments.append(None)
            continue
        if start >= end:
            raise ValueError(
                f"Non-special ET1 token {token_index} has empty offset {(start, end)}"
            )
        overlapping = [
            span
            for span in spans
            if min(end, span["character_end"]) > max(start, span["character_start"])
        ]
        if len(overlapping) != 1:
            raise ValueError(
                f"ET1 token {token_index} offset {(start, end)} overlaps "
                f"{len(overlapping)} passage words"
            )
        assignments.append(overlapping[0]["word_id_zero_based"])

    assigned_words = {item for item in assignments if item is not None}
    expected_words = {span["word_id_zero_based"] for span in spans}
    if assigned_words != expected_words:
        raise ValueError(
            "ET1 offset coverage does not match passage words: "
            f"missing {sorted(expected_words - assigned_words)}, "
            f"extra {sorted(assigned_words - expected_words)}"
        )
    return assignments, spans


def redistributor_from_log_sigmas(
    log_sigma_left: float,
    log_sigma_right: float,
    min_sigma: float,
) -> AsymGaussianRedistributor:
    """Construct the production redistributor with fixed checkpoint values."""
    module = AsymGaussianRedistributor(
        init_sigma_left=1.0,
        init_sigma_right=1.0,
        min_sigma=min_sigma,
    )
    with torch.no_grad():
        module.log_sigma_left.fill_(log_sigma_left)
        module.log_sigma_right.fill_(log_sigma_right)
    module.log_sigma_left.requires_grad_(False)
    module.log_sigma_right.requires_grad_(False)
    module.eval()
    return module


def redistribute_values(
    values: torch.Tensor,
    attention_mask: torch.Tensor,
    sigma_record: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply fixed, RMS-side-scale, and learned asymmetric redistribution."""
    fixed_symmetric, rms_symmetric, asymmetric = redistributors_from_sigma_record(
        sigma_record,
        values.device,
    )
    return apply_redistributors(
        values,
        attention_mask,
        fixed_symmetric,
        rms_symmetric,
        asymmetric,
    )


def symmetric_redistributor(
    effective_sigma: float,
    min_sigma: float,
    device: torch.device,
) -> AsymGaussianRedistributor:
    """Build one symmetric redistributor from an effective sigma value."""
    raw_sigma = float(effective_sigma) - float(min_sigma)
    if not math.isfinite(raw_sigma) or raw_sigma <= 0:
        raise ValueError("Symmetric sigma must exceed min_sigma")
    log_sigma = math.log(raw_sigma)
    return redistributor_from_log_sigmas(
        log_sigma,
        log_sigma,
        min_sigma,
    ).to(device)


def redistributors_from_sigma_record(
    sigma_record: dict,
    device: torch.device,
) -> tuple[
    AsymGaussianRedistributor,
    AsymGaussianRedistributor,
    AsymGaussianRedistributor,
]:
    """Build fixed, RMS-side-scale, and learned asymmetric redistributors."""
    asymmetric = redistributor_from_log_sigmas(
        sigma_record["log_sigma_left"],
        sigma_record["log_sigma_right"],
        sigma_record["min_sigma"],
    ).to(device)
    fixed_sigma = float(
        sigma_record.get(
            "sigma_symmetric_fixed",
            FIXED_SYMMETRIC_SIGMA,
        )
    )
    learned_sigma_left = float(
        sigma_record.get(
            "sigma_left",
            math.exp(float(sigma_record["log_sigma_left"]))
            + float(sigma_record["min_sigma"]),
        )
    )
    learned_sigma_right = float(
        sigma_record.get(
            "sigma_right",
            math.exp(float(sigma_record["log_sigma_right"]))
            + float(sigma_record["min_sigma"]),
        )
    )
    rms_sigma = float(
        sigma_record.get(
            "sigma_symmetric_rms_scale",
            rms_scale_symmetric_sigma(
                learned_sigma_left,
                learned_sigma_right,
            ),
        )
    )
    fixed_symmetric = symmetric_redistributor(
        fixed_sigma,
        sigma_record["min_sigma"],
        device,
    )
    rms_symmetric = symmetric_redistributor(
        rms_sigma,
        sigma_record["min_sigma"],
        device,
    )
    return fixed_symmetric, rms_symmetric, asymmetric


def apply_redistributors(
    values: torch.Tensor,
    attention_mask: torch.Tensor,
    fixed_symmetric: AsymGaussianRedistributor,
    rms_symmetric: AsymGaussianRedistributor,
    asymmetric: AsymGaussianRedistributor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply an already constructed fixed/RMS/asymmetric module triple."""
    with torch.inference_mode():
        asymmetric_values = asymmetric(values, attention_mask)
        fixed_symmetric_values = fixed_symmetric(values, attention_mask)
        rms_symmetric_values = rms_symmetric(values, attention_mask)
    return (
        fixed_symmetric_values,
        rms_symmetric_values,
        asymmetric_values,
    )


def build_redistribution_attention_mask(
    attention_mask: torch.Tensor,
    special_tokens_mask: list[int] | torch.Tensor,
    include_special_tokens: bool = True,
) -> torch.Tensor:
    """Build the production-faithful or special-token-excluding mask."""
    if attention_mask.dim() != 2:
        raise ValueError(
            f"attention_mask must be two-dimensional, got {tuple(attention_mask.shape)}"
        )
    special_mask = torch.as_tensor(
        special_tokens_mask,
        device=attention_mask.device,
    )
    if special_mask.dim() == 1:
        if attention_mask.shape[0] != 1:
            raise ValueError(
                "A one-dimensional special_tokens_mask requires batch size 1"
            )
        special_mask = special_mask.unsqueeze(0)
    if special_mask.shape != attention_mask.shape:
        raise ValueError(
            "special_tokens_mask shape does not match attention_mask: "
            f"{tuple(special_mask.shape)} != {tuple(attention_mask.shape)}"
        )
    if include_special_tokens:
        return attention_mask.clone()
    non_special = special_mask.eq(0).to(attention_mask.dtype)
    return attention_mask * non_special


def validate_mass_conservation(
    before: torch.Tensor,
    after: torch.Tensor,
    attention_mask: torch.Tensor,
    tolerance: float = 1e-5,
) -> float:
    """Return mass difference and fail when valid-token mass is not conserved."""
    mask = attention_mask.to(before.dtype)
    before_mass = float((before * mask).sum().item())
    after_mass = float((after * mask).sum().item())
    difference = after_mass - before_mass
    scale = max(1.0, abs(before_mass))
    if abs(difference) > tolerance * scale:
        raise ValueError(
            f"Redistribution mass changed from {before_mass} to {after_mass}"
        )
    return difference


def redistribution_mass_audit(
    before: torch.Tensor,
    after: torch.Tensor,
    attention_mask: torch.Tensor,
    redistribution_mask: torch.Tensor,
    special_tokens_mask: list[int] | torch.Tensor,
    assignments: list[int | None],
    evaluable_word_ids: set[int] | None = None,
) -> dict[str, float]:
    """Summarize conserved, word-assigned, and unassigned-special mass."""
    if before.shape != after.shape:
        raise ValueError(f"Mass audit shape mismatch: {before.shape} != {after.shape}")
    if before.shape != attention_mask.shape:
        raise ValueError(
            "Mass audit attention shape does not match values: "
            f"{attention_mask.shape} != {before.shape}"
        )
    if redistribution_mask.shape != before.shape:
        raise ValueError(
            "Mass audit redistribution mask does not match values: "
            f"{redistribution_mask.shape} != {before.shape}"
        )
    if before.shape[0] != 1 or len(assignments) != before.shape[1]:
        raise ValueError("Mass audit requires one passage and one assignment per token")

    special_mask = torch.as_tensor(
        special_tokens_mask,
        device=before.device,
    )
    if special_mask.dim() == 1:
        special_mask = special_mask.unsqueeze(0)
    if special_mask.shape != before.shape:
        raise ValueError(
            "Mass audit special-token shape does not match values: "
            f"{special_mask.shape} != {before.shape}"
        )

    attention = attention_mask.to(before.dtype)
    selected = redistribution_mask.to(before.dtype)
    assigned = torch.tensor(
        [[word_id is not None for word_id in assignments]],
        dtype=torch.bool,
        device=before.device,
    )
    assigned_word_ids = {int(word_id) for word_id in assignments if word_id is not None}
    if evaluable_word_ids is None:
        evaluable_word_ids = assigned_word_ids
    else:
        evaluable_word_ids = {int(word_id) for word_id in evaluable_word_ids}
    unknown_evaluable_ids = evaluable_word_ids - assigned_word_ids
    if unknown_evaluable_ids:
        raise ValueError(
            "Evaluable words do not have assigned ET1 tokens: "
            f"{sorted(unknown_evaluable_ids)}"
        )
    evaluable = torch.tensor(
        [
            [
                word_id is not None and int(word_id) in evaluable_word_ids
                for word_id in assignments
            ]
        ],
        dtype=torch.bool,
        device=before.device,
    )
    special = special_mask.ne(0)
    if bool((assigned & special).any()):
        raise ValueError("A special token cannot also be assigned to a word")

    word_mask = (assigned & attention_mask.ne(0)).to(before.dtype)
    evaluable_word_mask = (evaluable & attention_mask.ne(0)).to(before.dtype)
    unassigned_special_mask = (special & ~assigned & attention_mask.ne(0)).to(
        before.dtype
    )

    def masked_mass(values: torch.Tensor, mask: torch.Tensor) -> float:
        """Return scalar mass under one token mask."""
        return float((values * mask).sum().item())

    raw_valid_mass = masked_mass(before, selected)
    redistributed_valid_mass = masked_mass(after, selected)
    raw_attention_mass = masked_mass(before, attention)
    redistributed_attention_mass = masked_mass(after, attention)
    raw_word_mass = masked_mass(before, word_mask)
    redistributed_word_mass = masked_mass(after, word_mask)
    raw_evaluable_word_mass = masked_mass(before, evaluable_word_mask)
    redistributed_evaluable_word_mass = masked_mass(
        after,
        evaluable_word_mask,
    )
    raw_special_mass = masked_mass(before, unassigned_special_mask)
    redistributed_special_mass = masked_mass(
        after,
        unassigned_special_mask,
    )
    if abs(raw_word_mass) <= 1e-12:
        retention_fraction = float("nan")
    else:
        retention_fraction = redistributed_word_mass / raw_word_mass
    if abs(raw_attention_mass) <= 1e-12:
        raw_evaluable_fraction = float("nan")
    else:
        raw_evaluable_fraction = raw_evaluable_word_mass / raw_attention_mass
    if abs(redistributed_attention_mass) <= 1e-12:
        redistributed_evaluable_fraction = float("nan")
    else:
        redistributed_evaluable_fraction = (
            redistributed_evaluable_word_mass / redistributed_attention_mass
        )
    if abs(raw_evaluable_word_mass) <= 1e-12:
        evaluable_retention = float("nan")
    else:
        evaluable_retention = (
            redistributed_evaluable_word_mass / raw_evaluable_word_mass
        )

    return {
        "raw_valid_token_mass": raw_valid_mass,
        "redistributed_valid_token_mass": redistributed_valid_mass,
        "valid_token_mass_difference": (redistributed_valid_mass - raw_valid_mass),
        "raw_attention_token_mass": raw_attention_mass,
        "redistributed_attention_token_mass": redistributed_attention_mass,
        "raw_word_assigned_mass": raw_word_mass,
        "redistributed_word_assigned_mass": redistributed_word_mass,
        "raw_evaluable_word_mass": raw_evaluable_word_mass,
        "redistributed_evaluable_word_mass": (redistributed_evaluable_word_mass),
        "raw_evaluable_mass_fraction_of_attention": (raw_evaluable_fraction),
        "redistributed_evaluable_mass_fraction_of_attention": (
            redistributed_evaluable_fraction
        ),
        "evaluable_mass_retention_vs_raw": evaluable_retention,
        "raw_unassigned_special_mass": raw_special_mass,
        "redistributed_unassigned_special_mass": (redistributed_special_mass),
        "word_mass_retention_fraction": retention_fraction,
    }


class ET1NativePredictor:
    """Load one frozen ET1 predictor and expose native T5 offsets."""

    def __init__(
        self,
        checkpoint_path: Path | None = None,
        tokenizer_path: Path | None = None,
        tokenizer_cache_dir: Path | None = None,
    ):
        from .et1_model import ET1NativeModel

        self.predictor = ET1NativeModel(
            checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path,
            tokenizer_cache_dir=tokenizer_cache_dir,
        )

    def predict(self, text: str) -> dict:
        """Predict one passage and return checked native-token metadata."""
        tokenizer = self.predictor.fixTokenizer
        encoded = tokenizer(
            [text],
            padding=False,
            add_special_tokens=True,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        values, attention_mask = self.predictor.forward(sentences=[text])
        values = values.detach().cpu()
        attention_mask = attention_mask.detach().cpu()
        input_ids = encoded["input_ids"].detach().cpu()
        encoded_attention = encoded["attention_mask"].detach().cpu()
        if not torch.equal(attention_mask, encoded_attention):
            raise ValueError("ET1 prediction and offset attention masks differ")
        if values.shape != attention_mask.shape:
            raise ValueError(
                f"ET1 output shape {values.shape} does not match "
                f"attention shape {attention_mask.shape}"
            )
        if input_ids.shape != values.shape:
            raise ValueError(
                f"ET1 token shape {input_ids.shape} does not match "
                f"prediction shape {values.shape}"
            )
        offsets = [
            tuple(map(int, item)) for item in encoded["offset_mapping"][0].tolist()
        ]
        special_tokens_mask = [
            int(item) for item in encoded["special_tokens_mask"][0].tolist()
        ]
        assignments, spans = assign_token_offsets(
            text,
            offsets,
            special_tokens_mask,
        )
        return {
            "values": values,
            "attention_mask": attention_mask,
            "input_ids": input_ids,
            "tokens": tokenizer.convert_ids_to_tokens(input_ids[0].tolist()),
            "offsets": offsets,
            "special_tokens_mask": special_tokens_mask,
            "word_assignments": assignments,
            "word_spans": spans,
            "device": str(self.predictor.device),
            "cache_signature": self.predictor.cache_signature,
        }


def aggregate_tokens_to_words(
    token_values: np.ndarray,
    assignments: list[int | None],
    expected_word_count: int,
) -> dict[int, float]:
    """Sum native ET1 token values into every whitespace-token word."""
    totals = {word_id: 0.0 for word_id in range(expected_word_count)}
    coverage = {word_id: 0 for word_id in range(expected_word_count)}
    for value, word_id in zip(token_values, assignments):
        if word_id is None:
            continue
        totals[word_id] += float(value)
        coverage[word_id] += 1
    missing = [word_id for word_id, count in coverage.items() if count == 0]
    if missing:
        raise ValueError(f"Passage words without ET1 tokens: {missing}")
    return totals


def run_et1_inference(
    passages: pd.DataFrame,
    canonical_words: pd.DataFrame,
    sigma_records: list[dict],
    predictor: ET1NativePredictor,
    include_special_tokens_in_redistribution: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Generate ET1 tables with production-faithful special-token handling."""
    token_rows = []
    word_rows = []
    mass_rows = []
    metric_evaluable_word_count = 0
    effective_records = sigma_records or [
        {
            "checkpoint_id": "raw_only",
            "checkpoint": None,
            "sigma_left": None,
            "sigma_right": None,
            "sigma_symmetric": None,
        }
    ]
    redistributors = {
        record["checkpoint_id"]: redistributors_from_sigma_record(
            record,
            torch.device("cpu"),
        )
        for record in effective_records
        if record["checkpoint"] is not None
    }
    canonical_by_passage = {
        int(passage_id): group.sort_values("word_id_zero_based")
        for passage_id, group in canonical_words.groupby(
            "passage_id_zero_based",
            sort=True,
        )
    }

    for passage in passages.sort_values("passage_id_zero_based").itertuples():
        predicted = predictor.predict(passage.passage_text)
        raw = predicted["values"]
        mask = predicted["attention_mask"]
        redistribution_mask = build_redistribution_attention_mask(
            mask,
            predicted["special_tokens_mask"],
            include_special_tokens=(include_special_tokens_in_redistribution),
        )
        assignments = predicted["word_assignments"]
        word_count = len(predicted["word_spans"])
        passage_words = canonical_by_passage[int(passage.passage_id_zero_based)]
        if "ob1_evaluable" in passage_words:
            eligibility = passage_words["ob1_evaluable"]
            if eligibility.dtype != bool:
                normalized = eligibility.astype(str).str.lower()
                if not normalized.isin({"true", "false"}).all():
                    raise ValueError("Canonical ob1_evaluable values must be boolean")
                eligibility = normalized.eq("true")
            evaluation_words = passage_words.loc[eligibility]
        else:
            evaluation_words = passage_words
        metric_evaluable_word_count += len(evaluation_words)
        evaluable_word_ids = {
            int(word_id) for word_id in evaluation_words["word_id_zero_based"].tolist()
        }
        raw_numpy = raw[0].numpy()
        raw_word_totals = aggregate_tokens_to_words(
            raw_numpy,
            assignments,
            word_count,
        )

        for sigma_record in effective_records:
            checkpoint_id = sigma_record["checkpoint_id"]
            mass_context = {
                "checkpoint_id": checkpoint_id,
                "passage_id_zero_based": passage.passage_id_zero_based,
                "special_tokens_included_in_redistribution": (
                    include_special_tokens_in_redistribution
                ),
                "redistribution_special_token_policy": (
                    "include" if include_special_tokens_in_redistribution else "exclude"
                ),
            }
            mass_rows.append(
                {
                    **mass_context,
                    "condition": "et1_raw",
                    **redistribution_mass_audit(
                        raw,
                        raw,
                        mask,
                        redistribution_mask,
                        predicted["special_tokens_mask"],
                        assignments,
                        evaluable_word_ids,
                    ),
                }
            )
            if sigma_record["checkpoint"] is None:
                fixed_symmetric = rms_symmetric = asymmetric = None
                fixed_symmetric_word_totals = {}
                rms_symmetric_word_totals = {}
                asymmetric_word_totals = {}
            else:
                fixed_symmetric, rms_symmetric, asymmetric = apply_redistributors(
                    raw,
                    redistribution_mask,
                    *redistributors[checkpoint_id],
                )
                fixed_symmetric_difference = validate_mass_conservation(
                    raw,
                    fixed_symmetric,
                    redistribution_mask,
                )
                rms_symmetric_difference = validate_mass_conservation(
                    raw,
                    rms_symmetric,
                    redistribution_mask,
                )
                asymmetric_difference = validate_mass_conservation(
                    raw,
                    asymmetric,
                    redistribution_mask,
                )
                fixed_symmetric_word_totals = aggregate_tokens_to_words(
                    fixed_symmetric[0].numpy(),
                    assignments,
                    word_count,
                )
                rms_symmetric_word_totals = aggregate_tokens_to_words(
                    rms_symmetric[0].numpy(),
                    assignments,
                    word_count,
                )
                asymmetric_word_totals = aggregate_tokens_to_words(
                    asymmetric[0].numpy(),
                    assignments,
                    word_count,
                )
                mass_rows.extend(
                    [
                        {
                            **mass_context,
                            "condition": "et1_symmetric",
                            **redistribution_mass_audit(
                                raw,
                                fixed_symmetric,
                                mask,
                                redistribution_mask,
                                predicted["special_tokens_mask"],
                                assignments,
                                evaluable_word_ids,
                            ),
                        },
                        {
                            **mass_context,
                            "condition": "et1_rms_side_scale_symmetric",
                            **redistribution_mass_audit(
                                raw,
                                rms_symmetric,
                                mask,
                                redistribution_mask,
                                predicted["special_tokens_mask"],
                                assignments,
                                evaluable_word_ids,
                            ),
                        },
                        {
                            **mass_context,
                            "condition": "et1_asymmetric",
                            **redistribution_mass_audit(
                                raw,
                                asymmetric,
                                mask,
                                redistribution_mask,
                                predicted["special_tokens_mask"],
                                assignments,
                                evaluable_word_ids,
                            ),
                        },
                    ]
                )
                if not math.isclose(
                    mass_rows[-3]["valid_token_mass_difference"],
                    fixed_symmetric_difference,
                    rel_tol=0.0,
                    abs_tol=1e-7,
                ):
                    raise ValueError("Fixed symmetric mass-audit difference mismatch")
                if not math.isclose(
                    mass_rows[-2]["valid_token_mass_difference"],
                    rms_symmetric_difference,
                    rel_tol=0.0,
                    abs_tol=1e-7,
                ):
                    raise ValueError("RMS symmetric mass-audit difference mismatch")
                if not math.isclose(
                    mass_rows[-1]["valid_token_mass_difference"],
                    asymmetric_difference,
                    rel_tol=0.0,
                    abs_tol=1e-7,
                ):
                    raise ValueError("Asymmetric mass-audit difference mismatch")

            for token_index, (
                token_id,
                token,
                offset,
                is_special,
                attention,
                word_id,
                raw_value,
            ) in enumerate(
                zip(
                    predicted["input_ids"][0].tolist(),
                    predicted["tokens"],
                    predicted["offsets"],
                    predicted["special_tokens_mask"],
                    mask[0].tolist(),
                    assignments,
                    raw[0].tolist(),
                )
            ):
                token_rows.append(
                    {
                        "checkpoint_id": checkpoint_id,
                        "passage_id_zero_based": passage.passage_id_zero_based,
                        "token_index": token_index,
                        "token_id": token_id,
                        "token": token,
                        "character_start": offset[0],
                        "character_end": offset[1],
                        "is_special": is_special,
                        "attention_mask": int(attention),
                        "redistribution_mask": int(redistribution_mask[0, token_index]),
                        "word_id_zero_based": word_id,
                        "et1_raw_token_trt": raw_value,
                        "et1_symmetric_token_trt": (
                            float(fixed_symmetric[0, token_index])
                            if fixed_symmetric is not None
                            else np.nan
                        ),
                        "et1_rms_side_scale_symmetric_token_trt": (
                            float(rms_symmetric[0, token_index])
                            if rms_symmetric is not None
                            else np.nan
                        ),
                        "et1_asymmetric_token_trt": (
                            float(asymmetric[0, token_index])
                            if asymmetric is not None
                            else np.nan
                        ),
                    }
                )

            for word in passage_words.itertuples():
                word_id = int(word.word_id_zero_based)
                word_rows.append(
                    {
                        "checkpoint_id": checkpoint_id,
                        "passage_id_raw": int(word.passage_id_raw),
                        "passage_id_zero_based": int(word.passage_id_zero_based),
                        "word_id_zero_based": word_id,
                        "word_raw": word.word_raw,
                        "et1_raw_word_trt": raw_word_totals[word_id],
                        "et1_symmetric_word_trt": (
                            fixed_symmetric_word_totals[word_id]
                            if fixed_symmetric is not None
                            else np.nan
                        ),
                        "et1_rms_side_scale_symmetric_word_trt": (
                            rms_symmetric_word_totals[word_id]
                            if rms_symmetric is not None
                            else np.nan
                        ),
                        "et1_asymmetric_word_trt": (
                            asymmetric_word_totals[word_id]
                            if asymmetric is not None
                            else np.nan
                        ),
                    }
                )

    audit = {
        "passages": int(passages["passage_id_zero_based"].nunique()),
        "canonical_words": len(canonical_words),
        "metric_evaluable_words": int(metric_evaluable_word_count),
        "checkpoint_ids": [
            str(record["checkpoint_id"]) for record in effective_records
        ],
        "token_rows": len(token_rows),
        "word_rows": len(word_rows),
        "mass_checks": len(mass_rows),
        "special_tokens_included_in_redistribution": (
            include_special_tokens_in_redistribution
        ),
        "redistribution_special_token_policy": (
            "include" if include_special_tokens_in_redistribution else "exclude"
        ),
        "predictor_device": predictor.predictor.device.type,
        "predictor_cache_signature": predictor.predictor.cache_signature,
    }
    return (
        pd.DataFrame(token_rows),
        pd.DataFrame(word_rows),
        pd.DataFrame(mass_rows),
        audit,
    )


def write_et1_outputs(
    output_dir: Path,
    token_frame: pd.DataFrame,
    word_frame: pd.DataFrame,
    mass_frame: pd.DataFrame,
    audit: dict,
) -> None:
    """Write ET1 token, word, mass, and provenance outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    token_frame.to_csv(output_dir / "et1_token_values.csv", index=False)
    word_frame.to_csv(output_dir / "et1_word_values.csv", index=False)
    mass_frame.to_csv(output_dir / "et1_mass_audit.csv", index=False)
    with (output_dir / "et1_inference_audit.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
