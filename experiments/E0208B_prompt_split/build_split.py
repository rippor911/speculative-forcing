from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


EXP_DIR = Path(
    "experiments/E0208B_prompt_split"
)

SOURCE_PATH = Path(
    "prompts/vidprom_filtered_extended.txt"
)

MOVIEGEN_PATH = Path(
    "prompts/MovieGenVideoBench_extended.txt"
)

VBENCH_PATH = Path(
    "prompts/vbench/all_dimension_extended.txt"
)

TOKENIZER_PATH = Path(
    "wan_models/Wan2.1-T2V-1.3B/"
    "google/umt5-xxl"
)

TRAIN_COUNT = 2048
VALIDATION_COUNT = 256
RESERVE_COUNT = 256

HASH_NAMESPACE = "E0208B-prompt-split-v1"

COMMAND_PATTERN = re.compile(
    r"(?:^|\s)--[a-zA-Z][\w-]*"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(8 * 1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def normalize(text: str) -> str:
    return " ".join(
        text.strip().split()
    ).casefold()


def load_normalized_set(
    path: Path,
) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(path)

    return {
        normalized
        for line in path.read_text(
            encoding="utf-8",
        ).splitlines()
        if (normalized := normalize(line))
    }


def percentile(
    values: list[int],
    probability: float,
) -> float | None:
    if not values:
        return None

    values = sorted(values)
    position = probability * (
        len(values) - 1
    )

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return float(values[lower])

    weight = position - lower

    return (
        values[lower] * (1.0 - weight)
        + values[upper] * weight
    )


def summarize(
    values: list[int],
) -> dict[str, int | float | None]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "p10": percentile(values, 0.10),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def split_and_order_key(
    normalized: str,
) -> tuple[str, str]:
    digest = hashlib.sha256(
        (
            HASH_NAMESPACE
            + "\0"
            + normalized
        ).encode("utf-8")
    ).hexdigest()

    bucket = int(
        digest[:16],
        16,
    ) % 10000

    if bucket < 9000:
        split = "train"
    elif bucket < 9500:
        split = "validation"
    else:
        split = "reserve"

    return split, digest


def write_prompt_file(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    path.write_text(
        "".join(
            record["prompt"] + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    EXP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_lines = SOURCE_PATH.read_text(
        encoding="utf-8",
    ).splitlines()

    moviegen_set = load_normalized_set(
        MOVIEGEN_PATH
    )
    vbench_set = load_normalized_set(
        VBENCH_PATH
    )

    evaluation_set = (
        moviegen_set | vbench_set
    )

    seen: set[str] = set()

    excluded = Counter()
    candidates: dict[
        str,
        list[dict[str, Any]],
    ] = {
        "train": [],
        "validation": [],
        "reserve": [],
    }

    for line_index, raw_line in enumerate(
        source_lines
    ):
        prompt = " ".join(
            raw_line.strip().split()
        )
        normalized = normalize(prompt)

        if not normalized:
            excluded["empty"] += 1
            continue

        if len(prompt) < 20:
            excluded["shorter_than_20"] += 1
            continue

        if COMMAND_PATTERN.search(prompt):
            excluded["command_argument"] += 1
            continue

        if normalized in seen:
            excluded["duplicate"] += 1
            continue

        seen.add(normalized)

        if normalized in evaluation_set:
            excluded[
                "evaluation_overlap"
            ] += 1
            continue

        split, order_key = (
            split_and_order_key(
                normalized
            )
        )

        candidates[split].append(
            {
                "source_line_index": (
                    line_index
                ),
                "prompt": prompt,
                "normalized_sha256": (
                    hashlib.sha256(
                        normalized.encode(
                            "utf-8"
                        )
                    ).hexdigest()
                ),
                "order_key": order_key,
                "character_count": len(prompt),
                "word_count": len(
                    prompt.split()
                ),
                "is_ascii": (
                    prompt.isascii()
                ),
            }
        )

    for split_records in (
        candidates.values()
    ):
        split_records.sort(
            key=lambda record: (
                record["order_key"]
            )
        )

    required = {
        "train": TRAIN_COUNT,
        "validation": VALIDATION_COUNT,
        "reserve": RESERVE_COUNT,
    }

    selected: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    for split, count in required.items():
        available = candidates[split]

        if len(available) < count:
            raise RuntimeError(
                f"Not enough {split} prompts: "
                f"required={count}, "
                f"available={len(available)}"
            )

        selected[split] = [
            {
                **record,
                "split": split,
                "split_index": index,
            }
            for index, record in enumerate(
                available[:count]
            )
        ]

    tokenizer = (
        AutoTokenizer.from_pretrained(
            TOKENIZER_PATH,
            local_files_only=True,
            use_fast=True,
        )
    )

    all_selected = [
        record
        for split in (
            "train",
            "validation",
            "reserve",
        )
        for record in selected[split]
    ]

    batch_size = 256

    for start in range(
        0,
        len(all_selected),
        batch_size,
    ):
        batch_records = all_selected[
            start:start + batch_size
        ]

        encoded = tokenizer(
            [
                record["prompt"]
                for record in batch_records
            ],
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_length=True,
        )

        for record, token_count in zip(
            batch_records,
            encoded["length"],
        ):
            record["token_count"] = int(
                token_count
            )

    over_512 = [
        record
        for record in all_selected
        if record["token_count"] > 512
    ]

    if over_512:
        raise RuntimeError(
            "Selected prompts exceed "
            f"512 tokens: {len(over_512)}"
        )

    for split, records in selected.items():
        write_prompt_file(
            EXP_DIR
            / f"{split}_prompts.txt",
            records,
        )

        write_jsonl(
            EXP_DIR
            / f"{split}_records.jsonl",
            records,
        )

    report = {
        "status": "PASS",
        "experiment": (
            "E0208B_prompt_split"
        ),
        "source": {
            "path": str(SOURCE_PATH),
            "resolved_path": str(
                SOURCE_PATH.resolve()
            ),
            "sha256": file_sha256(
                SOURCE_PATH
            ),
            "raw_line_count": len(
                source_lines
            ),
        },
        "filtering": {
            "excluded": dict(excluded),
            "eligible_unique_count": sum(
                len(records)
                for records
                in candidates.values()
            ),
            "candidate_split_counts": {
                split: len(records)
                for split, records
                in candidates.items()
            },
        },
        "selection": {},
        "rules": {
            "minimum_characters": 20,
            "remove_command_arguments": True,
            "remove_normalized_duplicates": True,
            "exclude_moviegen": True,
            "exclude_vbench": True,
            "maximum_token_count": 512,
            "hash_namespace": HASH_NAMESPACE,
            "hash_split_ratio": {
                "train": 0.90,
                "validation": 0.05,
                "reserve": 0.05,
            },
        },
    }

    for split, records in selected.items():
        token_counts = [
            record["token_count"]
            for record in records
        ]

        report["selection"][split] = {
            "count": len(records),
            "ascii_count": sum(
                record["is_ascii"]
                for record in records
            ),
            "non_ascii_count": sum(
                not record["is_ascii"]
                for record in records
            ),
            "character_lengths": summarize(
                [
                    record[
                        "character_count"
                    ]
                    for record in records
                ]
            ),
            "word_lengths": summarize(
                [
                    record["word_count"]
                    for record in records
                ]
            ),
            "token_lengths": summarize(
                token_counts
            ),
            "over_256_tokens": sum(
                count > 256
                for count in token_counts
            ),
            "over_512_tokens": sum(
                count > 512
                for count in token_counts
            ),
            "prompt_file": str(
                (
                    EXP_DIR
                    / f"{split}_prompts.txt"
                ).resolve()
            ),
            "records_file": str(
                (
                    EXP_DIR
                    / f"{split}_records.jsonl"
                ).resolve()
            ),
        }

    report_path = (
        EXP_DIR / "report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("===== RESULT =====")
    print("status=PASS")
    print(
        "train=",
        len(selected["train"]),
    )
    print(
        "validation=",
        len(selected["validation"]),
    )
    print(
        "reserve=",
        len(selected["reserve"]),
    )
    print(
        "excluded=",
        dict(excluded),
    )
    print(
        "report=",
        report_path.resolve(),
    )


if __name__ == "__main__":
    main()
