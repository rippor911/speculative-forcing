from __future__ import annotations

import gc
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from run_overfit import (
    load_generator,
    resolve_depth_module,
)
from train_short import (
    BLOCK_FRAMES,
    CONFIG_PATH,
    EXPECTED_SHA256,
    build_states,
    build_steps,
    evaluate,
    load_manifest,
    load_samples,
    move_to_cpu,
    move_to_device,
    representative_backbone_snapshots,
    run_state,
    verify_backbone_snapshots,
)
from utils.wan_wrapper import WanTextEncoder


EXP_DIR = Path(
    "experiments/E0207A_depth1_multistate_training"
)

BASE_MCP_PATH = Path(
    "experiments/E0203_mcp_short_training/"
    "mcp_step_0032.pt"
)

BEST_CHECKPOINT_PATH = (
    EXP_DIR / "mcp_depth1_best.pt"
)
FINAL_CHECKPOINT_PATH = (
    EXP_DIR / "mcp_depth1_final.pt"
)
REPORT_PATH = EXP_DIR / "report.json"

NUM_UPDATES = 256
EVAL_EVERY = 16

LEARNING_RATE = 5.0e-5
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 10.0

ORDER_SEED = 20260730
FRESH_LOAD_TOLERANCE = 1.0e-6

MIN_TRAIN_IMPROVEMENT = 0.10
MIN_VALIDATION_IMPROVEMENT = 0.02


def module_state_cpu(
    module: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value
        in module.state_dict().items()
    }


def save_checkpoint(
    *,
    path: Path,
    step: int,
    mcp_state: dict[str, torch.Tensor],
    metrics: dict,
    optimizer_state: dict | None,
) -> None:
    payload = {
        "format": (
            "mcp_depth1_multistate_v1"
        ),
        "base_checkpoint": str(
            BASE_MCP_PATH.resolve()
        ),
        "backbone_sha256": (
            EXPECTED_SHA256
        ),
        "step": step,
        "trainable_scope": (
            "mcp.mcp_modules.0.*"
        ),
        "mcp": mcp_state,
        "metrics": metrics,
    }

    if optimizer_state is not None:
        payload["optimizer"] = (
            optimizer_state
        )

    torch.save(payload, path)


def depth1_noise_baseline(
    *,
    states: list[dict],
    samples: dict[int, dict],
) -> float:
    losses = []

    for state in states:
        sample_index = int(
            state["sample_index"]
        )
        anchor_block = int(
            state["anchor_block"]
        )

        payload = samples[sample_index]

        target_block = anchor_block + 1
        start = (
            target_block * BLOCK_FRAMES
        )
        end = start + BLOCK_FRAMES

        source_noise = payload[
            "source_noise"
        ][
            :,
            start:end,
        ].float()

        target_chunk = payload[
            "target_latent"
        ][
            :,
            start:end,
        ].float()

        loss = torch.nn.functional.mse_loss(
            source_noise,
            target_chunk,
        )

        losses.append(float(loss.item()))

    return sum(losses) / len(losses)


def relative_improvement(
    before: float,
    after: float,
) -> float:
    return (
        before - after
    ) / max(before, 1.0e-12)


def make_training_schedule(
    states: list[dict],
) -> list[dict]:
    schedule = []

    epoch = 0

    while len(schedule) < NUM_UPDATES:
        current = [
            dict(state)
            for state in states
        ]

        rng = random.Random(
            ORDER_SEED + epoch
        )
        rng.shuffle(current)

        schedule.extend(current)
        epoch += 1

    return schedule[:NUM_UPDATES]


def compare_frozen_mcp(
    *,
    current_state: dict[str, torch.Tensor],
    base_state: dict[str, torch.Tensor],
) -> dict:
    depth1_changed = []
    frozen_changed = []

    for key, current in current_state.items():
        expected = base_state[key]

        equal = torch.equal(
            current.cpu(),
            expected.cpu(),
        )

        if key.startswith(
            "mcp_modules.0."
        ):
            if not equal:
                depth1_changed.append(key)

        elif key.startswith(
            (
                "mcp_modules.1.",
                "mcp_modules.2.",
            )
        ):
            if not equal:
                frozen_changed.append(key)

    return {
        "depth1_changed_tensor_count": (
            len(depth1_changed)
        ),
        "depth1_changed_examples": (
            depth1_changed[:10]
        ),
        "frozen_depth23_changed_tensor_count": (
            len(frozen_changed)
        ),
        "frozen_depth23_changed_examples": (
            frozen_changed[:10]
        ),
    }


def main() -> None:
    EXP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not BASE_MCP_PATH.is_file():
        raise FileNotFoundError(
            BASE_MCP_PATH
        )

    manifest = load_manifest()
    samples = load_samples(manifest)

    train_states, validation_states = (
        build_states(samples)
    )

    if len(train_states) != 8192:
        raise RuntimeError(
            f"Expected 8192 train states, "
            f"found {len(train_states)}."
        )

    if len(validation_states) != 1024:
        raise RuntimeError(
            f"Expected 1024 validation states, "
            f"found {len(validation_states)}."
        )

    train_noise_mse = (
        depth1_noise_baseline(
            states=train_states,
            samples=samples,
        )
    )

    validation_noise_mse = (
        depth1_noise_baseline(
            states=validation_states,
            samples=samples,
        )
    )

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    torch.backends.cuda.matmul.allow_tf32 = (
        True
    )

    config = OmegaConf.merge(
        OmegaConf.load(
            "configs/default_config.yaml"
        ),
        OmegaConf.load(CONFIG_PATH),
    )

    print(
        "===== PRECOMPUTE PROMPT EMBEDDINGS =====",
        flush=True,
    )

    text_encoder = WanTextEncoder()
    text_encoder.eval().requires_grad_(False)
    text_encoder.to(
        device=device,
        dtype=torch.bfloat16,
    )

    prompt_embeddings: dict[int, Any] = {}

    for sample_index, payload in sorted(
        samples.items()
    ):
        with torch.inference_mode():
            embedding = text_encoder(
                text_prompts=[
                    payload["prompt"]
                ]
            )

        prompt_embeddings[
            sample_index
        ] = move_to_cpu(embedding)

        del embedding

    text_encoder.to("cpu")
    del text_encoder
    gc.collect()
    torch.cuda.empty_cache()

    print(
        "===== LOAD STEP-32 MCP =====",
        flush=True,
    )

    generator = load_generator(
        config=config,
        device=device,
        train_depth1=True,
    )

    depth1_module = resolve_depth_module(
        generator.mcp,
        0,
    )

    trainable_named = [
        (name, parameter)
        for name, parameter
        in generator.named_parameters()
        if parameter.requires_grad
    ]

    trainable_parameters = [
        parameter
        for _, parameter
        in trainable_named
    ]

    trainable_names = [
        name
        for name, _
        in trainable_named
    ]

    if len(trainable_parameters) != 56:
        raise RuntimeError(
            "Expected 56 trainable depth-1 "
            f"parameter tensors, found "
            f"{len(trainable_parameters)}."
        )

    if any(
        not name.startswith(
            "mcp.mcp_modules.0."
        )
        for name in trainable_names
    ):
        raise RuntimeError(
            "Parameters outside depth 1 "
            "became trainable."
        )

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter
        in trainable_parameters
    )

    if trainable_parameter_count != 132715840:
        raise RuntimeError(
            "Unexpected trainable parameter "
            f"count: {trainable_parameter_count}"
        )

    backbone_named = [
        (name, parameter)
        for name, parameter
        in generator.named_parameters()
        if not name.startswith("mcp.")
    ]

    backbone_snapshots = (
        representative_backbone_snapshots(
            backbone_named
        )
    )

    base_checkpoint = torch.load(
        BASE_MCP_PATH,
        map_location="cpu",
        weights_only=False,
    )

    base_mcp_state = base_checkpoint[
        "mcp"
    ]

    raw_steps, warped_steps = build_steps(
        config,
        generator.get_scheduler(),
    )

    print("raw_steps=", raw_steps)
    print("warped_steps=", warped_steps)
    print(
        "trainable_tensor_count=",
        len(trainable_parameters),
    )
    print(
        "trainable_parameter_count=",
        trainable_parameter_count,
    )

    print(
        "===== INITIAL TRAIN EVALUATION =====",
        flush=True,
    )

    initial_train = evaluate(
        name="initial_train",
        states=train_states,
        samples=samples,
        prompt_embeddings=(
            prompt_embeddings
        ),
        generator=generator,
        denoising_steps=warped_steps,
    )

    print(
        "===== INITIAL VALIDATION =====",
        flush=True,
    )

    initial_validation = evaluate(
        name="initial_validation",
        states=validation_states,
        samples=samples,
        prompt_embeddings=(
            prompt_embeddings
        ),
        generator=generator,
        denoising_steps=warped_steps,
    )

    initial_train_depth1 = float(
        initial_train[
            "depth_losses"
        ][0]
    )

    initial_validation_depth1 = float(
        initial_validation[
            "depth_losses"
        ][0]
    )

    initial_train_progress = (
        1.0
        - initial_train_depth1
        / train_noise_mse
    )

    initial_validation_progress = (
        1.0
        - initial_validation_depth1
        / validation_noise_mse
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        weight_decay=WEIGHT_DECAY,
        foreach=False,
    )

    schedule = make_training_schedule(
        train_states
    )

    validation_history = []
    training_history = []

    best_step = 0
    best_validation = initial_validation
    best_validation_depth1 = (
        initial_validation_depth1
    )
    best_mcp_state = module_state_cpu(
        generator.mcp
    )

    torch.cuda.reset_peak_memory_stats()

    print(
        "===== DEPTH-1 MULTISTATE TRAINING =====",
        flush=True,
    )

    training_start = time.perf_counter()

    for step, state in enumerate(
        schedule,
        start=1,
    ):
        sample_index = int(
            state["sample_index"]
        )
        anchor_block = int(
            state["anchor_block"]
        )

        conditional_dict = move_to_device(
            prompt_embeddings[sample_index],
            device,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        result = run_state(
            generator=generator,
            payload=samples[sample_index],
            conditional_dict=(
                conditional_dict
            ),
            denoising_steps=warped_steps,
            anchor_block=anchor_block,
            require_grad=True,
        )

        loss = result[
            "depth_loss_tensors"
        ][0]

        if not bool(
            torch.isfinite(loss).item()
        ):
            raise RuntimeError(
                f"Nonfinite loss at step {step}."
            )

        loss.backward()

        gradient_norm = (
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                MAX_GRAD_NORM,
            )
        )

        optimizer.step()

        training_record = {
            "step": step,
            "sample_index": sample_index,
            "anchor_block": anchor_block,
            "depth1_loss": float(
                loss.detach().item()
            ),
            "gradient_norm": float(
                gradient_norm.detach()
                .float()
                .item()
            ),
        }

        training_history.append(
            training_record
        )

        print(
            f"train step={step:04d}/{NUM_UPDATES} "
            f"sample={sample_index:03d} "
            f"anchor={anchor_block} "
            f"depth1_loss="
            f"{training_record['depth1_loss']:.8f} "
            f"grad="
            f"{training_record['gradient_norm']:.6f}",
            flush=True,
        )

        del conditional_dict
        del result
        del loss

        if (
            step % EVAL_EVERY != 0
            and step != NUM_UPDATES
        ):
            continue

        validation = evaluate(
            name=(
                f"step_{step:04d}_validation"
            ),
            states=validation_states,
            samples=samples,
            prompt_embeddings=(
                prompt_embeddings
            ),
            generator=generator,
            denoising_steps=warped_steps,
        )

        validation_depth1 = float(
            validation[
                "depth_losses"
            ][0]
        )

        validation_progress = (
            1.0
            - validation_depth1
            / validation_noise_mse
        )

        record = {
            "step": step,
            "depth1_loss": (
                validation_depth1
            ),
            "progress_to_target": (
                validation_progress
            ),
            "metrics": validation,
        }

        validation_history.append(record)

        print(
            f"validation step={step:04d} "
            f"depth1_loss="
            f"{validation_depth1:.8f} "
            f"progress="
            f"{validation_progress:.6f}",
            flush=True,
        )

        if (
            validation_depth1
            < best_validation_depth1
        ):
            best_step = step
            best_validation = validation
            best_validation_depth1 = (
                validation_depth1
            )

            del best_mcp_state
            best_mcp_state = (
                module_state_cpu(
                    generator.mcp
                )
            )

            print(
                f"new_best_step={best_step} "
                f"new_best_depth1="
                f"{best_validation_depth1:.8f}",
                flush=True,
            )

    training_seconds = (
        time.perf_counter()
        - training_start
    )

    print(
        "===== FINAL TRAIN EVALUATION =====",
        flush=True,
    )

    final_train = evaluate(
        name="final_train",
        states=train_states,
        samples=samples,
        prompt_embeddings=(
            prompt_embeddings
        ),
        generator=generator,
        denoising_steps=warped_steps,
    )

    final_validation = (
        validation_history[-1]["metrics"]
    )

    current_mcp_state = module_state_cpu(
        generator.mcp
    )

    frozen_comparison = (
        compare_frozen_mcp(
            current_state=(
                current_mcp_state
            ),
            base_state=base_mcp_state,
        )
    )

    verify_backbone_snapshots(
        generator,
        backbone_snapshots,
    )

    final_optimizer_state = move_to_cpu(
        optimizer.state_dict()
    )

    save_checkpoint(
        path=BEST_CHECKPOINT_PATH,
        step=best_step,
        mcp_state=best_mcp_state,
        metrics={
            "validation": best_validation,
        },
        optimizer_state=None,
    )

    save_checkpoint(
        path=FINAL_CHECKPOINT_PATH,
        step=NUM_UPDATES,
        mcp_state=current_mcp_state,
        metrics={
            "train": final_train,
            "validation": final_validation,
        },
        optimizer_state=(
            final_optimizer_state
        ),
    )

    peak_allocated_gib = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    final_train_depth1 = float(
        final_train[
            "depth_losses"
        ][0]
    )

    train_improvement = (
        relative_improvement(
            initial_train_depth1,
            final_train_depth1,
        )
    )

    validation_improvement = (
        relative_improvement(
            initial_validation_depth1,
            best_validation_depth1,
        )
    )

    generator.to("cpu")
    del generator
    del optimizer
    del depth1_module
    del trainable_parameters
    del current_mcp_state
    del base_mcp_state
    del base_checkpoint
    gc.collect()
    torch.cuda.empty_cache()

    print(
        "===== FRESH-LOAD BEST CHECKPOINT =====",
        flush=True,
    )

    fresh_generator = load_generator(
        config=config,
        device=device,
        train_depth1=False,
    )

    serialized_best = torch.load(
        BEST_CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    restore_result = (
        fresh_generator.mcp.load_state_dict(
            serialized_best["mcp"],
            strict=True,
        )
    )

    if (
        restore_result.missing_keys
        or restore_result.unexpected_keys
    ):
        raise RuntimeError(
            "Fresh-load best MCP failed."
        )

    fresh_validation = evaluate(
        name="fresh_best_validation",
        states=validation_states,
        samples=samples,
        prompt_embeddings=(
            prompt_embeddings
        ),
        generator=fresh_generator,
        denoising_steps=warped_steps,
    )

    fresh_depth_differences = [
        abs(
            float(observed)
            - float(expected)
        )
        for observed, expected in zip(
            fresh_validation[
                "depth_losses"
            ],
            best_validation[
                "depth_losses"
            ],
        )
    ]

    fresh_total_difference = abs(
        float(
            fresh_validation[
                "total_loss"
            ]
        )
        - float(
            best_validation[
                "total_loss"
            ]
        )
    )

    fresh_load_exact = bool(
        fresh_total_difference
        <= FRESH_LOAD_TOLERANCE
        and all(
            difference
            <= FRESH_LOAD_TOLERANCE
            for difference
            in fresh_depth_differences
        )
    )

    fresh_generator.to("cpu")
    del fresh_generator
    gc.collect()
    torch.cuda.empty_cache()

    gates = {
        "train_depth1_improved_at_least_10_percent": (
            train_improvement
            >= MIN_TRAIN_IMPROVEMENT
        ),
        "validation_depth1_improved_at_least_2_percent": (
            validation_improvement
            >= MIN_VALIDATION_IMPROVEMENT
        ),
        "best_step_after_initialization": (
            best_step > 0
        ),
        "depth1_parameters_changed": (
            frozen_comparison[
                "depth1_changed_tensor_count"
            ]
            > 0
        ),
        "depth2_and_depth3_unchanged": (
            frozen_comparison[
                "frozen_depth23_changed_tensor_count"
            ]
            == 0
        ),
        "backbone_unchanged": True,
        "fresh_load_exact": (
            fresh_load_exact
        ),
        "all_metrics_finite": all(
            torch.isfinite(
                torch.tensor(value)
            ).item()
            for value in (
                initial_train_depth1,
                final_train_depth1,
                initial_validation_depth1,
                best_validation_depth1,
            )
        ),
    }

    passed = all(gates.values())

    report = {
        "status": (
            "PASS" if passed else "FAIL"
        ),
        "experiment": (
            "E0207A_depth1_multistate_training"
        ),
        "purpose": (
            "Test depth-1 MCP generalization "
            "across multiple teacher states."
        ),
        "dataset": {
            "train_state_count": (
                len(train_states)
            ),
            "validation_state_count": (
                len(validation_states)
            ),
            "train_sample_indices": [
                0,
                1,
                2,
                3,
            ],
            "validation_sample_indices": [
                4,
                5,
            ],
            "anchors": [
                0,
                1,
                2,
                3,
            ],
            "train_noise_mse": (
                train_noise_mse
            ),
            "validation_noise_mse": (
                validation_noise_mse
            ),
        },
        "training": {
            "base_checkpoint": str(
                BASE_MCP_PATH.resolve()
            ),
            "num_updates": NUM_UPDATES,
            "eval_every": EVAL_EVERY,
            "learning_rate": (
                LEARNING_RATE
            ),
            "weight_decay": (
                WEIGHT_DECAY
            ),
            "trainable_scope": (
                "mcp.mcp_modules.0.*"
            ),
            "trainable_tensor_count": (
                len(trainable_names)
            ),
            "trainable_parameter_count": (
                trainable_parameter_count
            ),
            "training_seconds": (
                training_seconds
            ),
            "peak_allocated_gib": (
                peak_allocated_gib
            ),
        },
        "schedule": {
            "raw_steps": raw_steps,
            "warped_steps": warped_steps,
        },
        "metrics": {
            "initial_train": initial_train,
            "initial_validation": (
                initial_validation
            ),
            "final_train": final_train,
            "final_validation": (
                final_validation
            ),
            "best_validation": (
                best_validation
            ),
            "fresh_best_validation": (
                fresh_validation
            ),
            "initial_train_progress": (
                initial_train_progress
            ),
            "initial_validation_progress": (
                initial_validation_progress
            ),
            "best_validation_progress": (
                1.0
                - best_validation_depth1
                / validation_noise_mse
            ),
            "train_depth1_improvement": (
                train_improvement
            ),
            "validation_depth1_improvement": (
                validation_improvement
            ),
        },
        "best": {
            "step": best_step,
            "checkpoint": str(
                BEST_CHECKPOINT_PATH.resolve()
            ),
            "depth1_loss": (
                best_validation_depth1
            ),
        },
        "final_checkpoint": str(
            FINAL_CHECKPOINT_PATH.resolve()
        ),
        "frozen_parameter_audit": (
            frozen_comparison
        ),
        "fresh_load": {
            "total_difference": (
                fresh_total_difference
            ),
            "depth_differences": (
                fresh_depth_differences
            ),
            "exact": fresh_load_exact,
        },
        "gate": gates,
        "training_history": (
            training_history
        ),
        "validation_history": (
            validation_history
        ),
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("===== RESULT =====")
    print("status=", report["status"])
    print(
        "initial_train_depth1=",
        initial_train_depth1,
    )
    print(
        "final_train_depth1=",
        final_train_depth1,
    )
    print(
        "train_improvement=",
        train_improvement,
    )
    print(
        "initial_validation_depth1=",
        initial_validation_depth1,
    )
    print(
        "best_validation_depth1=",
        best_validation_depth1,
    )
    print(
        "validation_improvement=",
        validation_improvement,
    )
    print("best_step=", best_step)
    print(
        "best_validation_progress=",
        report["metrics"][
            "best_validation_progress"
        ],
    )
    print(
        "depth23_changed=",
        frozen_comparison[
            "frozen_depth23_changed_tensor_count"
        ],
    )
    print(
        "fresh_total_difference=",
        fresh_total_difference,
    )
    print(
        "fresh_depth_differences=",
        fresh_depth_differences,
    )
    print(
        "best_checkpoint=",
        BEST_CHECKPOINT_PATH.resolve(),
    )
    print(
        "final_checkpoint=",
        FINAL_CHECKPOINT_PATH.resolve(),
    )
    print(
        "report=",
        REPORT_PATH.resolve(),
    )


if __name__ == "__main__":
    main()
