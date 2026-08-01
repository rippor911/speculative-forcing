from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[2]

root_text = str(ROOT)
if root_text not in sys.path:
    sys.path.insert(0, root_text)

SOURCE_SCRIPT = ROOT / "experiments/E0209_depth1_formal_training/train_depth1_formal_impl.py"
FORMAL_MANIFEST = ROOT / "experiments/E0208C_teacher_rollout_formal/manifest.json"
OUTPUT_DIR = ROOT / "experiments/E0209_depth1_formal_training"
AUDIT_DIR = ROOT / "experiments/E0202_anchor_replay_audit"
OLD_EXPERIMENT_TOKEN = "E0207A_depth1_multistate_training"
NEW_EXPERIMENT_TOKEN = "E0209_depth1_formal_training"
EXPECTED_TRAIN_SAMPLES = 2048
EXPECTED_VALIDATION_SAMPLES = 256
EXPECTED_ANCHORS_PER_SAMPLE = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--schedule-seed", type=int, default=20260731)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_write(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def import_audit_replay(manifest_path: Path):
    sys.path.insert(0, str(AUDIT_DIR))
    import audit_replay  # type: ignore

    audit_replay.MANIFEST_PATH = manifest_path
    audit_replay.DATA_DIR = manifest_path.parent
    return audit_replay


def load_training_module() -> Any:
    # train_depth1.py imports sibling modules such as run_overfit.py.
    source_directory = str(SOURCE_SCRIPT.parent)
    if source_directory not in sys.path:
        sys.path.insert(0, source_directory)

    spec = importlib.util.spec_from_file_location(
        "e0209_depth1_formal_impl",
        SOURCE_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replace_experiment_value(value: Any) -> Any:
    if isinstance(value, Path):
        text = str(value)
        if OLD_EXPERIMENT_TOKEN in text:
            return Path(text.replace(OLD_EXPERIMENT_TOKEN, NEW_EXPERIMENT_TOKEN))
    if isinstance(value, str) and OLD_EXPERIMENT_TOKEN in value:
        return value.replace(OLD_EXPERIMENT_TOKEN, NEW_EXPERIMENT_TOKEN)
    return value


def patch_module_paths(module: Any) -> dict[str, str]:
    changed: dict[str, str] = {}
    for name, value in list(vars(module).items()):
        replacement = replace_experiment_value(value)
        if replacement is not value and replacement != value:
            setattr(module, name, replacement)
            changed[name] = f"{value} -> {replacement}"

    for candidate in ("EXP_DIR", "OUTPUT_DIR", "EXPERIMENT_DIR"):
        if hasattr(module, candidate):
            old = getattr(module, candidate)
            setattr(module, candidate, OUTPUT_DIR)
            changed[candidate] = f"{old} -> {OUTPUT_DIR}"

    return changed


def sequence_candidate(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Sequence[Any]:
    candidates: list[Sequence[Any]] = []
    for value in list(args) + list(kwargs.values()):
        if isinstance(value, (list, tuple)) and len(value) > 0:
            candidates.append(value)
    if not candidates:
        raise RuntimeError(
            "could not locate the training-state sequence passed to make_training_schedule"
        )
    return max(candidates, key=len)


def patch_dataset_index_globals(module: Any, manifest: dict[str, Any]) -> dict[str, Any]:
    train_indices = [
        int(record["sample_index"])
        for record in manifest["samples"]
        if record["split"] == "train"
    ]
    validation_indices = [
        int(record["sample_index"])
        for record in manifest["samples"]
        if record["split"] == "validation"
    ]
    changed: dict[str, Any] = {}

    for name, value in list(vars(module).items()):
        upper = name.upper()
        if not isinstance(value, (list, tuple, set)):
            continue
        if "SAMPLE" not in upper or "IND" not in upper:
            continue
        if "VALID" in upper:
            replacement = type(value)(validation_indices) if not isinstance(value, set) else set(validation_indices)
        elif "TRAIN" in upper:
            replacement = type(value)(train_indices) if not isinstance(value, set) else set(train_indices)
        else:
            continue
        setattr(module, name, replacement)
        changed[name] = {
            "old_length": len(value),
            "new_length": len(replacement),
        }

    return changed


def install_epoch_schedule(module: Any, epochs: int, schedule_seed: int) -> dict[str, Any]:
    if not hasattr(module, "make_training_schedule"):
        raise RuntimeError("E0207A trainer has no make_training_schedule")

    original = module.make_training_schedule
    signature = inspect.signature(original)
    schedule_metadata: dict[str, Any] = {
        "original_signature": str(signature),
        "epochs": epochs,
        "schedule_seed": schedule_seed,
    }

    def formal_schedule(*args: Any, **kwargs: Any) -> list[Any]:
        states = list(sequence_candidate(args, kwargs))
        if not states:
            raise RuntimeError("training state list is empty")
        expected_states = EXPECTED_TRAIN_SAMPLES * EXPECTED_ANCHORS_PER_SAMPLE
        if len(states) != expected_states:
            raise RuntimeError(
                f"expected {expected_states} train states, got {len(states)}; "
                "the source trainer is still filtering the old six-sample dataset"
            )

        schedule: list[Any] = []
        for epoch in range(epochs):
            epoch_states = list(states)
            random.Random(schedule_seed + epoch).shuffle(epoch_states)
            schedule.extend(epoch_states)

        schedule_metadata.update(
            {
                "num_unique_states": len(states),
                "num_updates": len(schedule),
            }
        )
        print(
            f"FORMAL_SCHEDULE states={len(states)} epochs={epochs} "
            f"updates={len(schedule)} seed={schedule_seed}",
            flush=True,
        )
        return schedule

    module.make_training_schedule = formal_schedule

    desired_updates = EXPECTED_TRAIN_SAMPLES * EXPECTED_ANCHORS_PER_SAMPLE * epochs
    for name in (
        "NUM_UPDATES",
        "TOTAL_UPDATES",
        "TRAINING_STEPS",
        "MAX_STEPS",
        "NUM_TRAIN_STEPS",
    ):
        if hasattr(module, name):
            current = getattr(module, name)
            if isinstance(current, int):
                setattr(module, name, desired_updates)
                schedule_metadata[f"patched_global_{name}"] = f"{current} -> {desired_updates}"

    schedule_metadata["desired_updates"] = desired_updates

    desired_eval_every = (
        EXPECTED_TRAIN_SAMPLES
        * EXPECTED_ANCHORS_PER_SAMPLE
    )

    previous_eval_every = getattr(
        module,
        "EVAL_EVERY",
        None,
    )

    module.EVAL_EVERY = desired_eval_every

    schedule_metadata[
        "patched_global_EVAL_EVERY"
    ] = (
        f"{previous_eval_every} -> "
        f"{desired_eval_every}"
    )

    previous_order_seed = getattr(
        module,
        "ORDER_SEED",
        None,
    )

    module.ORDER_SEED = schedule_seed

    schedule_metadata[
        "patched_global_ORDER_SEED"
    ] = (
        f"{previous_order_seed} -> "
        f"{schedule_seed}"
    )

    return schedule_metadata



def install_evaluation_policy(
    module: Any,
) -> dict[str, Any]:
    original_evaluate = getattr(
        module,
        "evaluate",
        None,
    )

    if not callable(original_evaluate):
        raise RuntimeError(
            "Trainer evaluate() is unavailable."
        )

    expected_train_states = (
        EXPECTED_TRAIN_SAMPLES
        * EXPECTED_ANCHORS_PER_SAMPLE
    )

    train_probe_size = (
        EXPECTED_VALIDATION_SAMPLES
        * EXPECTED_ANCHORS_PER_SAMPLE
    )

    def formal_evaluate(
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        states = kwargs.get("states")

        if (
            isinstance(states, (list, tuple))
            and len(states) == expected_train_states
        ):
            indices = [
                (
                    index
                    * expected_train_states
                )
                // train_probe_size
                for index in range(
                    train_probe_size
                )
            ]

            modified_kwargs = dict(kwargs)
            modified_kwargs["states"] = [
                states[index]
                for index in indices
            ]

            print(
                "FORMAL_TRAIN_EVAL_PROBE "
                f"full={expected_train_states} "
                f"probe={train_probe_size}",
                flush=True,
            )

            return original_evaluate(
                *args,
                **modified_kwargs,
            )

        return original_evaluate(
            *args,
            **kwargs,
        )

    module.evaluate = formal_evaluate

    return {
        "train_state_count": (
            expected_train_states
        ),
        "train_probe_state_count": (
            train_probe_size
        ),
        "validation_state_count": (
            EXPECTED_VALIDATION_SAMPLES
            * EXPECTED_ANCHORS_PER_SAMPLE
        ),
        "validation_uses_full_set": True,
    }


def preflight(epochs: int, schedule_seed: int) -> dict[str, Any]:
    if not SOURCE_SCRIPT.is_file():
        raise FileNotFoundError(SOURCE_SCRIPT)
    if not AUDIT_DIR.is_dir():
        raise FileNotFoundError(AUDIT_DIR)

    fallback_manifest = ROOT / "experiments/E0201_teacher_rollout_smoke/manifest.json"
    audit_replay = import_audit_replay(fallback_manifest)
    module = load_training_module()
    path_changes = patch_module_paths(module)
    schedule_metadata = install_epoch_schedule(
        module,
        epochs,
        schedule_seed,
    )
    evaluation_metadata = (
        install_evaluation_policy(module)
    )

    return {
        "status": "PASS",
        "source_script": str(SOURCE_SCRIPT),
        "source_sha256": file_sha256(SOURCE_SCRIPT),
        "audit_module": str(Path(audit_replay.__file__).resolve()),
        "path_changes": path_changes,
        "schedule": schedule_metadata,
        "evaluation": evaluation_metadata,
        "has_main": callable(getattr(module, "main", None)),
        "output_dir": str(OUTPUT_DIR),
    }


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")

    report = preflight(args.epochs, args.schedule_seed)
    print(json.dumps(report, indent=2))
    print("E0209_PREFLIGHT=PASS")

    if args.preflight_only:
        return

    if not FORMAL_MANIFEST.is_file():
        raise FileNotFoundError(FORMAL_MANIFEST)
    manifest = json.loads(FORMAL_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS":
        raise RuntimeError("formal teacher manifest is not PASS")
    if int(manifest["generation"]["num_train"]) != 2048:
        raise RuntimeError("formal train count differs")
    if int(manifest["generation"]["num_validation"]) != 256:
        raise RuntimeError("formal validation count differs")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()

    audit_replay = import_audit_replay(FORMAL_MANIFEST)
    module = load_training_module()
    path_changes = patch_module_paths(module)
    dataset_index_changes = patch_dataset_index_globals(module, manifest)
    schedule_metadata = install_epoch_schedule(
        module,
        args.epochs,
        args.schedule_seed,
    )
    evaluation_metadata = (
        install_evaluation_policy(module)
    )

    # Make the imported function explicit even if the source used
    # `from audit_replay import load_manifest`.
    module.load_manifest = audit_replay.load_manifest

    contract = {
        "status": "RUNNING",
        "source_script": str(SOURCE_SCRIPT),
        "source_sha256": file_sha256(SOURCE_SCRIPT),
        "teacher_manifest": str(FORMAL_MANIFEST),
        "teacher_manifest_sha256": file_sha256(FORMAL_MANIFEST),
        "output_dir": str(OUTPUT_DIR),
        "epochs": args.epochs,
        "schedule_seed": args.schedule_seed,
        "path_changes": path_changes,
        "dataset_index_changes": dataset_index_changes,
        "schedule": schedule_metadata,
        "evaluation": evaluation_metadata,
        "started_unix": started,
    }
    atomic_json_write(contract, OUTPUT_DIR / "formal_training_contract.json")

    if not callable(getattr(module, "main", None)):
        raise RuntimeError("E0207A trainer main() is unavailable")

    module.main()

    checkpoints = sorted(
        str(path.resolve())
        for path in OUTPUT_DIR.rglob("*.pt")
        if path.is_file()
    )
    finished = time.time()
    final_report = {
        **contract,
        "status": "PASS",
        "finished_unix": finished,
        "elapsed_seconds": finished - started,
        "checkpoints": checkpoints,
        "num_checkpoints": len(checkpoints),
    }
    atomic_json_write(final_report, OUTPUT_DIR / "wrapper_status.json")
    print(json.dumps(final_report, indent=2))
    print("E0209_DEPTH1_FORMAL_TRAINING=PASS")


if __name__ == "__main__":
    main()
