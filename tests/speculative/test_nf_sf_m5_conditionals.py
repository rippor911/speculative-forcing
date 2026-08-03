from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch

import utils.nf_sf_m4 as m4
import utils.nf_sf_m5_conditionals as conditionals
import utils.nf_sf_m5_formal_plan as formal_plan
from utils.nf_sf_m3 import M3_REFERENCE_CHECKPOINT_SHA256, file_sha256

TEST_GIT_SHA = "0123456789abcdef0123456789abcdef01234567"
TEST_MODEL_SHA = "1" * 64


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _formal_record(split: str, split_index: int, sample_index: int) -> dict[str, Any]:
    prompt = f"{split} prompt {split_index}"
    file_name = f"payloads/{split}_{split_index:06d}.pt"
    return {
        "status": "GENERATED",
        "sample_index": sample_index,
        "sample_id": f"{split}-{split_index:06d}",
        "split": split,
        "split_index": split_index,
        "source_line_index": sample_index,
        "shard_id": 0,
        "plan_index": sample_index,
        "file": file_name,
        "file_sha256": _sha256_text(f"{file_name}:payload"),
        "prompt": prompt,
        "prompt_sha256": _sha256_text(prompt),
    }


def _records() -> list[dict[str, Any]]:
    values = []
    for split_index in range(formal_plan.M5_FORMAL_TRAIN_SAMPLE_COUNT):
        values.append(_formal_record("train", split_index, split_index))
    for split_index in range(formal_plan.M5_FORMAL_VALIDATION_SAMPLE_COUNT):
        values.append(
            _formal_record("validation", split_index, 100_000 + split_index)
        )
    return values


def _write_manifest(directory: Path) -> Path:
    samples = list(reversed(_records()))
    manifest = {
        "status": "PASS",
        "experiment": "E0208C_teacher_rollout_formal",
        "format": "self_forcing_teacher_manifest_v2",
        "writer_format": "e0208_teacher_writer_v1",
        "writer_git_head": TEST_GIT_SHA,
        "checkpoint": {
            "path": "checkpoints/self_forcing_dmd.pt",
            "sha256": M3_REFERENCE_CHECKPOINT_SHA256,
        },
        "generation": {
            "num_samples": len(samples),
            "num_completed": len(samples),
            "num_train": formal_plan.M5_FORMAL_TRAIN_SAMPLE_COUNT,
            "num_validation": formal_plan.M5_FORMAL_VALIDATION_SAMPLE_COUNT,
            "num_reserve": 0,
            "num_frames": 15,
            "num_frame_per_block": 3,
            "num_blocks": 5,
            "mcp_depth": 3,
            "mcp_num_modules": 0,
            "mcp_accel_depths": 0,
            "last_step_only": True,
        },
        "samples": samples,
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return path


def _case(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    manifest_path = _write_manifest(tmp_path)
    plan = formal_plan.build_m5_formal_sample_plan(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
    )
    return manifest_path, dataset_root, plan


def _entry_by_identity(plan: dict[str, Any], identity: str) -> dict[str, Any]:
    for split in ("train", "validation"):
        for entry in plan["samples"][split]:
            if entry["identity"] == identity:
                return dict(entry)
    raise AssertionError(f"missing identity: {identity}")


def _encoder_provenance() -> dict[str, str]:
    return {
        "encoder_class": "tests.TinyPromptEncoder",
        "model_checkpoint_path": "checkpoints/tiny_text_encoder.pt",
        "model_checkpoint_sha256": TEST_MODEL_SHA,
        "tokenizer_path": "tokenizers/tiny",
        "dtype": "torch.float32",
    }


def _tensor(value: float = 1.0, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.full((1, 2, 3), value, dtype=dtype)


def _write_item(
    artifact_dir: Path,
    *,
    identity: str,
    prompt_sha256: str,
    tensor: torch.Tensor | None = None,
) -> dict[str, Any]:
    tensor = _tensor() if tensor is None else tensor
    rel_path = (
        f"{conditionals.M5_CONDITIONAL_ARTIFACT_ITEMS_DIR}/"
        f"{conditionals.m5_conditional_item_name(identity)}"
    )
    item_path = artifact_dir / rel_path
    item_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": conditionals.M5_CONDITIONAL_ITEM_SCHEMA,
        "identity": identity,
        "prompt_sha256": prompt_sha256,
        "tensors": {"prompt_embeds": tensor.detach().contiguous().cpu()},
    }
    torch.save(payload, item_path)
    return {
        "prompt_sha256": prompt_sha256,
        "item_relative_path": rel_path,
        "item_file_sha256": file_sha256(item_path),
        "tensors": {
            "prompt_embeds": {
                "shape": [int(dim) for dim in tensor.shape],
                "dtype": str(tensor.dtype),
            }
        },
    }


def _write_minimal_artifact(
    artifact_dir: Path,
    plan: dict[str, Any],
    *,
    materialized_identities: tuple[str, ...],
) -> dict[str, Any]:
    artifact_dir.mkdir()
    (artifact_dir / conditionals.M5_CONDITIONAL_ARTIFACT_ITEMS_DIR).mkdir()
    items = {}
    for identity in (*plan["train_sample_identities"], *plan["validation_sample_identities"]):
        entry = _entry_by_identity(plan, identity)
        if identity in materialized_identities:
            items[identity] = _write_item(
                artifact_dir,
                identity=identity,
                prompt_sha256=entry["prompt_sha256"],
            )
        else:
            rel_path = (
                f"{conditionals.M5_CONDITIONAL_ARTIFACT_ITEMS_DIR}/"
                f"{conditionals.m5_conditional_item_name(identity)}"
            )
            items[identity] = {
                "prompt_sha256": entry["prompt_sha256"],
                "item_relative_path": rel_path,
                "item_file_sha256": "0" * 64,
                "tensors": {
                    "prompt_embeds": {
                        "shape": [1, 2, 3],
                        "dtype": "torch.float32",
                    }
                },
            }
    manifest = {
        "schema": conditionals.M5_CONDITIONAL_ARTIFACT_SCHEMA,
        "status": "PASS",
        "sample_plan_sha256": plan["sample_plan_sha256"],
        "teacher_manifest_sha256": plan["manifest_sha256"],
        "encoder_provenance": _encoder_provenance(),
        "train_identities": list(plan["train_sample_identities"]),
        "validation_identities": list(plan["validation_sample_identities"]),
        "items": items,
    }
    manifest["artifact_sha256"] = conditionals.m5_conditional_artifact_sha256(manifest)
    m4.write_m4_json(
        manifest,
        artifact_dir / conditionals.M5_CONDITIONAL_ARTIFACT_MANIFEST_NAME,
    )
    return manifest


class TinyPromptEncoder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.live_count = 0
        self.max_live_count = 0

    def __call__(self, prompts: list[str]) -> dict[str, torch.Tensor]:
        self.calls.append(list(prompts))
        self.live_count += 1
        self.max_live_count = max(self.max_live_count, self.live_count)
        try:
            value = float(len(prompts[0]))
            return {"prompt_embeds": _tensor(value)}
        finally:
            self.live_count -= 1


def test_encode_prompt_condition_smoke_encodes_one_prompt() -> None:
    encoder = TinyPromptEncoder()
    conditional = conditionals.encode_m5_prompt_condition(encoder, "hello")
    assert encoder.calls == [["hello"]]
    assert set(conditional) == {"prompt_embeds"}
    assert conditional["prompt_embeds"].device.type == "cpu"
    assert conditional["prompt_embeds"].is_contiguous()


def test_builder_writes_per_identity_artifact_and_preserves_order(
    tmp_path: Path,
) -> None:
    manifest_path, dataset_root, plan = _case(tmp_path)
    encoder = TinyPromptEncoder()
    output_dir = tmp_path / "conditional_artifact"

    report = conditionals.build_m5_conditional_artifact(
        sample_plan=plan,
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        output_dir=output_dir,
        text_encoder=encoder,
        encoder_provenance=_encoder_provenance(),
    )

    manifest = conditionals.load_m5_conditional_artifact_manifest(
        output_dir / conditionals.M5_CONDITIONAL_ARTIFACT_MANIFEST_NAME
    )
    assert report["status"] == "PASS"
    assert report["encoded_count"] == 2304
    assert report["max_live_conditionals"] == 1
    assert encoder.max_live_count == 1
    assert len(encoder.calls) == 2304
    assert manifest["train_identities"] == plan["train_sample_identities"]
    assert manifest["validation_identities"] == plan["validation_sample_identities"]
    assert manifest["sample_plan_sha256"] == plan["sample_plan_sha256"]
    assert manifest["teacher_manifest_sha256"] == plan["manifest_sha256"]
    assert len(list((output_dir / "items").glob("*.pt"))) == 2304
    assert not list(output_dir.glob("*.pt"))
    conditionals.validate_m5_conditional_artifact_manifest(
        manifest,
        artifact_dir=output_dir,
        sample_plan=plan,
        check_item_files=True,
    )


def test_store_initialization_is_lazy_and_single_acquire_loads_one_item(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _manifest_path, _dataset_root, plan = _case(tmp_path)
    artifact_dir = tmp_path / "artifact"
    identity = plan["train_sample_identities"][0]
    _write_minimal_artifact(
        artifact_dir,
        plan,
        materialized_identities=(identity,),
    )
    real_load = torch.load
    load_calls = []

    def counted_load(*args: Any, **kwargs: Any) -> Any:
        load_calls.append((args, kwargs))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(conditionals.torch, "load", counted_load)
    store = conditionals.M5ConditionalArtifactStore(
        artifact_dir=artifact_dir,
        sample_plan=plan,
    )
    assert load_calls == []
    assert store.train_identities == tuple(plan["train_sample_identities"])
    assert store.validation_identities == tuple(plan["validation_sample_identities"])

    with store.acquire(identity) as conditional:
        assert set(conditional) == {"prompt_embeds"}
        assert store.live_conditional_count == 1
    assert len(load_calls) == 1
    assert store.live_conditional_count == 0
    assert store.max_live_conditional_count == 1
    assert store.total_load_count == 1


def test_store_nested_acquire_rejects(tmp_path: Path) -> None:
    _manifest_path, _dataset_root, plan = _case(tmp_path)
    artifact_dir = tmp_path / "artifact"
    first = plan["train_sample_identities"][0]
    second = plan["train_sample_identities"][1]
    _write_minimal_artifact(
        artifact_dir,
        plan,
        materialized_identities=(first, second),
    )
    store = conditionals.M5ConditionalArtifactStore(
        artifact_dir=artifact_dir,
        sample_plan=plan,
    )

    with store.acquire(first), pytest.raises(RuntimeError, match="active"), store.acquire(
        second
    ):
        pass
    assert store.live_conditional_count == 0
    assert store.max_live_conditional_count == 1


def test_item_sha_tampering_rejects_on_acquire(tmp_path: Path) -> None:
    _manifest_path, _dataset_root, plan = _case(tmp_path)
    artifact_dir = tmp_path / "artifact"
    identity = plan["train_sample_identities"][0]
    manifest = _write_minimal_artifact(
        artifact_dir,
        plan,
        materialized_identities=(identity,),
    )
    item_path = artifact_dir / manifest["items"][identity]["item_relative_path"]
    item_path.write_bytes(item_path.read_bytes() + b"tamper")
    store = conditionals.M5ConditionalArtifactStore(
        artifact_dir=artifact_dir,
        sample_plan=plan,
    )

    with pytest.raises(RuntimeError, match="SHA256 mismatch"), store.acquire(identity):
        pass
    assert store.live_conditional_count == 0
    assert store.total_load_count == 0


@pytest.mark.parametrize(
    ("metadata_key", "value", "message"),
    [
        ("shape", [99], "shape"),
        ("dtype", "torch.float64", "dtype"),
    ],
)
def test_shape_or_dtype_manifest_tampering_rejects_on_acquire(
    tmp_path: Path,
    metadata_key: str,
    value: Any,
    message: str,
) -> None:
    _manifest_path, _dataset_root, plan = _case(tmp_path)
    artifact_dir = tmp_path / "artifact"
    identity = plan["train_sample_identities"][0]
    manifest = _write_minimal_artifact(
        artifact_dir,
        plan,
        materialized_identities=(identity,),
    )
    manifest["items"][identity]["tensors"]["prompt_embeds"][metadata_key] = value
    manifest["artifact_sha256"] = conditionals.m5_conditional_artifact_sha256(manifest)
    m4.write_m4_json(
        manifest,
        artifact_dir / conditionals.M5_CONDITIONAL_ARTIFACT_MANIFEST_NAME,
    )
    store = conditionals.M5ConditionalArtifactStore(
        artifact_dir=artifact_dir,
        sample_plan=plan,
    )

    with pytest.raises(RuntimeError, match=message), store.acquire(identity):
        pass
    assert store.live_conditional_count == 0
    assert store.total_load_count == 0


def test_sample_plan_sha_mismatch_rejects(tmp_path: Path) -> None:
    _manifest_path, _dataset_root, plan = _case(tmp_path)
    artifact_dir = tmp_path / "artifact"
    identity = plan["train_sample_identities"][0]
    _write_minimal_artifact(
        artifact_dir,
        plan,
        materialized_identities=(identity,),
    )
    tampered_plan = copy.deepcopy(plan)
    tampered_plan["sample_plan_sha256"] = "0" * 64

    with pytest.raises(RuntimeError, match="sample plan SHA256"):
        conditionals.M5ConditionalArtifactStore(
            artifact_dir=artifact_dir,
            sample_plan=tampered_plan,
        )


def test_item_prompt_sha_mismatch_against_sample_plan_rejects(
    tmp_path: Path,
) -> None:
    _manifest_path, _dataset_root, plan = _case(tmp_path)
    artifact_dir = tmp_path / "artifact"
    identity = plan["train_sample_identities"][0]
    manifest = _write_minimal_artifact(
        artifact_dir,
        plan,
        materialized_identities=(identity,),
    )
    bad_prompt_sha256 = _sha256_text("different prompt")
    item_path = artifact_dir / manifest["items"][identity]["item_relative_path"]
    payload = torch.load(item_path, map_location="cpu", weights_only=False)
    payload["prompt_sha256"] = bad_prompt_sha256
    torch.save(payload, item_path)
    manifest["items"][identity]["prompt_sha256"] = bad_prompt_sha256
    manifest["items"][identity]["item_file_sha256"] = file_sha256(item_path)
    manifest["artifact_sha256"] = conditionals.m5_conditional_artifact_sha256(manifest)
    m4.write_m4_json(
        manifest,
        artifact_dir / conditionals.M5_CONDITIONAL_ARTIFACT_MANIFEST_NAME,
    )

    with pytest.raises(RuntimeError, match="prompt_sha256 differs from sample plan"):
        conditionals.M5ConditionalArtifactStore(
            artifact_dir=artifact_dir,
            sample_plan=plan,
        )


def test_tampered_sample_plan_contents_with_stale_sha_rejects(
    tmp_path: Path,
) -> None:
    _manifest_path, _dataset_root, plan = _case(tmp_path)
    artifact_dir = tmp_path / "artifact"
    identity = plan["train_sample_identities"][0]
    _write_minimal_artifact(
        artifact_dir,
        plan,
        materialized_identities=(identity,),
    )
    tampered_plan = copy.deepcopy(plan)
    tampered_plan["samples"]["train"][0]["prompt_sha256"] = "2" * 64

    with pytest.raises(RuntimeError, match="sample plan SHA256"):
        conditionals.M5ConditionalArtifactStore(
            artifact_dir=artifact_dir,
            sample_plan=tampered_plan,
        )


def test_missing_model_checkpoint_sha256_rejects(tmp_path: Path) -> None:
    _manifest_path, _dataset_root, plan = _case(tmp_path)
    artifact_dir = tmp_path / "artifact"
    identity = plan["train_sample_identities"][0]
    manifest = _write_minimal_artifact(
        artifact_dir,
        plan,
        materialized_identities=(identity,),
    )
    manifest["encoder_provenance"].pop("model_checkpoint_sha256")
    manifest["artifact_sha256"] = conditionals.m5_conditional_artifact_sha256(manifest)
    m4.write_m4_json(
        manifest,
        artifact_dir / conditionals.M5_CONDITIONAL_ARTIFACT_MANIFEST_NAME,
    )

    with pytest.raises(RuntimeError, match="model_checkpoint_sha256"):
        conditionals.M5ConditionalArtifactStore(
            artifact_dir=artifact_dir,
            sample_plan=plan,
        )


def test_invalid_model_checkpoint_sha256_rejects(tmp_path: Path) -> None:
    _manifest_path, _dataset_root, plan = _case(tmp_path)
    artifact_dir = tmp_path / "artifact"
    identity = plan["train_sample_identities"][0]
    manifest = _write_minimal_artifact(
        artifact_dir,
        plan,
        materialized_identities=(identity,),
    )
    manifest["encoder_provenance"]["model_checkpoint_sha256"] = "not-a-sha"
    manifest["artifact_sha256"] = conditionals.m5_conditional_artifact_sha256(manifest)
    m4.write_m4_json(
        manifest,
        artifact_dir / conditionals.M5_CONDITIONAL_ARTIFACT_MANIFEST_NAME,
    )

    with pytest.raises(RuntimeError, match="model_checkpoint_sha256"):
        conditionals.M5ConditionalArtifactStore(
            artifact_dir=artifact_dir,
            sample_plan=plan,
        )


def test_context_body_exception_restores_state(tmp_path: Path) -> None:
    _manifest_path, _dataset_root, plan = _case(tmp_path)
    artifact_dir = tmp_path / "artifact"
    identity = plan["train_sample_identities"][0]
    second = plan["train_sample_identities"][1]
    _write_minimal_artifact(
        artifact_dir,
        plan,
        materialized_identities=(identity, second),
    )
    store = conditionals.M5ConditionalArtifactStore(
        artifact_dir=artifact_dir,
        sample_plan=plan,
    )

    with pytest.raises(ValueError, match="body failed"), store.acquire(identity):
        assert store.live_conditional_count == 1
        raise ValueError("body failed")
    assert store.live_conditional_count == 0

    with store.acquire(second):
        assert store.live_conditional_count == 1
    assert store.live_conditional_count == 0
    assert store.max_live_conditional_count == 1
    assert store.total_load_count == 2
