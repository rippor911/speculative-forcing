from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Optional

import torch

from speculative.adapters.runtime_state import ObjectStateSpec, TensorRegionSpec
from speculative.adapters.self_forcing_runtime import (
    RuntimeOperation,
    RuntimeStateSpecBundle,
    RuntimeWindowDescriptor,
    SelfForcingMCPRuntimeContext,
)
from speculative.adapters.wan_state_planner import (
    SUPPORTED_ATTENTION_MODE,
    WanStateMutationPlan,
    WanTouchedRangePlanner,
)
from speculative.types import (
    BlockRef,
    CommitRequest,
    ControlRequest,
    DraftCandidate,
    FallbackResult,
    ProposalBatch,
)


CacheKind = Literal["kv_cache", "crossattn_cache"]


def _is_strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


class SelfForcingWanMCPBackend:
    """Minimal Wan backend for the existing Self-Forcing MCP runtime.

    The backend borrows a `SelfForcingTrainingPipeline`, prompt conditional
    dictionary, and Wan touched-range planner. It does not own the runtime
    output buffer, commit bookkeeping, controller cursor, verifier, VAE, or
    runtime transactions.
    """

    def __init__(
        self,
        *,
        pipeline: Any,
        conditional_dict: Mapping[str, Any],
        planner: WanTouchedRangePlanner,
    ) -> None:
        if not isinstance(conditional_dict, Mapping):
            raise TypeError("conditional_dict must be a mapping.")
        if "prompt_embeds" not in conditional_dict:
            raise ValueError("conditional_dict must contain prompt_embeds.")
        if not isinstance(planner, WanTouchedRangePlanner):
            raise TypeError("planner must be a WanTouchedRangePlanner.")

        self._pipeline = pipeline
        self._conditional_dict = conditional_dict
        self._planner = planner

        self._validate_static_pipeline_config()
        self._migrate_model_freqs()

    @property
    def pipeline(self) -> Any:
        return self._pipeline

    @property
    def conditional_dict(self) -> Mapping[str, Any]:
        return self._conditional_dict

    @property
    def planner(self) -> WanTouchedRangePlanner:
        return self._planner

    def prepare_cross_attention(
        self,
        runtime_context: SelfForcingMCPRuntimeContext,
    ) -> None:
        self._validate_operation_prelude(runtime_context)
        block = self._prepare_block(runtime_context)
        source_noise = self._source_noise_for_block(runtime_context, block)
        timestep = self._timestep_for(source_noise)
        live_crossattn_cache = runtime_context.cross_attention_cache

        self._validate_crossattn_cache(live_crossattn_cache)
        self._validate_kv_cache(runtime_context.kv_cache)

        staging_cache = None
        try:
            self._pipeline._initialize_crossattn_cache(
                source_noise.shape[0],
                source_noise.dtype,
                source_noise.device,
            )
            staging_cache = self._pipeline.crossattn_cache
            self._pipeline.crossattn_cache = live_crossattn_cache
            if staging_cache is live_crossattn_cache:
                raise RuntimeError("prepare staging cross-attention cache must be independent.")

            with torch.no_grad():
                self._generator(
                    noisy_image_or_video=source_noise,
                    conditional_dict=self._conditional_dict,
                    timestep=timestep,
                    kv_cache=runtime_context.kv_cache,
                    crossattn_cache=staging_cache,
                    current_start=0,
                )

            self._publish_staging_cross_attention(
                staging_cache,
                live_crossattn_cache,
            )
        finally:
            self._pipeline.crossattn_cache = live_crossattn_cache

        self._validate_pipeline_cache_identity(runtime_context)

    def propose_anchor_and_drafts(
        self,
        request: ControlRequest,
        runtime_context: SelfForcingMCPRuntimeContext,
    ) -> ProposalBatch:
        if not isinstance(request, ControlRequest):
            raise TypeError("request must be a ControlRequest.")
        self._validate_operation_prelude(runtime_context)
        self._require_plan_block(runtime_context, request.anchor_block)

        anchor_block = request.anchor_block
        anchor_noise = self._source_noise_for_block(runtime_context, anchor_block)
        timestep = self._timestep_for(anchor_noise)
        current_start = runtime_context.current_start_for(anchor_block)
        future_noises, future_starts = self._future_chunks(
            request,
            runtime_context,
        )

        kwargs: dict[str, Any] = {}
        if future_noises:
            kwargs["mcp_future_noises"] = future_noises
            kwargs["mcp_future_start_frames"] = future_starts

        with torch.no_grad():
            result = self._generator(
                noisy_image_or_video=anchor_noise,
                conditional_dict=self._conditional_dict,
                timestep=timestep,
                kv_cache=runtime_context.kv_cache,
                crossattn_cache=runtime_context.cross_attention_cache,
                current_start=current_start,
                **kwargs,
            )

        flow_pred, denoised_pred, mcp_flow_preds = self._parse_generator_result(
            result,
            expect_mcp=bool(future_noises),
        )
        del flow_pred
        self._require_tensor_exact_compatible(
            denoised_pred,
            anchor_noise,
            "proposal anchor latent",
        )

        drafts = self._draft_candidates_from_mcp(
            runtime_context=runtime_context,
            anchor_block=anchor_block,
            future_noises=future_noises,
            mcp_flow_preds=mcp_flow_preds,
        )
        batch = ProposalBatch(
            anchor=CommitRequest(
                block=anchor_block,
                latent=denoised_pred,
                source="anchor",
            ),
            drafts=drafts,
        )
        self._validate_pipeline_cache_identity(runtime_context)
        return batch

    def generate_target_fallback(
        self,
        candidate: DraftCandidate,
        runtime_context: SelfForcingMCPRuntimeContext,
    ) -> FallbackResult:
        if not isinstance(candidate, DraftCandidate):
            raise TypeError("candidate must be a DraftCandidate.")
        self._validate_operation_prelude(runtime_context)
        self._require_plan_block(runtime_context, candidate.block)
        self._require_tensor_exact_compatible(
            candidate.source_noise,
            self._output_slice_for_block(runtime_context, candidate.block),
            "fallback source_noise",
        )

        timestep = self._timestep_for(candidate.source_noise)
        current_start = runtime_context.current_start_for(candidate.block)
        with torch.no_grad():
            result = self._generator(
                noisy_image_or_video=candidate.source_noise,
                conditional_dict=self._conditional_dict,
                timestep=timestep,
                kv_cache=runtime_context.kv_cache,
                crossattn_cache=runtime_context.cross_attention_cache,
                current_start=current_start,
            )

        flow_pred, denoised_pred, _ = self._parse_generator_result(
            result,
            expect_mcp=False,
        )
        del flow_pred
        self._require_tensor_exact_compatible(
            denoised_pred,
            self._output_slice_for_block(runtime_context, candidate.block),
            "fallback latent",
        )
        self._validate_pipeline_cache_identity(runtime_context)
        return FallbackResult(
            block=candidate.block,
            latent=denoised_pred,
            source_noise=candidate.source_noise,
        )

    def commit_context_block(
        self,
        request: CommitRequest,
        runtime_context: SelfForcingMCPRuntimeContext,
    ) -> None:
        if not isinstance(request, CommitRequest):
            raise TypeError("request must be a CommitRequest.")
        self._validate_operation_prelude(runtime_context)
        self._require_plan_block(runtime_context, request.block)
        if request.block.start_frame is None:
            raise RuntimeError("commit block start_frame is required.")
        expected_start = runtime_context.current_start_for(request.block)
        current_cache_start = self._current_cache_start(runtime_context.kv_cache)
        if current_cache_start != expected_start:
            raise RuntimeError(
                f"commit cache start {current_cache_start} does not match "
                f"block current_start {expected_start}."
            )

        self._pipeline._commit_context_block(
            latent=request.latent,
            start_frame=request.block.start_frame,
            conditional_dict=self._conditional_dict,
        )
        self._validate_cache_end(runtime_context, request.block)
        self._validate_pipeline_cache_identity(runtime_context)

    def temporary_state_specs(
        self,
        operation: RuntimeOperation,
        target: object,
        runtime_context: SelfForcingMCPRuntimeContext,
    ) -> RuntimeStateSpecBundle:
        self._validate_operation_prelude(runtime_context)
        if operation == "prepare":
            plan = self._planner.plan_prepare_scratch(
                self._prepare_block(runtime_context)
            )
            return self._state_specs_from_plan(plan, runtime_context)
        if operation == "proposal":
            if not isinstance(target, ControlRequest):
                raise TypeError("proposal target must be a ControlRequest.")
            plan = self._planner.plan_proposal(target)
            return self._state_specs_from_plan(plan, runtime_context)
        if operation == "fallback":
            if not isinstance(target, DraftCandidate):
                raise TypeError("fallback target must be a DraftCandidate.")
            plan = self._planner.plan_fallback(target)
            return self._state_specs_from_plan(plan, runtime_context)
        raise ValueError(f"Unsupported runtime operation {operation!r}.")

    def prepare_persistent_state_specs(
        self,
        runtime_context: SelfForcingMCPRuntimeContext,
    ) -> RuntimeStateSpecBundle:
        self._validate_operation_prelude(runtime_context)
        return self._state_specs_from_plan(
            self._planner.plan_prepare_cross_attention(),
            runtime_context,
        )

    def window_state_specs(
        self,
        window: RuntimeWindowDescriptor,
        runtime_context: SelfForcingMCPRuntimeContext,
    ) -> RuntimeStateSpecBundle:
        self._validate_operation_prelude(runtime_context)
        if not isinstance(window, RuntimeWindowDescriptor):
            raise TypeError("window must be a RuntimeWindowDescriptor.")
        return self._state_specs_from_plan(
            self._planner.plan_window(window),
            runtime_context,
        )

    def _state_specs_from_plan(
        self,
        plan: WanStateMutationPlan,
        runtime_context: SelfForcingMCPRuntimeContext,
    ) -> RuntimeStateSpecBundle:
        if not isinstance(plan, WanStateMutationPlan):
            raise TypeError("plan must be a WanStateMutationPlan.")

        tensor_regions = tuple(
            self._bind_tensor_range(descriptor, runtime_context)
            for descriptor in plan.backend_tensor_ranges
        )
        tensor_values = tuple(
            self._bind_tensor_value(name, runtime_context)
            for name in plan.backend_tensor_value_names
        )
        object_states = tuple(
            self._bind_python_value(name, runtime_context)
            for name in plan.backend_python_value_names
        )
        return RuntimeStateSpecBundle(
            tensor_regions=tensor_regions,
            tensor_values=tensor_values,
            object_states=object_states,
            capture_rng=plan.capture_rng,
            capture_cuda_rng=plan.capture_cuda_rng,
        )

    def _bind_tensor_range(
        self,
        descriptor: Any,
        runtime_context: SelfForcingMCPRuntimeContext,
    ) -> TensorRegionSpec:
        cache_kind, layer_index, field_name = self._parse_state_name(
            descriptor.state_name
        )
        if field_name not in ("k", "v"):
            raise ValueError(
                f"{descriptor.state_name!r} is not a tensor range state."
            )
        tensor = self._cache_tensor(
            cache_kind,
            layer_index,
            field_name,
            runtime_context,
        )
        return TensorRegionSpec(
            tensor=tensor,
            dim=descriptor.dimension,
            start=descriptor.start,
            end=descriptor.end,
            name=descriptor.state_name,
        )

    def _bind_tensor_value(
        self,
        state_name: str,
        runtime_context: SelfForcingMCPRuntimeContext,
    ) -> torch.Tensor:
        cache_kind, layer_index, field_name = self._parse_state_name(state_name)
        if cache_kind != "kv_cache" or field_name not in (
            "global_end_index",
            "local_end_index",
        ):
            raise ValueError(f"{state_name!r} is not a supported tensor value state.")
        tensor = self._cache_layer(cache_kind, layer_index, runtime_context)[field_name]
        self._require_scalar_tensor(tensor, state_name)
        return tensor

    def _bind_python_value(
        self,
        state_name: str,
        runtime_context: SelfForcingMCPRuntimeContext,
    ) -> ObjectStateSpec[bool]:
        cache_kind, layer_index, field_name = self._parse_state_name(state_name)
        if cache_kind != "crossattn_cache" or field_name != "is_init":
            raise ValueError(f"{state_name!r} is not a supported Python value state.")
        layer = self._cache_layer(cache_kind, layer_index, runtime_context)
        self._require_bool(layer.get("is_init"), state_name)
        return self._is_init_state_spec(layer, state_name)

    def _is_init_state_spec(
        self,
        layer: dict[str, Any],
        name: str,
    ) -> ObjectStateSpec[bool]:
        def getter(layer: dict[str, Any] = layer, name: str = name) -> bool:
            value = layer.get("is_init")
            self._require_bool(value, name)
            return value

        def setter(
            value: Any,
            layer: dict[str, Any] = layer,
            name: str = name,
        ) -> None:
            self._require_bool(value, name)
            layer["is_init"] = value

        return ObjectStateSpec(
            getter=getter,
            setter=setter,
            copy_fn=bool,
            name=name,
        )

    def _parse_state_name(self, state_name: str) -> tuple[CacheKind, int, str]:
        parts = state_name.split(".")
        if len(parts) != 3:
            raise ValueError(f"Unsupported Wan state name {state_name!r}.")
        cache_name, layer_name, field_name = parts
        if cache_name == "kv_cache":
            allowed = {"k", "v", "global_end_index", "local_end_index"}
        elif cache_name == "crossattn_cache":
            allowed = {"k", "v", "is_init"}
        else:
            raise ValueError(f"Unsupported Wan state name {state_name!r}.")

        prefix = "layer_"
        layer_digits = layer_name[len(prefix):] if layer_name.startswith(prefix) else ""
        if len(layer_digits) != 3 or not layer_digits.isdigit():
            raise ValueError(f"Unsupported Wan state name {state_name!r}.")
        if field_name not in allowed:
            raise ValueError(f"Unsupported Wan state name {state_name!r}.")
        return cache_name, int(layer_digits), field_name  # type: ignore[return-value]

    def _cache_tensor(
        self,
        cache_kind: CacheKind,
        layer_index: int,
        field_name: str,
        runtime_context: SelfForcingMCPRuntimeContext,
    ) -> torch.Tensor:
        value = self._cache_layer(cache_kind, layer_index, runtime_context)[field_name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"{cache_kind}.layer_{layer_index:03d}.{field_name} must be "
                f"a torch.Tensor, got {type(value).__name__}."
            )
        return value

    def _cache_layer(
        self,
        cache_kind: CacheKind,
        layer_index: int,
        runtime_context: SelfForcingMCPRuntimeContext,
    ) -> dict[str, Any]:
        cache = (
            runtime_context.kv_cache
            if cache_kind == "kv_cache"
            else runtime_context.cross_attention_cache
        )
        if not isinstance(cache, list):
            raise TypeError(f"{cache_kind} must be a list of layer dicts.")
        if layer_index < 0 or layer_index >= len(cache):
            raise IndexError(
                f"{cache_kind} layer {layer_index} is outside "
                f"{len(cache)} layers."
            )
        layer = cache[layer_index]
        if not isinstance(layer, dict):
            raise TypeError(f"{cache_kind} layer {layer_index} must be a dict.")
        return layer

    def _validate_operation_prelude(
        self,
        runtime_context: SelfForcingMCPRuntimeContext,
    ) -> None:
        self._assert_freqs_on_model_device()
        self._validate_pipeline_cache_identity(runtime_context)
        self._validate_runtime_context(runtime_context)
        self._validate_kv_cache(runtime_context.kv_cache)
        self._validate_crossattn_cache(runtime_context.cross_attention_cache)

    def _validate_runtime_context(
        self,
        runtime_context: SelfForcingMCPRuntimeContext,
    ) -> None:
        config = runtime_context.config
        if tuple(config.anchor_denoising_steps) != (1000,):
            raise ValueError("Wan backend supports only denoising schedule (1000,).")
        if config.mcp_depth not in (1, 2, 3):
            raise ValueError("Wan backend supports MCP depth 1, 2, or 3.")
        if config.frame_seq_length != self._planner.frame_seq_length:
            raise ValueError("runtime frame_seq_length must match the Wan planner.")
        if config.num_frame_per_block != self._planner.num_frame_per_block:
            raise ValueError("runtime num_frame_per_block must match the Wan planner.")
        if config.mcp_depth != self._planner.mcp_depth:
            raise ValueError("runtime mcp_depth must match the Wan planner.")
        if config.attention_mode != SUPPORTED_ATTENTION_MODE:
            raise ValueError(
                f"Wan backend supports only {SUPPORTED_ATTENTION_MODE!r}."
            )
        self._validate_latent_layout(runtime_context.source_noise, "source_noise")
        self._validate_latent_layout(runtime_context.output, "output")
        source_device_type = runtime_context.source_noise.device.type
        if source_device_type not in ("cpu", "cuda"):
            raise ValueError(
                f"source_noise.device.type must be 'cpu' or 'cuda', got "
                f"{source_device_type!r}."
            )
        if self._planner.runtime_device_type != source_device_type:
            raise ValueError(
                "planner.runtime_device_type must match source_noise.device.type: "
                f"{self._planner.runtime_device_type!r} != {source_device_type!r}."
            )
        if runtime_context.source_noise.shape[1] != config.all_num_frames:
            raise ValueError("source_noise frame count must match runtime config.")
        if runtime_context.output.shape[1] < config.all_num_frames:
            raise ValueError("output frame count must cover runtime config.")
        if runtime_context.output.dtype != runtime_context.source_noise.dtype:
            raise ValueError("output and source_noise dtype must match.")
        if runtime_context.output.device != runtime_context.source_noise.device:
            raise ValueError("output and source_noise device must match.")
        if runtime_context.output.layout != runtime_context.source_noise.layout:
            raise ValueError("output and source_noise layout must match.")

        prompt_embeds = self._conditional_dict["prompt_embeds"]
        if isinstance(prompt_embeds, torch.Tensor):
            if prompt_embeds.shape[0] != 1:
                raise ValueError("Wan backend does not support CFG-expanded prompt batches.")
        elif isinstance(prompt_embeds, Sequence):
            if len(prompt_embeds) != 1:
                raise ValueError("Wan backend supports exactly one prompt embedding.")

    def _validate_static_pipeline_config(self) -> None:
        self._require_pipeline_method("_initialize_crossattn_cache")
        self._require_pipeline_method("_mcp_future_chunks")
        self._require_pipeline_method("_commit_context_block")

        if tuple(getattr(self._pipeline, "denoising_step_list", ())) != (1000,):
            raise ValueError("Wan backend supports only denoising_step_list == [1000].")
        if bool(getattr(self._pipeline, "i2v", False)):
            raise ValueError("Wan backend supports T2V only.")
        if bool(getattr(self._pipeline, "independent_first_frame", False)):
            raise ValueError("Wan backend does not support independent_first_frame.")

        if self._planner.local_attn_size != -1:
            raise ValueError("Wan backend supports only local_attn_size == -1.")
        if self._planner.sink_size != 0:
            raise ValueError("Wan backend supports only sink_size == 0.")
        if self._planner.mcp_depth not in (1, 2, 3):
            raise ValueError("Wan backend supports MCP depth 1, 2, or 3.")
        self._require_pipeline_equal("frame_seq_length", self._planner.frame_seq_length)
        self._require_pipeline_equal(
            "num_frame_per_block",
            self._planner.num_frame_per_block,
        )
        self._require_pipeline_equal(
            "num_transformer_blocks",
            self._planner.cache_layout.num_layers,
        )
        self._require_pipeline_equal(
            "kv_cache_size",
            self._planner.cache_layout.cache_capacity,
        )
        self._require_pipeline_equal("mcp_num_modules", self._planner.mcp_depth)
        self._require_pipeline_equal("mcp_accel_depths", self._planner.mcp_depth)
        self._require_pipeline_equal("context_noise", 0)

        generator = self._generator
        model = self._model
        if getattr(model, "model_type", None) != "t2v":
            raise ValueError("Wan backend supports T2V only.")
        if getattr(model, "local_attn_size", -1) != -1:
            raise ValueError("Wan backend supports only local_attn_size == -1.")
        if self._model_sink_size(model) != 0:
            raise ValueError("Wan backend supports only sink_size == 0.")
        if not hasattr(generator, "uniform_timestep"):
            raise ValueError("generator.uniform_timestep is required and must be False.")
        if generator.uniform_timestep is not False:
            raise ValueError(
                "generator.uniform_timestep must be False for the causal Wan MCP wrapper."
            )

    def _require_pipeline_method(self, name: str) -> None:
        if not callable(getattr(self._pipeline, name, None)):
            raise TypeError(f"pipeline must provide callable {name}.")

    def _require_pipeline_equal(self, field_name: str, expected: object) -> None:
        if not hasattr(self._pipeline, field_name):
            raise ValueError(f"pipeline.{field_name} is required.")
        actual = getattr(self._pipeline, field_name)
        if actual != expected:
            raise ValueError(
                f"pipeline.{field_name} must be {expected!r}, got {actual!r}."
            )

    def _validate_pipeline_cache_identity(
        self,
        runtime_context: SelfForcingMCPRuntimeContext,
    ) -> None:
        if self._pipeline.kv_cache1 is not runtime_context.kv_cache:
            raise RuntimeError("pipeline.kv_cache1 must be the live runtime KV cache.")
        if self._pipeline.crossattn_cache is not runtime_context.cross_attention_cache:
            raise RuntimeError(
                "pipeline.crossattn_cache must be the live runtime cross-attention cache."
            )

    def _validate_kv_cache(self, cache: Any) -> None:
        if not isinstance(cache, list):
            raise TypeError("kv_cache must be a list of layer dicts.")
        if len(cache) != self._planner.cache_layout.num_layers:
            raise ValueError("kv_cache layer count must match planner layout.")
        for layer_index, layer in enumerate(cache):
            if not isinstance(layer, dict):
                raise TypeError(f"kv_cache layer {layer_index} must be a dict.")
            for field_name in ("k", "v", "global_end_index", "local_end_index"):
                if field_name not in layer:
                    raise KeyError(
                        f"kv_cache.layer_{layer_index:03d}.{field_name} is missing."
                    )
            for field_name in ("k", "v"):
                tensor = layer[field_name]
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(
                        f"kv_cache.layer_{layer_index:03d}.{field_name} must be a tensor."
                    )
                if tensor.ndim < 2:
                    raise ValueError(
                        f"kv_cache.layer_{layer_index:03d}.{field_name} must have token dim 1."
                    )
                if tensor.shape[0] != 1:
                    raise ValueError("Wan backend supports batch_size == 1 only.")
            for field_name in ("global_end_index", "local_end_index"):
                self._require_scalar_tensor(
                    layer[field_name],
                    f"kv_cache.layer_{layer_index:03d}.{field_name}",
                )

    def _validate_crossattn_cache(self, cache: Any) -> None:
        if not isinstance(cache, list):
            raise TypeError("crossattn_cache must be a list of layer dicts.")
        if len(cache) != self._planner.cache_layout.num_layers:
            raise ValueError("crossattn_cache layer count must match planner layout.")
        for layer_index, layer in enumerate(cache):
            if not isinstance(layer, dict):
                raise TypeError(f"crossattn_cache layer {layer_index} must be a dict.")
            for field_name in ("k", "v", "is_init"):
                if field_name not in layer:
                    raise KeyError(
                        f"crossattn_cache.layer_{layer_index:03d}.{field_name} is missing."
                    )
            for field_name in ("k", "v"):
                tensor = layer[field_name]
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(
                        f"crossattn_cache.layer_{layer_index:03d}.{field_name} must be a tensor."
                    )
                if tensor.ndim < 2:
                    raise ValueError(
                        f"crossattn_cache.layer_{layer_index:03d}.{field_name} "
                        "must have token dim 1."
                    )
                if tensor.shape[0] != 1:
                    raise ValueError("Wan backend supports batch_size == 1 only.")
            self._require_bool(
                layer["is_init"],
                f"crossattn_cache.layer_{layer_index:03d}.is_init",
            )

    def _publish_staging_cross_attention(
        self,
        staging_cache: Any,
        live_cache: Any,
    ) -> None:
        if not isinstance(staging_cache, list):
            raise TypeError("staging crossattn_cache must be a list.")
        if len(staging_cache) != len(live_cache):
            raise RuntimeError("staging and live cross-attention layer counts differ.")

        live_pairs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        live_layers: list[dict[str, Any]] = []
        for layer_index, (staging_layer, live_layer) in enumerate(zip(staging_cache, live_cache)):
            if not isinstance(staging_layer, dict):
                raise TypeError(f"staging crossattn layer {layer_index} must be a dict.")
            if not isinstance(live_layer, dict):
                raise TypeError(f"live crossattn layer {layer_index} must be a dict.")
            for field_name in ("k", "v", "is_init"):
                if field_name not in staging_layer:
                    raise KeyError(
                        f"staging crossattn layer {layer_index} missing {field_name}."
                    )
                if field_name not in live_layer:
                    raise KeyError(
                        f"live crossattn layer {layer_index} missing {field_name}."
                    )
            if staging_layer["is_init"] is not True:
                raise RuntimeError(
                    f"staging crossattn layer {layer_index} was not initialized."
                )
            if not isinstance(live_layer["is_init"], bool):
                raise TypeError(
                    f"live crossattn layer {layer_index}.is_init must be a bool."
                )
            staging_k = staging_layer["k"]
            staging_v = staging_layer["v"]
            live_k = live_layer["k"]
            live_v = live_layer["v"]
            for field_name, staging_tensor, live_tensor in (
                ("k", staging_k, live_k),
                ("v", staging_v, live_v),
            ):
                if not isinstance(staging_tensor, torch.Tensor):
                    raise TypeError(
                        f"staging crossattn layer {layer_index}.{field_name} "
                        "must be a tensor."
                    )
                if not isinstance(live_tensor, torch.Tensor):
                    raise TypeError(
                        f"live crossattn layer {layer_index}.{field_name} must be a tensor."
                    )
                self._require_tensor_exact_compatible(
                    staging_tensor,
                    live_tensor,
                    f"staging crossattn layer {layer_index}.{field_name}",
                    include_stride=True,
                )
            live_pairs.append((live_k, staging_k, live_v, staging_v))
            live_layers.append(live_layer)

        with torch.no_grad():
            for live_k, staging_k, live_v, staging_v in live_pairs:
                live_k.copy_(staging_k)
                live_v.copy_(staging_v)
            for live_layer in live_layers:
                live_layer["is_init"] = True

    def _future_chunks(
        self,
        request: ControlRequest,
        runtime_context: SelfForcingMCPRuntimeContext,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[int, ...]]:
        depth_limit = min(request.max_depth, runtime_context.config.mcp_depth)
        if depth_limit <= 0:
            return (), ()
        all_num_frames = tuple(
            self._block_num_frames(block)
            for block in runtime_context.rollout_plan.blocks
        )
        noises, starts = self._pipeline._mcp_future_chunks(
            request.anchor_block.index,
            tuple(runtime_context.rollout_plan.block_starts),
            all_num_frames,
            runtime_context.source_noise,
            0,
        )
        if len(noises) < depth_limit or len(starts) < depth_limit:
            raise RuntimeError("_mcp_future_chunks returned fewer entries than requested.")

        valid_noises: list[torch.Tensor] = []
        valid_starts: list[int] = []
        for depth, (noise, start) in enumerate(
            zip(noises[:depth_limit], starts[:depth_limit]),
            start=1,
        ):
            future_block_index = request.anchor_block.index + depth
            if future_block_index >= len(runtime_context.rollout_plan.blocks):
                if noise is not None or start is not None:
                    raise RuntimeError(
                        "_mcp_future_chunks returned data beyond the rollout plan."
                    )
                break
            expected_start = runtime_context.rollout_plan.blocks[
                future_block_index
            ].start_frame
            if expected_start is None:
                raise RuntimeError("future rollout block start_frame is required.")
            if noise is None:
                if start is not None:
                    raise RuntimeError(
                        "_mcp_future_chunks returned a start frame with None noise."
                    )
                break
            if start is None:
                raise RuntimeError("_mcp_future_chunks returned noise without a start frame.")
            if not isinstance(noise, torch.Tensor):
                raise TypeError("MCP future noise must be a torch.Tensor or None.")
            if not _is_strict_int(start):
                raise TypeError("MCP future start frame must be a strict int.")
            if start != expected_start:
                raise RuntimeError(
                    f"MCP future start frame {start} does not match rollout block "
                    f"{future_block_index} start_frame {expected_start}."
                )
            valid_noises.append(noise)
            valid_starts.append(start)
        return tuple(valid_noises), tuple(valid_starts)

    def _draft_candidates_from_mcp(
        self,
        *,
        runtime_context: SelfForcingMCPRuntimeContext,
        anchor_block: BlockRef,
        future_noises: tuple[torch.Tensor, ...],
        mcp_flow_preds: Optional[Sequence[torch.Tensor]],
    ) -> tuple[DraftCandidate, ...]:
        if not future_noises:
            if mcp_flow_preds not in (None, ()):
                raise RuntimeError("generator returned MCP flows without requested futures.")
            return ()
        if not isinstance(mcp_flow_preds, (list, tuple)):
            raise RuntimeError("generator must return a list/tuple of MCP flow predictions.")
        if len(mcp_flow_preds) != len(future_noises):
            raise RuntimeError("MCP flow count must match valid future noise count.")

        drafts: list[DraftCandidate] = []
        for depth, (source_noise, flow_pred) in enumerate(
            zip(future_noises, mcp_flow_preds),
            start=1,
        ):
            if not isinstance(flow_pred, torch.Tensor):
                raise TypeError("MCP flow prediction must be a torch.Tensor.")
            self._require_tensor_exact_compatible(
                flow_pred,
                source_noise,
                f"MCP flow prediction depth {depth}",
            )
            block_index = anchor_block.index + depth
            if block_index >= len(runtime_context.rollout_plan.blocks):
                raise RuntimeError("MCP flow prediction targets a block outside the rollout plan.")
            block = runtime_context.rollout_plan.blocks[block_index]
            self._require_tensor_exact_compatible(
                source_noise,
                self._output_slice_for_block(runtime_context, block),
                f"MCP source noise depth {depth}",
            )
            drafts.append(
                DraftCandidate(
                    block=block,
                    depth=depth,
                    latent=source_noise - flow_pred,
                    source_noise=source_noise,
                )
            )
        return tuple(drafts)

    def _parse_generator_result(
        self,
        result: object,
        *,
        expect_mcp: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[Sequence[torch.Tensor]]]:
        if not isinstance(result, tuple):
            raise RuntimeError("generator must return a tuple.")
        expected_len = 3 if expect_mcp else 2
        if len(result) != expected_len:
            raise RuntimeError(
                f"generator returned {len(result)} values; expected {expected_len}."
            )
        flow_pred = result[0]
        denoised_pred = result[1]
        if not isinstance(flow_pred, torch.Tensor):
            raise TypeError("generator flow prediction must be a torch.Tensor.")
        if not isinstance(denoised_pred, torch.Tensor):
            raise TypeError("generator denoised prediction must be a torch.Tensor.")
        mcp_flow_preds = result[2] if expect_mcp else None
        return flow_pred, denoised_pred, mcp_flow_preds  # type: ignore[return-value]

    def _prepare_block(
        self,
        runtime_context: SelfForcingMCPRuntimeContext,
    ) -> BlockRef:
        if not runtime_context.rollout_plan.blocks:
            raise RuntimeError("rollout plan has no blocks.")
        block = runtime_context.rollout_plan.blocks[0]
        if block.index != 0:
            raise RuntimeError("prepare scratch block must have index 0.")
        if block.start_frame != 0:
            raise RuntimeError("prepare scratch block must start at frame 0.")
        return block

    def _source_noise_for_block(
        self,
        runtime_context: SelfForcingMCPRuntimeContext,
        block: BlockRef,
    ) -> torch.Tensor:
        if block.start_frame is None or block.num_frames is None:
            raise RuntimeError("BlockRef start_frame and num_frames are required.")
        source_noise = runtime_context.source_noise
        if not isinstance(source_noise, torch.Tensor):
            raise TypeError("runtime source_noise must be a torch.Tensor.")
        start = block.start_frame
        end = start + block.num_frames
        sliced = source_noise[:, start:end]
        self._require_tensor_exact_compatible(
            sliced,
            self._output_slice_for_block(runtime_context, block),
            "source noise block slice",
        )
        return sliced

    def _output_slice_for_block(
        self,
        runtime_context: SelfForcingMCPRuntimeContext,
        block: BlockRef,
    ) -> torch.Tensor:
        if block.start_frame is None or block.num_frames is None:
            raise RuntimeError("BlockRef start_frame and num_frames are required.")
        return runtime_context.output[:, block.start_frame:block.start_frame + block.num_frames]

    def _timestep_for(self, noise: torch.Tensor) -> torch.Tensor:
        if not isinstance(noise, torch.Tensor):
            raise TypeError("noise must be a torch.Tensor.")
        if noise.ndim != 5:
            raise ValueError("Wan backend expects latent layout [B, F, C, H, W].")
        return torch.full(
            (noise.shape[0], noise.shape[1]),
            1000,
            device=noise.device,
            dtype=torch.int64,
        )

    def _current_cache_start(self, kv_cache: Any) -> int:
        self._validate_kv_cache(kv_cache)
        starts: list[int] = []
        for layer_index, layer in enumerate(kv_cache):
            global_index = int(layer["global_end_index"].item())
            local_index = int(layer["local_end_index"].item())
            if global_index != local_index:
                raise RuntimeError(
                    f"kv_cache layer {layer_index} global/local indices differ."
                )
            starts.append(global_index)
        first = starts[0]
        if any(start != first for start in starts):
            raise RuntimeError(f"kv_cache start differs across layers: {starts}.")
        return first

    def _validate_cache_end(
        self,
        runtime_context: SelfForcingMCPRuntimeContext,
        block: BlockRef,
    ) -> None:
        expected_end = self._expected_token_end(runtime_context, block)
        for layer_index, layer in enumerate(runtime_context.kv_cache):
            global_index = int(layer["global_end_index"].item())
            local_index = int(layer["local_end_index"].item())
            if global_index != expected_end:
                raise RuntimeError(
                    f"kv_cache layer {layer_index} global_end_index "
                    f"{global_index} != expected {expected_end}."
                )
            if local_index != expected_end:
                raise RuntimeError(
                    f"kv_cache layer {layer_index} local_end_index "
                    f"{local_index} != expected {expected_end}."
                )

    def _expected_token_end(
        self,
        runtime_context: SelfForcingMCPRuntimeContext,
        block: BlockRef,
    ) -> int:
        if block.start_frame is None or block.num_frames is None:
            raise RuntimeError("BlockRef start_frame and num_frames are required.")
        return (block.start_frame + block.num_frames) * runtime_context.config.frame_seq_length

    def _block_num_frames(self, block: BlockRef) -> int:
        if block.num_frames is None:
            raise RuntimeError("BlockRef.num_frames is required.")
        return block.num_frames

    def _require_plan_block(
        self,
        runtime_context: SelfForcingMCPRuntimeContext,
        block: BlockRef,
    ) -> None:
        if not runtime_context.rollout_plan.contains_block(block):
            raise RuntimeError(f"BlockRef {block!r} is outside the rollout plan.")

    def _validate_latent_layout(self, tensor: Any, name: str) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if tensor.ndim != 5:
            raise ValueError(f"{name} must use latent layout [B, F, C, H, W].")
        if tensor.shape[0] != 1:
            raise ValueError("Wan backend supports batch_size == 1 only.")

    def _require_tensor_exact_compatible(
        self,
        actual: Any,
        expected: torch.Tensor,
        name: str,
        *,
        include_stride: bool = False,
    ) -> None:
        if not isinstance(actual, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if actual.shape != expected.shape:
            raise RuntimeError(
                f"{name} shape {tuple(actual.shape)} does not match "
                f"{tuple(expected.shape)}."
            )
        if actual.dtype != expected.dtype:
            raise RuntimeError(f"{name} dtype {actual.dtype} does not match {expected.dtype}.")
        if actual.device != expected.device:
            raise RuntimeError(f"{name} device {actual.device} does not match {expected.device}.")
        if actual.layout != expected.layout:
            raise RuntimeError(f"{name} layout {actual.layout} does not match {expected.layout}.")
        if include_stride and tuple(actual.stride()) != tuple(expected.stride()):
            raise RuntimeError(f"{name} stride does not match live tensor stride.")

    def _require_scalar_tensor(self, value: Any, name: str) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if value.numel() != 1:
            raise ValueError(f"{name} must be a scalar tensor.")

    def _require_bool(self, value: Any, name: str) -> None:
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a Python bool.")

    @property
    def _generator(self) -> Any:
        generator = getattr(self._pipeline, "generator", None)
        if generator is None:
            raise TypeError("pipeline.generator is required.")
        return generator

    @property
    def _model(self) -> Any:
        model = getattr(self._generator, "model", None)
        if model is None:
            raise TypeError("pipeline.generator.model is required.")
        return model

    def _migrate_model_freqs(self) -> None:
        model = self._model
        freqs = getattr(model, "freqs", None)
        patch_embedding = getattr(model, "patch_embedding", None)
        weight = getattr(patch_embedding, "weight", None)
        if not isinstance(freqs, torch.Tensor):
            raise TypeError("generator.model.freqs must be a torch.Tensor.")
        if not isinstance(weight, torch.Tensor):
            raise TypeError("generator.model.patch_embedding.weight must be a torch.Tensor.")
        if freqs.device != weight.device:
            model.freqs = freqs.to(device=weight.device)

    def _assert_freqs_on_model_device(self) -> None:
        model = self._model
        freqs = getattr(model, "freqs", None)
        weight = getattr(getattr(model, "patch_embedding", None), "weight", None)
        if not isinstance(freqs, torch.Tensor):
            raise TypeError("generator.model.freqs must be a torch.Tensor.")
        if not isinstance(weight, torch.Tensor):
            raise TypeError("generator.model.patch_embedding.weight must be a torch.Tensor.")
        if freqs.device != weight.device:
            raise RuntimeError(
                "generator.model.freqs device must match "
                "generator.model.patch_embedding.weight.device before forward."
            )

    def _model_sink_size(self, model: Any) -> int:
        if hasattr(model, "sink_size"):
            return int(model.sink_size)
        blocks = getattr(model, "blocks", None)
        if blocks:
            first_block = blocks[0]
            self_attn = getattr(first_block, "self_attn", None)
            if hasattr(self_attn, "sink_size"):
                return int(self_attn.sink_size)
        return int(self._planner.sink_size)


__all__ = ["SelfForcingWanMCPBackend"]
