from __future__ import annotations

from collections.abc import Mapping as MappingABC
from typing import Any, Callable, Mapping

from speculative.evaluation import CompositeCandidateEvaluator, IdentityCandidateDecoder
from speculative.interfaces import Evaluator, Policy
from speculative.policies.fixed_threshold import FixedThresholdPolicy
from speculative.policies.scripted import ScriptedPolicy
from speculative.scoring import (
    MeanScoreAggregator,
    MinScoreAggregator,
    ScriptedCandidateScorer,
)


PolicyFactory = Callable[..., Policy]
ConfigFactory = Callable[[Mapping[str, Any], str], object]

CANONICAL_SPECULATIVE_CONFIG_FIELDS = frozenset(
    {
        "controller",
        "proposal",
        "evaluator",
        "acceptance",
        "fallback",
        "verification",
        "trace",
    }
)


def _reject_unknown_kwargs(policy_name: str, kwargs: dict[str, Any], allowed: set[str]) -> None:
    unknown = set(kwargs) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown argument(s) for {policy_name}: {names}.")


def _always_accept_factory(**kwargs: Any) -> Policy:
    _reject_unknown_kwargs("always_accept", kwargs, set())
    return ScriptedPolicy.always_accept()


def _always_reject_factory(**kwargs: Any) -> Policy:
    _reject_unknown_kwargs("always_reject", kwargs, set())
    return ScriptedPolicy.always_reject()


def _reject_at_depth_factory(**kwargs: Any) -> Policy:
    _reject_unknown_kwargs("reject_at_depth", kwargs, {"reject_depth"})
    reject_depth = kwargs.get("reject_depth")
    if reject_depth is None:
        raise ValueError("reject_at_depth requires reject_depth.")
    if not isinstance(reject_depth, int) or isinstance(reject_depth, bool):
        raise ValueError("reject_at_depth reject_depth must be an integer.")
    return ScriptedPolicy.reject_at_depth(reject_depth)


def _fixed_threshold_factory(**kwargs: Any) -> Policy:
    _reject_unknown_kwargs("fixed_threshold", kwargs, {"threshold", "metadata"})
    if "threshold" not in kwargs or kwargs["threshold"] is None:
        raise ValueError("fixed_threshold requires threshold.")
    return FixedThresholdPolicy(
        threshold=kwargs["threshold"],
        metadata=kwargs.get("metadata", {}),
    )


POLICY_FACTORIES: dict[str, PolicyFactory] = {
    "always_accept": _always_accept_factory,
    "always_reject": _always_reject_factory,
    "fixed_threshold": _fixed_threshold_factory,
    "reject_at_depth": _reject_at_depth_factory,
}

POLICY_FACTORY_MAP = POLICY_FACTORIES


def _require_mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, MappingABC):
        raise ValueError(f"{path} must be a mapping.")
    return value


def _component_type(config: Mapping[str, Any], path: str) -> str:
    if "type" not in config or config["type"] is None:
        raise ValueError(f"{path}.type must be set.")
    component_type = config["type"]
    if not isinstance(component_type, str) or not component_type:
        raise ValueError(f"{path}.type must be a non-empty string.")
    return component_type


def _reject_unknown_config_keys(
    path: str,
    config: Mapping[str, Any],
    allowed: set[str],
) -> None:
    unknown = set(config) - allowed
    if unknown:
        names = ", ".join(sorted(str(name) for name in unknown))
        raise ValueError(f"{path} has unknown field(s): {names}.")


def _create_from_config(
    config: object,
    *,
    path: str,
    factories: Mapping[str, ConfigFactory],
) -> object:
    mapping = _require_mapping(config, path)
    component_type = _component_type(mapping, path)
    try:
        factory = factories[component_type]
    except KeyError as exc:
        known = ", ".join(sorted(factories))
        raise ValueError(
            f"{path}.type {component_type!r} is not supported; known: {known}."
        ) from exc
    return factory(mapping, path)


def _identity_decoder_from_config(config: Mapping[str, Any], path: str) -> IdentityCandidateDecoder:
    _reject_unknown_config_keys(path, config, {"type", "metadata"})
    return IdentityCandidateDecoder(metadata=config.get("metadata", {}))


def _scripted_scorer_from_config(config: Mapping[str, Any], path: str) -> ScriptedCandidateScorer:
    _reject_unknown_config_keys(
        path,
        config,
        {"type", "scores_by_depth", "scorer_name", "metadata"},
    )
    if "scores_by_depth" not in config or config["scores_by_depth"] is None:
        raise ValueError(f"{path}.scores_by_depth must be set.")
    return ScriptedCandidateScorer(
        scores_by_depth=config["scores_by_depth"],
        scorer_name=config.get("scorer_name", "scripted"),
        metadata=config.get("metadata", {}),
    )


def _min_aggregator_from_config(config: Mapping[str, Any], path: str) -> MinScoreAggregator:
    _reject_unknown_config_keys(path, config, {"type"})
    return MinScoreAggregator()


def _mean_aggregator_from_config(config: Mapping[str, Any], path: str) -> MeanScoreAggregator:
    _reject_unknown_config_keys(path, config, {"type"})
    return MeanScoreAggregator()


def _fixed_threshold_from_config(config: Mapping[str, Any], path: str) -> FixedThresholdPolicy:
    _reject_unknown_config_keys(path, config, {"type", "threshold", "metadata", "depth_thresholds"})
    if config.get("depth_thresholds") is not None:
        raise ValueError(f"{path}.depth_thresholds is not supported in Milestone F1.")
    if "threshold" not in config or config["threshold"] is None:
        raise ValueError(f"{path}.threshold must be set.")
    return FixedThresholdPolicy(
        threshold=config["threshold"],
        metadata=config.get("metadata", {}),
    )


def _always_accept_from_config(config: Mapping[str, Any], path: str) -> ScriptedPolicy:
    _reject_unknown_config_keys(path, config, {"type"})
    return ScriptedPolicy.always_accept()


def _always_reject_from_config(config: Mapping[str, Any], path: str) -> ScriptedPolicy:
    _reject_unknown_config_keys(path, config, {"type"})
    return ScriptedPolicy.always_reject()


def _reject_at_depth_from_config(config: Mapping[str, Any], path: str) -> ScriptedPolicy:
    _reject_unknown_config_keys(path, config, {"type", "reject_depth"})
    if "reject_depth" not in config or config["reject_depth"] is None:
        raise ValueError(f"{path}.reject_depth must be set.")
    return ScriptedPolicy.reject_at_depth(config["reject_depth"])


DECODER_FACTORIES: dict[str, ConfigFactory] = {
    "fake": _identity_decoder_from_config,
    "identity": _identity_decoder_from_config,
}

SCORER_FACTORIES: dict[str, ConfigFactory] = {
    "fake": _scripted_scorer_from_config,
    "scripted": _scripted_scorer_from_config,
}

AGGREGATOR_FACTORIES: dict[str, ConfigFactory] = {
    "mean": _mean_aggregator_from_config,
    "mean_frame": _mean_aggregator_from_config,
    "min": _min_aggregator_from_config,
    "min_frame": _min_aggregator_from_config,
}

ACCEPTANCE_FACTORIES: dict[str, ConfigFactory] = {
    "always_accept": _always_accept_from_config,
    "always_reject": _always_reject_from_config,
    "fixed_threshold": _fixed_threshold_from_config,
    "reject_at_depth": _reject_at_depth_from_config,
}


def create_policy(name: str, **kwargs: Any) -> Policy:
    """Create a policy from the explicit factory map."""

    try:
        factory = POLICY_FACTORIES[name]
    except KeyError as exc:
        known = ", ".join(sorted(POLICY_FACTORIES))
        raise ValueError(f"Unknown speculative policy {name!r}; known: {known}.") from exc
    return factory(**kwargs)


def create_decoder_from_config(
    config: object,
    *,
    path: str = "speculative.evaluator.decoder",
) -> object:
    return _create_from_config(config, path=path, factories=DECODER_FACTORIES)


def create_scorer_from_config(
    config: object,
    *,
    path: str = "speculative.evaluator.scorer",
) -> object:
    return _create_from_config(config, path=path, factories=SCORER_FACTORIES)


def create_aggregator_from_config(
    config: object,
    *,
    path: str = "speculative.evaluator.aggregator",
) -> object:
    return _create_from_config(config, path=path, factories=AGGREGATOR_FACTORIES)


def create_acceptance_policy_from_config(
    config: object,
    *,
    path: str = "speculative.acceptance",
) -> Policy:
    return _create_from_config(config, path=path, factories=ACCEPTANCE_FACTORIES)  # type: ignore[return-value]


def create_evaluator_from_config(
    config: object,
    *,
    path: str = "speculative.evaluator",
) -> Evaluator:
    mapping = _require_mapping(config, path)
    _reject_unknown_config_keys(path, mapping, {"decoder", "scorer", "aggregator"})
    for field_name in ("decoder", "scorer", "aggregator"):
        if field_name not in mapping or mapping[field_name] is None:
            raise ValueError(f"{path}.{field_name} must be set.")
    return CompositeCandidateEvaluator(
        decoder=create_decoder_from_config(
            mapping["decoder"],
            path=f"{path}.decoder",
        ),
        scorer=create_scorer_from_config(
            mapping["scorer"],
            path=f"{path}.scorer",
        ),
        aggregator=create_aggregator_from_config(
            mapping["aggregator"],
            path=f"{path}.aggregator",
        ),
    )


def create_verifier_from_config(config: object) -> tuple[Evaluator, Policy]:
    root = _require_mapping(config, "config")
    _reject_unknown_config_keys("config", root, {"speculative"})
    if "speculative" not in root or root["speculative"] is None:
        raise ValueError("speculative must be set.")
    speculative = _require_mapping(root["speculative"], "speculative")
    _reject_unknown_config_keys(
        "speculative",
        speculative,
        set(CANONICAL_SPECULATIVE_CONFIG_FIELDS),
    )
    if "evaluator" not in speculative or speculative["evaluator"] is None:
        raise ValueError("speculative.evaluator must be set.")
    if "acceptance" not in speculative or speculative["acceptance"] is None:
        raise ValueError("speculative.acceptance must be set.")
    return (
        create_evaluator_from_config(speculative["evaluator"]),
        create_acceptance_policy_from_config(speculative["acceptance"]),
    )


__all__ = [
    "ACCEPTANCE_FACTORIES",
    "AGGREGATOR_FACTORIES",
    "CANONICAL_SPECULATIVE_CONFIG_FIELDS",
    "DECODER_FACTORIES",
    "POLICY_FACTORIES",
    "POLICY_FACTORY_MAP",
    "SCORER_FACTORIES",
    "create_acceptance_policy_from_config",
    "create_aggregator_from_config",
    "create_decoder_from_config",
    "create_evaluator_from_config",
    "create_policy",
    "create_scorer_from_config",
    "create_verifier_from_config",
]
