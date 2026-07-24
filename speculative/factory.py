from __future__ import annotations

from typing import Any, Callable

from speculative.interfaces import Policy
from speculative.policies.scripted import ScriptedPolicy


PolicyFactory = Callable[..., Policy]


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


POLICY_FACTORY_MAP: dict[str, PolicyFactory] = {
    "always_accept": _always_accept_factory,
    "always_reject": _always_reject_factory,
    "reject_at_depth": _reject_at_depth_factory,
}


def create_policy(name: str, **kwargs: Any) -> Policy:
    """Create a policy from the explicit factory map."""

    try:
        factory = POLICY_FACTORY_MAP[name]
    except KeyError as exc:
        known = ", ".join(sorted(POLICY_FACTORY_MAP))
        raise ValueError(f"Unknown speculative policy {name!r}; known: {known}.") from exc
    return factory(**kwargs)
