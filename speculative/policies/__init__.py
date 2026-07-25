"""Speculative policy implementations."""

from speculative.policies.fixed_threshold import FixedThresholdPolicy
from speculative.policies.scripted import ScriptedPolicy

__all__ = ["FixedThresholdPolicy", "ScriptedPolicy"]
