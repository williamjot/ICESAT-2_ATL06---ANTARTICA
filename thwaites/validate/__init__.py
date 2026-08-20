"""Validações cruzadas com dados independentes."""

from thwaites.validate.velocity import (
    sample_velocity_at, aggregate_velocity_to_nodes, crosscheck_stable_zones,
    joint_classification, summarize_dynamics, distance_to_grounding_line,
    effective_sample_size, correlation_with_autocorrelation, flow_acceleration,
    ACCELERATION_BLOCKED_MSG, CLASS_STABLE_FAST, CLASS_THIN_FAST,
    CLASS_THIN_SLOW, CLASS_STABLE, CLASS_INCONCLUSIVE,
)

__all__ = [
    "sample_velocity_at", "aggregate_velocity_to_nodes", "crosscheck_stable_zones",
    "joint_classification", "summarize_dynamics", "distance_to_grounding_line",
    "effective_sample_size", "correlation_with_autocorrelation", "flow_acceleration",
    "ACCELERATION_BLOCKED_MSG", "CLASS_STABLE_FAST", "CLASS_THIN_FAST",
    "CLASS_THIN_SLOW", "CLASS_STABLE", "CLASS_INCONCLUSIVE",
]
