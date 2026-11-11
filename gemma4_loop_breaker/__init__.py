"""
Gemma 4 Loop-Breaker: Graph-Guided Counterfactual Context Reset (G-CCR) &
Phase-Adaptive Thinking Engine for Quantized Coding Agents.
"""

from gemma4_loop_breaker.loop_breaker import (
    AgentPhase,
    LoopBreakerEngine,
    LoopDiagnostic,
    NegativeConstraint,
    SamplingPolicy,
    TurnRecord,
    compute_action_entropy,
    compute_diff_similarity,
    extract_error_signature,
)
from gemma4_loop_breaker.evaluate_benchmark import (
    execute_patch_in_sandbox,
    extract_python_code_from_response,
    run_full_ablation_suite,
)

__all__ = [
    "AgentPhase",
    "LoopBreakerEngine",
    "LoopDiagnostic",
    "NegativeConstraint",
    "SamplingPolicy",
    "TurnRecord",
    "compute_action_entropy",
    "compute_diff_similarity",
    "extract_error_signature",
    "execute_patch_in_sandbox",
    "extract_python_code_from_response",
    "run_full_ablation_suite",
]
