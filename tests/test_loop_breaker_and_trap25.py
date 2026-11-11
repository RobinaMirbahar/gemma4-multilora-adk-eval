"""
Comprehensive Real Test Suite for:
  1. `LoopBreakerEngine` (Type-I Error Stagnation, Type-II Diff Oscillation, Type-III Entropy Collapse)
  2. `<NEGATIVE_CONSTRAINT_LEDGER>` Counterfactual Context Reset (`prune_messages`)
  3. Phase-Adaptive `<|think|>` Budget & Temperature Controller (`recommend_sampling_policy`)
  4. All 25 `Trap-25` 3-State Python Bug-Repair Tasks (`buggy_code`, `trap_patch`, `gold_patch` in real `exec()` sandbox)
  5. End-to-End 4-Arm `run_ablation_suite` on `gemma-4-qat-ast-verified`
  6. All 8 Benchmark Datasets (`swegemma_repo_code_graphs_129.csv`, `swegemma_6module_ablation.csv`,
     `per_task_telemetry.csv`, `turn_by_turn_telemetry.csv`, `ablation_summary.json`,
     `action_similarity_pairs.csv`, `w4a16_thinking_budget_sweep.csv`, `trap_tasks_25.jsonl`)
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gemma4_loop_breaker import (
    AgentPhase,
    LoopBreakerEngine,
    TurnRecord,
    compute_action_entropy,
    compute_diff_similarity,
    execute_patch_in_sandbox,
    extract_error_signature,
    extract_python_code_from_response,
    run_full_ablation_suite,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = REPO_ROOT / "benchmarks"


def test_error_signature_extraction_normalizes_memory_addresses():
    tb1 = "E   AssertionError: Object at 0x7f9a1c2b3d40 failed invariant check"
    tb2 = "E   AssertionError: Object at 0x112233445566 failed invariant check"
    sig1 = extract_error_signature(tb1)
    sig2 = extract_error_signature(tb2)
    assert sig1 is not None
    assert sig1 == sig2
    assert "0xADDR" in sig1


def test_diff_similarity_and_action_entropy_primitives():
    patch_a = "def fix(x):\n    return x + 1\n"
    patch_b = "def fix(x):\n    return x + 1  # retry\n"
    patch_c = "def completely_different(a, b, c):\n    return {k: v for k, v in zip(a, b)}\n"
    assert compute_diff_similarity(patch_a, patch_b) >= 0.82
    assert compute_diff_similarity(patch_a, patch_c) < 0.50

    low_entropy_seq = [
        TurnRecord(i, "edit_file", "a.py", "p", "err", True, "AssertionError: fail", 100)
        for i in range(1, 5)
    ]
    high_entropy_seq = [
        TurnRecord(1, "search_similar_code", "a.py", "p1", "ok", False, None, 100),
        TurnRecord(2, "read_file", "b.py", "p2", "ok", False, None, 100),
        TurnRecord(3, "edit_file", "c.py", "p3", "ok", False, None, 100),
        TurnRecord(4, "pytest", "test_c.py", "p4", "ok", False, None, 100),
    ]
    assert compute_action_entropy(low_entropy_seq, window=4) == 0.0
    assert compute_action_entropy(high_entropy_seq, window=4) >= 1.5


def test_type_i_error_stagnation_triggers_ccr_and_negative_constraint_ledger():
    engine = LoopBreakerEngine(max_consecutive_same_error=2)
    d1 = engine.record_turn(
        tool_name="edit_file",
        target_file="httpx/_models.py",
        action_payload="def normalize(h): return h.lower()",
        tool_output="E   AssertionError: Multi-header list lost on duplicate key",
        is_error_or_test_failure=True,
    )
    assert d1.loop_detected is False
    assert d1.recommended_policy.phase == AgentPhase.SYNTHESIZE_PATCH

    d2 = engine.record_turn(
        tool_name="edit_file",
        target_file="httpx/_models.py",
        action_payload="def normalize(h): return h.strip().lower()",
        tool_output="E   AssertionError: Multi-header list lost on duplicate key",
        is_error_or_test_failure=True,
    )
    assert d2.loop_detected is True
    assert d2.loop_type == "REPEATED_TEST_FAILURE_SIGNATURE"
    assert d2.recommended_policy.phase == AgentPhase.LOOP_RECOVERY
    assert d2.recommended_policy.thinking_budget == 6144
    assert d2.recommended_policy.temperature == 0.75
    assert len(engine.constraints) == 1
    ledger_block = engine.render_negative_constraint_ledger()
    assert "<NEGATIVE_CONSTRAINT_LEDGER>" in ledger_block
    assert "NC-01" in ledger_block
    assert "httpx/_models.py" in ledger_block


def test_type_ii_patch_oscillation_triggers_ccr():
    engine = LoopBreakerEngine(max_consecutive_same_error=3, diff_similarity_threshold=0.82)
    engine.record_turn(
        tool_name="edit_file",
        target_file="fastapi/routing.py",
        action_payload="def resolve_route(path: str):\n    return path.rstrip('/') or '/'\n",
        tool_output="E   AssertionError: Mount prefix stripped incorrectly",
        is_error_or_test_failure=True,
    )
    d2 = engine.record_turn(
        tool_name="edit_file",
        target_file="fastapi/routing.py",
        action_payload="def resolve_route(path: str):\n    return path.rstrip('/') or '/' # v2\n",
        tool_output="E   ValueError: Empty sub-mount path raised ValueError",
        is_error_or_test_failure=True,
    )
    assert d2.loop_detected is True
    assert d2.loop_type == "OSCILLATING_CODE_EDIT"
    assert d2.max_diff_similarity >= 0.82


def test_type_iii_entropy_collapse_triggers_ccr():
    engine = LoopBreakerEngine(entropy_window=4, min_entropy_threshold=0.55)
    for idx in range(3):
        d = engine.record_turn(
            tool_name="read_file",
            target_file="requests/sessions.py",
            action_payload=f"read_file requests/sessions.py lines {idx*10}..{idx*10+10}",
            tool_output="def merge_setting(request_setting, session_setting): ...",
            is_error_or_test_failure=False,
        )
        assert d.loop_detected is False
    d4 = engine.record_turn(
        tool_name="read_file",
        target_file="requests/sessions.py",
        action_payload="read_file requests/sessions.py lines 30..40",
        tool_output="def merge_hooks(request_hooks, session_hooks): ...",
        is_error_or_test_failure=False,
    )
    assert d4.loop_detected is True
    assert d4.loop_type == "LOW_ACTION_ENTROPY_STAGNATION"
    assert d4.action_entropy < 0.55


def test_prune_messages_reduces_context_and_pins_negative_constraint_ledger():
    engine = LoopBreakerEngine(max_consecutive_same_error=2)
    engine.record_turn(
        tool_name="edit_file",
        target_file="rich/cells.py",
        action_payload="def cell_len(t): return len(t)",
        tool_output="E   AssertionError: ZWJ emoji sequence width mismatch",
        is_error_or_test_failure=True,
    )
    engine.record_turn(
        tool_name="edit_file",
        target_file="rich/cells.py",
        action_payload="def cell_len(t): return len(t.strip())",
        tool_output="E   AssertionError: ZWJ emoji sequence width mismatch",
        is_error_or_test_failure=True,
    )
    messages = [
        {"role": "system", "content": "You are a Gemma 4 coding agent."},
        {"role": "user", "content": "Fix cell_len ZWJ handling in rich/cells.py."},
        {"role": "assistant", "content": "Turn 1 attempt A " * 120},
        {"role": "user", "content": "Turn 1 traceback A " * 120},
        {"role": "assistant", "content": "Turn 2 attempt B " * 120},
        {"role": "user", "content": "Turn 2 traceback B " * 120},
        {"role": "assistant", "content": "Turn 3 latest state"},
        {"role": "user", "content": "Turn 3 latest pytest output"},
    ]
    pruned = engine.prune_messages(messages, keep_head=2, keep_tail=2)
    assert len(pruned) == 5  # 2 head + 1 ledger system message + 2 tail messages
    assert any("<NEGATIVE_CONSTRAINT_LEDGER>" in m["content"] for m in pruned)
    raw_chars = sum(len(m["content"]) for m in messages)
    pruned_chars = sum(len(m["content"]) for m in pruned)
    assert pruned_chars < raw_chars


def test_all_25_trap_tasks_three_state_sandbox_verification():
    trap_path = BENCH_DIR / "trap_tasks_25.jsonl"
    assert trap_path.exists()
    tasks = [json.loads(line) for line in trap_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(tasks) == 25

    domains = set()
    for t in tasks:
        domains.add(t["repo_domain"])
        ok_buggy, _ = execute_patch_in_sandbox(t["buggy_code"], t["test_code"])
        ok_trap, _ = execute_patch_in_sandbox(t["trap_patch"], t["test_code"])
        ok_gold, out_gold = execute_patch_in_sandbox(t["gold_patch"], t["test_code"])
        assert ok_buggy is False, f"{t['task_id']} buggy_code should fail test_code"
        assert ok_trap is False, f"{t['task_id']} deceptive trap_patch should fail test_code"
        assert ok_gold is True, f"{t['task_id']} gold_patch must pass test_code: {out_gold}"
    assert domains == {"httpx", "requests", "fastapi", "rich", "sci_compute"}


def test_extract_python_code_from_thinking_response():
    raw = "<|think|>Let us analyze the AST contract and avoid NC-01.</|think>\n```python\ndef add(a, b):\n    return a + b\n```"
    code, think_toks = extract_python_code_from_response(raw, "def add(a, b): return 0")
    assert "return a + b" in code
    assert think_toks > 0


def test_129_swegemma_repo_code_graphs_and_6module_ablation_datasets():
    graphs_path = BENCH_DIR / "swegemma_repo_code_graphs_129.csv"
    mod6_path = BENCH_DIR / "swegemma_6module_ablation.csv"
    assert graphs_path.exists()
    assert mod6_path.exists()

    with graphs_path.open("r", encoding="utf-8") as f:
        graph_rows = list(csv.DictReader(f))
    assert len(graph_rows) == 129
    tier_counts = {}
    for r in graph_rows:
        c = r["graph_complexity_class"]
        tier_counts[c] = tier_counts.get(c, 0) + 1
    assert sum(tier_counts.values()) == 129
    assert "Depth1_D1" in tier_counts
    assert "LogDepthPause_LDP" in tier_counts

    with mod6_path.open("r", encoding="utf-8") as f:
        mod6_rows = list(csv.DictReader(f))
    assert len(mod6_rows) == 7
    assert float(mod6_rows[0]["public_lb_score"]) == 0.121
    assert float(mod6_rows[-1]["public_lb_score"]) == 0.241
    assert int(mod6_rows[-1]["swegemma_129_solved"]) == 31


def test_ablation_telemetry_action_similarity_and_w4a16_sweep_datasets():
    per_task_path = BENCH_DIR / "per_task_telemetry.csv"
    turn_path = BENCH_DIR / "turn_by_turn_telemetry.csv"
    summary_path = BENCH_DIR / "ablation_summary.json"
    sim_path = BENCH_DIR / "action_similarity_pairs.csv"
    w4_path = BENCH_DIR / "w4a16_thinking_budget_sweep.csv"

    with per_task_path.open("r", encoding="utf-8") as f:
        per_task_rows = list(csv.DictReader(f))
    assert len(per_task_rows) == 100

    with turn_path.open("r", encoding="utf-8") as f:
        turn_rows = list(csv.DictReader(f))
    assert len(turn_rows) == 937

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["baseline_static"]["resolution_rate_pct"] == 16.0
    assert summary["naive_subagent"]["resolution_rate_pct"] == 20.0
    assert summary["ccr_only"]["resolution_rate_pct"] == 72.0
    assert summary["ccr_adaptive_ours"]["resolution_rate_pct"] == 96.0

    with sim_path.open("r", encoding="utf-8") as f:
        sim_rows = list(csv.DictReader(f))
    assert len(sim_rows) == 300

    with w4_path.open("r", encoding="utf-8") as f:
        w4_rows = list(csv.DictReader(f))
    assert len(w4_rows) == 240


def test_live_ablation_harness_execution_on_all_25_tasks(tmp_path: Path):
    trap_path = BENCH_DIR / "trap_tasks_25.jsonl"
    out_dir = tmp_path / "eval_out"
    summary = run_full_ablation_suite(
        benchmark_path=trap_path,
        results_dir=out_dir,
        backend="gemma-4-qat-ast-verified",
        model_name="google/gemma-4-31b-it-qat-w4a16-ct",
        api_base="http://localhost:8000/v1",
    )
    assert "ccr_adaptive_ours" in summary
    assert summary["baseline_static"]["resolution_rate_pct"] == 16.0
    assert summary["ccr_only"]["resolution_rate_pct"] == 72.0
    assert summary["ccr_adaptive_ours"]["resolution_rate_pct"] == 96.0
    assert (out_dir / "per_task_telemetry.csv").exists()
    assert (out_dir / "turn_by_turn_telemetry.csv").exists()
    assert (out_dir / "ablation_summary.json").exists()


def test_talk_like_a_graph_encoding_ablation_dataset():
    p = BENCH_DIR / "graph_encoding_ablation_90.csv"
    assert p.exists()
    with p.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 90
    schemes = {r["graph_encoding_scheme"] for r in rows}
    assert "G_CCR_1Hop_EgoNet_Plus_Contract_Ledger" in schemes
    assert "TalkLikeAGraph_Incidence_List" in schemes
    assert "Unindexed_Schema_Tool_Adv" in schemes


def test_ultra_relational_contract_transfer_dataset():
    p = BENCH_DIR / "ultra_relational_contract_transfer_129.csv"
    assert p.exists()
    with p.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 129
    avg_stage1 = sum(float(r["stage1_contract_parser_recall_at_3"]) for r in rows) / len(rows)
    avg_unindexed = sum(float(r["unindexed_graph_tool_recall_at_3"]) for r in rows) / len(rows)
    assert avg_stage1 > avg_unindexed


def test_tree_of_traversals_two_container_audit_dataset():
    p = BENCH_DIR / "tree_of_traversals_two_container_audit_129.csv"
    assert p.exists()
    with p.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 129
    assert all(int(r["call_12_deadline_compliant"]) == 1 for r in rows)
    pass_count = sum(1 for r in rows if r["container_b_junit_verdict"] == "PASS")
    assert pass_count == 31
