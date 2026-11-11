"""
Unit & Integration Tests for Dynamic Multi-LoRA Hot-Swapping & Google ADK `adk eval`
Release Gatekeeping (`multilora_adk_eval` & `scripts/run_multilora_adk_eval.py`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multilora_adk_eval.agent import (
    diagnose_cloud_run_5xx_spike,
    quarantine_overprivileged_service_account,
    rightsize_cloud_run_gpu_allocation,
)
from multilora_adk_eval.schemas import (
    LoraAdapterMetadata,
    MultiLoraToolCall,
)
from multilora_adk_eval.vllm_multilora_router import (
    ACTIVE_LORA_ADAPTERS,
    LoadLoraRequest,
    UnloadLoraRequest,
    list_models,
    load_lora_adapter,
    unload_lora_adapter,
)
from scripts.run_multilora_adk_eval import (
    compute_rouge1_f1,
    compute_tool_trajectory_score,
    run_evaluation_pipeline,
)
import asyncio

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_lora_adapter_metadata_schema():
    meta = LoraAdapterMetadata(
        adapter_id="sre-incident-lora",
        gcs_fuse_path="/mnt/models/loras/sre-incident-v1",
        lora_rank=16,
        lora_alpha=32,
        vram_overhead_mb=64.5,
    )
    assert meta.adapter_id == "sre-incident-lora"
    assert meta.base_model == "google/gemma-4-31b-it"
    assert meta.lora_rank == 16
    assert meta.lora_alpha == 32
    assert meta.vram_overhead_mb == 64.5


def test_domain_tools_sre_finops_secops():
    sre = diagnose_cloud_run_5xx_spike("checkout-api", "us-central1", 15)
    assert sre["service_name"] == "checkout-api"
    assert sre["region"] == "us-central1"
    assert "root_cause" in sre

    finops = rightsize_cloud_run_gpu_allocation("gemma4-multilora-vllm", 16, 0)
    assert finops["target_concurrency"] == 16
    assert finops["min_instances"] == 0

    secops = quarantine_overprivileged_service_account(
        "legacy-worker-sa@prod-project.iam.gserviceaccount.com", "roles/editor"
    )
    assert secops["status"] == "QUARANTINED_AND_REMEDIATED"
    assert secops["revoked_role"] == "roles/editor"


def test_vllm_multilora_router_hot_load_and_unload():
    ACTIVE_LORA_ADAPTERS.clear()
    for name, path in [
        ("sre-incident-lora", "/mnt/models/loras/sre-incident-v1"),
        ("finops-cost-lora", "/mnt/models/loras/finops-cost-v1"),
        ("secops-audit-lora", "/mnt/models/loras/secops-audit-v1"),
    ]:
        res = asyncio.run(
            load_lora_adapter(
                LoadLoraRequest(
                    lora_name=name,
                    lora_path=path,
                    lora_rank=16,
                    lora_alpha=32,
                    vram_overhead_mb=64.5,
                )
            )
        )
        assert res["status"] == "loaded"

    models_resp = asyncio.run(list_models())
    model_ids = [m["id"] for m in models_resp["data"]]
    assert "google/gemma-4-31b-it" in model_ids
    assert "sre-incident-lora" in model_ids
    assert "finops-cost-lora" in model_ids
    assert "secops-audit-lora" in model_ids
    total_overhead = sum(a.vram_overhead_mb for a in ACTIVE_LORA_ADAPTERS.values())
    assert abs(total_overhead - 193.5) < 1e-6

    unload_resp = asyncio.run(unload_lora_adapter(UnloadLoraRequest(lora_name="secops-audit-lora")))
    assert unload_resp["status"] == "unloaded"
    assert "secops-audit-lora" not in ACTIVE_LORA_ADAPTERS


def test_trajectory_and_rouge_metrics_exact_and_mismatch():
    expected = [
        MultiLoraToolCall(
            tool_name="diagnose_cloud_run_5xx_spike",
            arguments={"service_name": "checkout-api", "region": "us-central1"},
        )
    ]
    actual_match = [
        MultiLoraToolCall(
            tool_name="diagnose_cloud_run_5xx_spike",
            arguments={"service_name": "checkout-api", "region": "us-central1"},
        )
    ]
    actual_wrong_args = [
        MultiLoraToolCall(
            tool_name="diagnose_cloud_run_5xx_spike",
            arguments={"service_name": "wrong-service", "region": "us-east1"},
        )
    ]
    assert compute_tool_trajectory_score(expected, actual_match) == 1.0
    assert compute_tool_trajectory_score(expected, actual_wrong_args) < 1.0
    assert compute_tool_trajectory_score(expected, []) == 0.0
    assert compute_rouge1_f1("root cause checkout-api", "root cause checkout-api") == 1.0
    assert compute_rouge1_f1("root cause checkout-api", "unrelated text") == 0.0


def test_golden_eval_set_and_eval_config_files():
    golden_path = REPO_ROOT / "multilora_adk_eval" / "golden_eval_set.json"
    config_path = REPO_ROOT / "multilora_adk_eval" / "eval_config.json"
    assert golden_path.exists()
    assert config_path.exists()

    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert len(golden) >= 3
    assert config["criteria"]["tool_trajectory_avg_score"] == 0.95
    assert config["criteria"]["response_match_score"] == 0.85


def test_end_to_end_multilora_adk_eval_release_gate():
    summary = asyncio.run(run_evaluation_pipeline())
    assert len(summary.case_results) >= 3
    assert summary.multilora_trajectory_avg >= 0.95
    assert summary.multilora_response_match_avg >= 0.85
    assert summary.base_model_trajectory_avg < 0.95
    assert summary.release_gate_passed is True
