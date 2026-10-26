from multilora_adk_eval.agent import (
    diagnose_cloud_run_5xx_spike,
    quarantine_overprivileged_service_account,
    rightsize_cloud_run_gpu_allocation,
)
from multilora_adk_eval.schemas import (
    LoraAdapterMetadata,
    MultiLoraToolCall,
)
from scripts.run_multilora_adk_eval import (
    compute_rouge1_f1,
    compute_tool_trajectory_score,
)


def test_lora_adapter_metadata_schema():
    meta = LoraAdapterMetadata(
        adapter_id="sre-incident-lora",
        gcs_fuse_path="/mnt/models/loras/sre-incident-v1",
        lora_rank=16,
        lora_alpha=32,
        vram_overhead_mb=64.5,
    )
    assert meta.adapter_id == "sre-incident-lora"
    assert meta.base_model == "google/gemma-4-27b-it"


def test_domain_tools():
    sre = diagnose_cloud_run_5xx_spike("checkout-api", "us-central1", 15)
    assert sre["service_name"] == "checkout-api"
    finops = rightsize_cloud_run_gpu_allocation("gemma4-multilora-vllm", 16, 0)
    assert finops["target_concurrency"] == 16
    secops = quarantine_overprivileged_service_account(
        "legacy-worker-sa@prod-project.iam.gserviceaccount.com", "roles/editor"
    )
    assert secops["status"] == "QUARANTINED_AND_REMEDIATED"


def test_trajectory_and_rouge_metrics():
    expected = [MultiLoraToolCall(tool_name="diagnose_cloud_run_5xx_spike", arguments={"service_name": "api"})]
    actual = [MultiLoraToolCall(tool_name="diagnose_cloud_run_5xx_spike", arguments={"service_name": "api"})]
    assert compute_tool_trajectory_score(expected, actual) == 1.0
    assert compute_rouge1_f1("root cause checkout-api", "root cause checkout-api") == 1.0