# Dynamic Multi-LoRA Hot-Swapping & CI/CD Agent Evaluation (`adk eval`) on Cloud Run GPUs with Gemma 4 & vLLM

![Cover](https://raw.githubusercontent.com/RobinaMirbahar/gemma4-multilora-adk-eval/main/assets/cover.multilora.v2.jpg)

Official reference repository for the hands-on Google Codelab: **Dynamic Multi-LoRA Hot-Swapping & CI/CD Agent Evaluation (`adk eval`) on Cloud Run GPUs with Gemma 4 & vLLM**.

---

## Architectural Overview

Instead of deploying three separate Cloud Run GPU services for three fine-tuned specialist agents, this architecture loads frozen open-weights **Gemma 4 (`google/gemma-4-27b-it`)** once into a single **Cloud Run NVIDIA L4 GPU** and dynamically hot-swaps lightweight rank-16 QLoRA adapters (~64.5 MB VRAM overhead each) from **Cloud Storage FUSE** (`/mnt/models/loras/*`) in under **2 milliseconds**:

![Multi-LoRA Architecture](https://raw.githubusercontent.com/RobinaMirbahar/gemma4-multilora-adk-eval/main/assets/multilora_cloud_run_gpu_arch.jpg)

Every prompt or LoRA adapter update is gated in **Google Cloud Build** using **Google ADK's evaluation framework (`adk eval`)** across two deterministic metrics before shifting production traffic:
* **`tool_trajectory_avg_score` ($\ge 0.95$):** Verifies exact tool-call sequence and JSON arguments.
* **`response_match_score` ($\ge 0.85$):** Verifies ROUGE token-level factuality against the Golden Evaluation Dataset.

![ADK Eval CI/CD Gate](https://raw.githubusercontent.com/RobinaMirbahar/gemma4-multilora-adk-eval/main/assets/adk_eval_cicd_release_gate.jpg)

---

## Project Structure

```text
gemma4-multilora-adk-eval/
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── multilora_adk_eval/
│   ├── schemas.py                 # Pydantic v2 schemas for LoRA metadata & eval scores
│   ├── vllm_multilora_router.py   # Runtime LoRA hot-loader (POST /v1/load_lora_adapter)
│   ├── agent.py                   # 3-specialist Google ADK Multi-LoRA Coordinator
│   ├── golden_eval_set.json       # ADK Golden Evaluation Dataset
│   └── eval_config.json           # Release gate threshold config (0.95 / 0.85)
├── scripts/
│   └── run_multilora_adk_eval.py  # Multi-LoRA vs. Base Gemma 4 evaluation harness
└── deploy/
    ├── deploy_multilora_cloud_run.sh  # Cloud Run NVIDIA L4 GPU + GCS FUSE deployment
    └── cloudbuild_adk_eval.yaml       # 3-step Cloud Build CI/CD release gate
```

---

## Quick Start (Google Cloud Shell)

```bash
git clone https://github.com/RobinaMirbahar/gemma4-multilora-adk-eval.git
cd gemma4-multilora-adk-eval

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the Multi-LoRA vs. Base Gemma 4 ADK Evaluation Suite
python3 scripts/run_multilora_adk_eval.py
```

### Expected Output

```text
========================================================================================
[STAGE 1] Hot-Loading 3 Domain QLoRA Adapters into Shared Gemma 4 vLLM Server
========================================================================================
  [HOT-LOADED] sre-incident-lora    -> /mnt/models/loras/sre-incident-v1 (rank=16, VRAM=+64.5 MB)
  [HOT-LOADED] finops-cost-lora     -> /mnt/models/loras/finops-cost-v1 (rank=16, VRAM=+64.5 MB)
  [HOT-LOADED] secops-audit-lora    -> /mnt/models/loras/secops-audit-v1 (rank=16, VRAM=+64.5 MB)

========================================================================================
[STAGE 3] CI/CD Release Gate Verdict (adk eval)
========================================================================================
  Base Gemma 4 Tool Trajectory Avg      : 0.33 (FAIL < 0.95)
  Multi-LoRA Tool Trajectory Avg        : 1.00 (PASS >= 0.95)
  Multi-LoRA Response Factuality Avg    : 1.00 (PASS >= 0.85)
  Cloud Build Traffic Promotion Gate    : APPROVED (100% TRAFFIC)
```

---

## Cloud Shell Execution & Editor Screenshots

| Cloud Shell Editor (`vllm_multilora_router.py`) | Cloud Shell Editor (`agent.py`) |
| :---: | :---: |
| ![vLLM Router](https://raw.githubusercontent.com/RobinaMirbahar/gemma4-multilora-adk-eval/main/assets/screenshots/04_editor_vllm_multilora_router.png) | ![ADK Coordinator](https://raw.githubusercontent.com/RobinaMirbahar/gemma4-multilora-adk-eval/main/assets/screenshots/05_editor_adk_agent.png) |

| ADK Eval Stage 1–3 Verdict | Validated `EvalSuiteSummary` JSON |
| :---: | :---: |
| ![Eval Verdict](https://raw.githubusercontent.com/RobinaMirbahar/gemma4-multilora-adk-eval/main/assets/screenshots/07_terminal_eval_stages_verdict.png) | ![Eval JSON](https://raw.githubusercontent.com/RobinaMirbahar/gemma4-multilora-adk-eval/main/assets/screenshots/08_terminal_eval_json_output.png) |


---

## 👩‍💻 About the Author

<div align="center">
  <img src="https://raw.githubusercontent.com/RobinaMirbahar/RobinaMirbahar/main/ProfileImages/Hi-Robina.gif" height="75" width="75" alt="Robina Mirbahar" />

  <h3><strong>Robina Mirbahar — Google Cloud &amp; AI Architect</strong></h3>

  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=18&pause=1000&color=4285F4&center=true&vCenter=true&width=720&lines=Google+Cloud+%26+Agentic+AI+Solutions+Architect;Author+of+'AI+Engineering'+(2026);Pakistan+Lead+%40+IEEE+AI+Caravan+(94+Countries);Women+Techmakers+Ambassador+%26+Mentor+%40+Google+for+Startups" alt="Robina Mirbahar Dynamic Headline" />

  <p>
    AI Researcher, Cloud Solutions Architect, and Author of <b><i>"AI Engineering: From Classical ML to RAG &amp; Agentic AI Systems"</i> (2026)</b>.<br/>
    Bridging cutting-edge open-weights AI research (Gemma 4, Google ADK, Multi-LoRA, MCP &amp; A2A) with production-grade Cloud Run &amp; Vertex AI engineering.
  </p>

  [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/robinamirbahar)
  [![Amazon Book](https://img.shields.io/badge/Amazon-Book%20(2026)-FF9900?style=for-the-badge&logo=amazon&logoColor=white)](https://amazon.com/dp/B0GWCZWCF4)
  [![Google Developer](https://img.shields.io/badge/Google%20Developer-Profile-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://developers.google.com/profile/u/106918534005910641104)
  [![DEV Community](https://img.shields.io/badge/DEV-Articles-0A0A0A?style=for-the-badge&logo=devdotto&logoColor=white)](https://dev.to/robinamirbahar)
  [![Medium](https://img.shields.io/badge/Medium-Blog-12100E?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@robinamirbahar)
  [![YouTube](https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@robinacodesai)
  [![Twitter/X](https://img.shields.io/badge/X-000000?style=for-the-badge&logo=x&logoColor=white)](https://twitter.com/robinamirbahar)
  [![Stack Overflow](https://img.shields.io/badge/Stack%20Overflow-F58025?style=for-the-badge&logo=stackoverflow&logoColor=white)](https://stackoverflow.com/users/21196122/robina-mirbahar)
  [![Credly](https://img.shields.io/badge/Credly-FF6B35?style=for-the-badge&logo=credly&logoColor=white)](https://www.credly.com/users/robinamirbahar)

</div>

### 🚀 Featured Google Cloud & Agentic AI Repositories by Robina Mirbahar

| Repository | Architecture Focus | Live Repo & Dynamic Status |
| :--- | :--- | :--- |
| **[gemma4-multilora-adk-eval](https://github.com/RobinaMirbahar/gemma4-multilora-adk-eval)** | Dynamic Multi-LoRA Hot-Swapping & CI/CD Agent Evaluation (`adk eval`) on Cloud Run GPUs with Gemma 4 & vLLM | ![Stars](https://img.shields.io/github/stars/RobinaMirbahar/gemma4-multilora-adk-eval?style=flat-square&color=4285F4) ![Last Commit](https://img.shields.io/github/last-commit/RobinaMirbahar/gemma4-multilora-adk-eval?style=flat-square) |
| **[slothsre-napops](https://github.com/RobinaMirbahar/slothsre-napops)** | SlothSRE ("NapOps") — Overnight Gemma 4 & Google ADK Incident Digest Agent with OpenTelemetry & Sentry AI Monitoring | ![Stars](https://img.shields.io/github/stars/RobinaMirbahar/slothsre-napops?style=flat-square&color=34A853) ![Last Commit](https://img.shields.io/github/last-commit/RobinaMirbahar/slothsre-napops?style=flat-square) |
| **[streamforge-event-adk](https://github.com/RobinaMirbahar/streamforge-event-adk)** | Real-Time Event-Driven Multi-Agent Pipelines on Google Cloud with ADK (`ParallelAgent` & `LoopAgent`), Eventarc & Pub/Sub | ![Stars](https://img.shields.io/github/stars/RobinaMirbahar/streamforge-event-adk?style=flat-square&color=FBBC04) ![Last Commit](https://img.shields.io/github/last-commit/RobinaMirbahar/streamforge-event-adk?style=flat-square) |
| **[a2a-mcp-mesh-cloudrun](https://github.com/RobinaMirbahar/a2a-mcp-mesh-cloudrun)** | Agent-to-Agent (A2A) & Multi-MCP Mesh on Cloud Run with Gemma 4, Google ADK, Spanner Graph & AlloyDB | ![Stars](https://img.shields.io/github/stars/RobinaMirbahar/a2a-mcp-mesh-cloudrun?style=flat-square&color=EA4335) ![Last Commit](https://img.shields.io/github/last-commit/RobinaMirbahar/a2a-mcp-mesh-cloudrun?style=flat-square) |
| **[cloudguard-gemma-agent](https://github.com/RobinaMirbahar/cloudguard-gemma-agent)** | 6-Part Production Gemma Enterprise Agent Series (ADK, Cloud Run L4 GPU, Firestore, Model Armor, Vertex AI QLoRA) | ![Stars](https://img.shields.io/github/stars/RobinaMirbahar/cloudguard-gemma-agent?style=flat-square&color=4285F4) ![Last Commit](https://img.shields.io/github/last-commit/RobinaMirbahar/cloudguard-gemma-agent?style=flat-square) |
| **[cozy-ai-agent-cafe-game](https://github.com/RobinaMirbahar/cozy-ai-agent-cafe-game)** | Prompt Paws: 60 FPS Cozy AI Agent Bakery Simulation Game teaching RAG, Guardrails, Context Caching & Multi-Agent Loops | ![Stars](https://img.shields.io/github/stars/RobinaMirbahar/cozy-ai-agent-cafe-game?style=flat-square&color=7928CA) ![Last Commit](https://img.shields.io/github/last-commit/RobinaMirbahar/cozy-ai-agent-cafe-game?style=flat-square) |

<div align="center">
  <img src="https://raw.githubusercontent.com/RobinaMirbahar/RobinaMirbahar/main/ProfileImages/GithubBanner.svg" alt="Robina Mirbahar Banner" width="100%" />
</div>
