# AIMLZG521 Assignment 2 – PS2: LLM Deployment Benchmark (run guide)

Deliverable notebook: `AIMLZG521_Assignment2_PS2_LLM_Deployment_Benchmark.ipynb` (unexecuted – you run it in the lab, then export the executed notebook to PDF).

## What the notebook does
| Task | Content |
|---|---|
| 1 | Healthcare patient-support use case, Qwen2.5-3B-Instruct, grounded system prompt + knowledge base, 40-query labelled dataset (4 categories), rule-based quality/safety rubric (validated on reference answers + negative controls) |
| 2 | 3 deployment configs: **C1** HF Transformers FP16, **C2** HF 4-bit NF4 (bitsandbytes), **C3** vLLM OpenAI server FP16 (auto-fallback to HF dynamic batching if vLLM fails). Load time, TTFT, latency P50/P95, tokens/s, peak VRAM, T4-projected throughput |
| 3 | Closed-loop workload test L1 (1 user) vs L2 (4 concurrent users), 2 reps each, scaling ratios, timeline plot |
| 4 | Quality/safety scores, safety gate (0 critical errors), 8 edge cases through a validating gateway (empty/None/too-long input, output cap, injection, emergencies), 13 SLOs with PASS/FAIL |
| 5 | Cost model (fleet size, $/month, $/1k requests, break-even vs managed API), decision matrix, data-driven recommendation, architecture diagram, rollout plan |

All analysis text that quotes numbers is generated from the measured tables, so it always matches your run. Every task ends with Explanation → Justification → Inference → Limitations → Improvements.

## Running in the university virtual lab (A40)
1. Start a GPU lab instance (NVIDIA A40). Upload the `.ipynb` and open it in Jupyter.
2. Run the **pip install cell** (§0.2), then **Kernel → Restart**.
3. **Run All.** Expected time ≈ 30–60 min (model download ≈ 6 GB; vLLM start-up 2–5 min). Watch your $10 budget – stop the instance when done.
4. Results are cached in `results/*.json`; re-running skips completed configs unless `FORCE_RERUN = True`. Download the whole `results/` folder as evidence.

`EMULATE_T4 = True` caps usable VRAM at 15 GB so conclusions transfer to the T4 in the brief; T4 throughput is reported as a labelled **ESTIMATE**. The notebook also runs unchanged on Colab T4 (Runtime → T4 GPU).

## Before submitting – you must do these
1. **Group details & contribution table** (first cell): replace `<Student n name>`, `<ID>`, `<email>`; adjust ownership rows.
2. **Manual review**: open `results/manual_review.csv`, check every row flagged critical/unsafe/unsupported, fill `override_correct` / `override_unsafe` / `override_critical` (0/1) and `reviewer_note` where the rubric is wrong, then re-run all cells from §2.8 onwards (overrides are applied automatically; benchmark runs are cached, so Run All also works).
3. **Verify prices/URLs** in §5.1 (T4 + VM hourly price, GPT-4o-mini token prices) and the References cell.
4. **AI-assistance disclosure** at the end – fill in per course policy.
5. Read the generated findings and recommendation; add your own interpretation where you disagree.

## PDF export
* Jupyter: **File → Save and Export Notebook As → HTML**, open it in a browser, **Print → Save as PDF**; or
* `jupyter nbconvert --to webpdf --allow-chromium-download <notebook>.ipynb`.

Submit the executed `.ipynb` + PDF (and `results/` if allowed).

## Troubleshooting
| Symptom | Fix |
|---|---|
| vLLM server fails to start | See `results/vllm_server.log`. Try `VLLM_EXTRA_ARGS = ["--enforce-eager"]` in the config cell and set `FORCE_RERUN = True` (or delete `results/C3_vLLM_FP16.json`). If it still fails, the notebook automatically runs the HF dynamic-batching fallback and documents the failure – that is acceptable evidence. |
| `pip` conflicts with the preinstalled torch | Restart the kernel after the install cell; if vLLM pulled a different torch, restart again before Run All. |
| bitsandbytes / C2 error | Requires a CUDA GPU; the failure is recorded and analysis continues with the other configs. |
| CUDA OOM | Lower `CONCURRENCY` or `MAX_MODEL_LEN`; keep `EMULATE_T4 = True`. |
| HF download slow/blocked | Set `HF_TOKEN` or pre-download the model; `MODEL_ID` can be changed to another ≤4B instruct model (update the model-choice justification). |

## Code self-test
The notebook was smoke-tested end-to-end on CPU with a tiny model (`PS2_SMOKE_TEST=1` → SmolLM2-135M). That mode only checks the code path; its numbers are meaningless and must never be reported.
