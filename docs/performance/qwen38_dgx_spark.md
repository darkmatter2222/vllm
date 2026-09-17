# Qwen3.8 Flash Next on one DGX Spark

This branch adds experimental, opt-in candidates to the NVIDIA Qwen4Exp
implementation used by this fork. It targets single-request latency on GB10
(SM121). It does not establish a speedup or model-quality result. Every new
runtime path is disabled by default; existing weights, expert routing, and
cache precision remain unchanged.

## Implemented candidates

| Setting | Candidate | Eligibility |
| --- | --- | --- |
| `VLLM_QWEN4_SPARK_PLE_LOOKUP=1` | Deduplicate repeated PLE row loads within eight-token tiles | SM121, pinned-host lookup, contiguous input, at least eight tokens |
| `VLLM_QWEN4_SPARK_HC_FUSION=1` | Fuse the HC up-projection, sigmoid, and stream reduction | SM121, BF16 unquantized replicated projection, 1–8 tokens, rank at most 512 |
| `VLLM_QWEN4_SPARK_QSA_PREFIX=1` | Enumerate visible blocks without scoring when every block fits the selection budget | SM121, short paged prefill, outside CUDA graph capture |
| `VLLM_QWEN4_SPARK_GEMM_CONFIG=/absolute/profile.json` | Dispatch measured small-row GEMM configurations | SM121, BF16, exact profiled shapes and row counts |

HC and QSA candidates are disabled in batch-invariant mode. HC preserves the
BF16 gate-output boundary, but reduction order can change rounding. QSA keeps
the selected block set, but changes its order. Neither implies bitwise model
output equivalence. PLE uses byte-preserving FP8 loads. The existing SM121 MoE
backend safety guard is retained.

## Measure before enabling

Use the exact checkpoint, quantization, context limit, offload settings, and
launch command that already work on your Spark. Do not infer a memory-fitting
launch configuration from this document. Keep the server otherwise idle.

Build/install the fork using its normal CUDA workflow. Run the GPU correctness
suite before activating candidates:

```bash
.venv/bin/python -m pytest tests/models/qwen4_exp/test_spark_ops.py -v
```

Create a JSONL corpus with unique string IDs and either text or token-ID lists:

```json
{"id":"short","prompt":"Explain why the sky appears blue."}
{"id":"code","prompt":"Write a Python function that merges two sorted lists."}
```

Include representative long prompts, coding, reasoning, retrieval and actual
production tasks. The small examples above are only a format demonstration.
Use identical corpus bytes for every run.

Start the server with all four new settings unset, then record a baseline:

```bash
.venv/bin/python benchmarks/spark/benchmark_qwen38.py \
  --model YOUR_SERVED_NAME --checkpoint-id YOUR_EXACT_REVISION_AND_QUANTIZATION \
  --prompts prompts.jsonl --label baseline --repeats 5 --max-tokens 256 \
  --reset-prefix-cache --output results/baseline.json
```

Restart with exactly one candidate enabled and run the same command with a new
label/output plus `--reference results/baseline.json`. The tool resets the prefix
cache on the dedicated server, measures sequential cold and repeated-prompt
requests, and saves raw generated token IDs. A greedy-token mismatch produces
a nonzero exit status after saving the report. This is a regression check;
run your model-quality evaluations separately, including non-greedy sampling
if used in production. Prefix-cache warmth must be confirmed from server
metrics; repeated requests alone do not prove a cache hit.

TTFT includes transport, queueing, prefill, and first-output overhead. The
input-token/TTFT ratio is not kernel-only prefill throughput. Decode rate counts
token IDs rather than SSE chunks, excludes the complete first burst, and is
unavailable when all output arrives in one burst. This also supports measuring
the existing MTP path without mistaking a speculative burst for a single token.

For GEMM, stop the server and tune in the same CUDA/CuTe DSL environment:

```bash
.venv/bin/python benchmarks/kernels/benchmark_qwen4_spark_gemm.py \
  --output spark-gemm.json --rows 1,2,4,8 --repeats 30
```

The tuner checks numerical tolerances and retains only configurations beating
the baseline by at least 5% in both hot-cache and cache-flushed measurements.
Default shapes cover candidate TP=1 projections; use repeated `--shape NxK`
arguments for actual local checkpoint shapes. An empty profile is valid and
keeps the original dispatch. These microbenchmark results are not an
end-to-end guarantee. Restart with the profile setting to evaluate the full
model. Retune after changing CUDA, driver, CuTe DSL, or the device environment.

Reject candidates that regress latency, memory use, or your quality threshold.
Measure each independently, then measure the winning combination: gains are
not additive. Unset the settings and restart to restore baseline behavior.

## Ten-workstream status

| Workstream | Included here | Remaining work |
| --- | --- | --- |
| Small-batch MoE | Existing backend choices and SM121 guard preserved | Profile routing and expert GEMMs; implement and validate a GB10-specific winner |
| Native MTP | Token-aware comparison tool works with the existing implementation | Benchmark depths and acceptance; adaptive depth controller |
| PLE memory traffic | Tiled duplicate-row reuse | Measure reuse; design a bounded cache using the actual memory budget |
| Prefill specialization | Cold/warm latency measurements and PLE/QSA candidates | Shape-tuned attention and expert prefill kernels |
| GDN recurrent path | Existing fused/FlashInfer paths preserved | GPU profiling and a verified faster implementation |
| QSA selection | Short-prefill all-block shortcut | Long-context selector and attention optimization |
| Launch overhead | Existing capture behavior retained; HC operation has a fake implementation | Model-wide graph coverage and CPU scheduling profile |
| Small-row GEMMs | SM121 tuner, validated profile loader and dispatch | Run the tuner and retain measured winners |
| HyperConnection fusion | Small-row BF16 fused candidate and correctness tests | GPU timing and end-to-end quality evaluation |
| Hybrid prefix reuse | Cold/repeated-prefix comparison tool | Verify hybrid-state cache hits and optimize against actual traces |

This is an initial implementation, not ten completed rewrites. The remaining
items need the target checkpoint and hardware measurements to choose and
validate an implementation without compromising accuracy.

## Validation recorded for this branch

41 CPU unit tests passed, covering profile rejection/acceptance and streamed
benchmark accounting. Another 19 numerical cases passed using the real Triton
arithmetic in its CPU interpreter. All three new kernels compiled to SM121
CUDA binaries offline (PLE checked in BF16 and byte-copy variants). CUDA execution, full-model evaluations, and DGX Spark latency
measurements have not been performed in the development environment. Do not
enable these candidates as production defaults on the strength of CPU checks.

Ruff checks and formatting passed on changed Python files; `INP001` was excluded
because validation used a partial source snapshot. The full pre-commit run was
interrupted while fetching a hook dependency and has not passed. Re-run the
complete hooks in a full checkout before merging. AI assistance was used for
this implementation and validation.
