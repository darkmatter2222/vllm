#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "Stopped at line $LINENO. No automatic source-build fallback." >&2' ERR
[[ "$(uname -m)" == aarch64 ]] || { echo 'Requires ARM64'; exit 1; }
NATIVE_ROOT="$HOME/qwen38-spark-native"
mkdir -p "$NATIVE_ROOT"
cd "$NATIVE_ROOT"
sudo apt-get update
sudo apt-get install -y python3-venv libnuma1 libgomp1 libgl1 libglib2.0-0t64 libibverbs1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --only-binary=:all: uv==0.12.15
export UV_HTTP_TIMEOUT=300
WHEEL='https://wheels.vllm.ai/75c71390d5b399f5397a9166920fc45902f99f14/vllm-0.3.1.dev34%2Bg75c71390d-cp38-abi3-manylinux_2_28_aarch64.whl'
uv pip install --python "$VIRTUAL_ENV/bin/python" --only-binary=:all: --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://flashinfer.ai/whl "$WHEEL" 'torch==2.13.0+cu130'
python - <<'PYFORK'
import pathlib, importlib.util, urllib.request, time
root=pathlib.Path(importlib.util.find_spec('vllm').origin).parent
base='https://raw.githubusercontent.com/darkmatter2222/vllm/9399e732d7c8831c81f35e690b988150eb4957e7/vllm/'
staged=[]
for name in ['platforms/interface.py', 'envs.py', 'models/qwen4_exp/nvidia/hyperconnection.py', 'models/qwen4_exp/nvidia/low_latency_gemm.py', 'models/qwen4_exp/nvidia/ngram_embedding.py', 'models/qwen4_exp/nvidia/ops/qsa_indexer.py', 'models/qwen4_exp/nvidia/ops/spark.py', 'models/qwen4_exp/nvidia/spark_gemm_config.py']:
    for attempt in range(5):
        try:
            with urllib.request.urlopen(base+name, timeout=90) as response:
                data=response.read()
            break
        except Exception:
            if attempt == 4: raise
            time.sleep(5*(attempt+1))
    compile(data, name, 'exec')
    staged.append((root/name,data))
for path,data in staged:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(data)
print('Applied fork files and restored the matching platform interface.')
PYFORK
export HF_TOKEN="${HF_TOKEN:-}"
export HF_HOME=/home/darkmatter2222/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export MODEL_REPO=nvidia/Qwen3.8-Flash-Next-NVFP4
export MODEL_REVISION=fc694b54fb0174e0913e6adf86691ef85a4ead47
export MODEL_DIR=/home/darkmatter2222/models/Qwen3.8-Flash-Next-NVFP4-nvidia
export DOWNLOAD_MB_PER_SEC=50
export PATCH_REVISION=6ad1c8f15cbab1ababd2048e8e5f94094dbfc4a0
export SERVED_MODEL_NAME=qwen3.8-flash-next
export MAX_MODEL_LEN=262144
export MAX_NUM_SEQS=4
export MAX_BATCHED_TOKENS=4096
export GPU_MEMORY_UTILIZATION=0.80
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_DEEP_GEMM=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUTE_DSL_ARCH=sm_121a
export TORCH_CUDA_ARCH_LIST=12.1a
export FLASHINFER_CUDA_ARCH_LIST=12.1a
export FLASHINFER_DISABLE_VERSION_CHECK=1
export QWEN4EXP_PLE_MMAP=1
export QWEN4EXP_PLE_STAGED=1
export QWEN4EXP_PLE_MMAP_THREADS=64
export VLLM_QWEN4_SPARK_HC_FUSION="${VLLM_QWEN4_SPARK_HC_FUSION:-1}"
export VLLM_QWEN4_SPARK_QSA_PREFIX="${VLLM_QWEN4_SPARK_QSA_PREFIX:-1}"
export VLLM_QWEN4_SPARK_PLE_LOOKUP=0
export VLLM_QWEN4_SPARK_GEMM_CONFIG="${VLLM_QWEN4_SPARK_GEMM_CONFIG:-}"
export QWEN4EXP_DRAFT_VOCAB=65536
export VLLM_BUILD_COMMIT=9399e732d7c8831c81f35e690b988150eb4957e7
if [[ ! -f "$MODEL_DIR/.complete-$MODEL_REVISION" && ! -w "$MODEL_DIR" ]]; then
  sudo chown -R "$(id -u):$(id -g)" "$MODEL_DIR"
fi
python - <<'PYPORT'
import socket
with socket.socket() as s:
    try: s.bind(('0.0.0.0',8420))
    except OSError as e: raise SystemExit('Port 8420 is occupied. Stop the existing model service first: '+str(e))
PYPORT
set -Eeuo pipefail

echo "Qwen3.8 Flash-Next | single Spark | fork 9399e73"
echo "MTP3 + decode CUDA graphs + FP8 KV"
echo "262K native context | 4 active requests | vision"
echo "MTP FP8_PB_WO compatibility fix included"

# Fail before downloading weights if a stale/wrong image was launched.
python3 - <<'PYCODE'
import os
import pathlib
import torch
import vllm.envs as envs

expected = "9399e732d7c8831c81f35e690b988150eb4957e7"
if os.environ.get("VLLM_BUILD_COMMIT") != expected:
    raise RuntimeError("Wrong native fork revision")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable; check the NVIDIA container runtime")
if torch.cuda.get_device_capability() != (12, 1):
    raise RuntimeError("This configuration targets one DGX Spark/SM121")
for name in (
    "VLLM_QWEN4_SPARK_HC_FUSION",
    "VLLM_QWEN4_SPARK_QSA_PREFIX",
    "VLLM_QWEN4_SPARK_PLE_LOOKUP",
    "VLLM_QWEN4_SPARK_GEMM_CONFIG",
):
    if name not in envs.environment_variables:
        raise RuntimeError(f"Missing fork setting: {name}")
    print(f"{name}={getattr(envs, name)!r}", flush=True)
from vllm.models.qwen4_exp.nvidia.spark_gemm_config import load_profile
profile = os.environ.get("VLLM_QWEN4_SPARK_GEMM_CONFIG", "")
if profile:
    if not pathlib.Path(profile).is_file():
        raise RuntimeError(f"GEMM profile is not mounted: {profile}")
    plans = load_profile(profile)
    print(f"Validated GEMM profile: {len(plans)} shapes", flush=True)
print(f"GPU: {torch.cuda.get_device_name()}; CUDA: {torch.version.cuda}", flush=True)
PYCODE

######################################################################
# 1. APPLY PINNED RUNTIME OVERLAYS AND MTP COMPATIBILITY FIX
######################################################################

python3 - <<'PY'
import importlib.util
import pathlib
import os
import py_compile
import time
import urllib.request

revision = os.environ["PATCH_REVISION"]
base = (
    "https://raw.githubusercontent.com/"
    "tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark/"
    + revision + "/single-spark-vllm-tp1/patch/"
)

spec = importlib.util.find_spec("vllm")
if spec is None or spec.origin is None:
    raise RuntimeError("vLLM package not found")
root = pathlib.Path(spec.origin).parent

# Apache-2.0 community overlays by Tony DeAngelo / Kai,
# including credited upstream vLLM fixes.
# Source attribution headers remain intact.
files = {
    "ple_layer.py":
        "models/qwen4_exp/nvidia/ple_layer.py",
    "ple_mmap.py":
        "models/qwen4_exp/nvidia/ops/ple_mmap.py",
    "model_state.py":
        "models/qwen4_exp/nvidia/model_state.py",
    "mtp_draft_vocab.py":
        "models/qwen4_exp/nvidia/mtp.py",
    "upstream-overlays/ops_ple.py":
        "models/qwen4_exp/nvidia/ops/ple.py",
    "upstream-overlays/ops_qsa.py":
        "models/qwen4_exp/nvidia/ops/qsa.py",
    "upstream-overlays/qsa.py":
        "models/qwen4_exp/nvidia/qsa.py",
    "upstream-overlays/modelopt.py":
        "model_executor/layers/quantization/modelopt.py",
}

staged = []
for source, destination in files.items():
    target = root / destination
    if source != "ple_mmap.py" and not target.is_file():
        raise RuntimeError(
            f"Runtime layout mismatch: {target}"
        )

    for attempt in range(6):
        try:
            with urllib.request.urlopen(
                base + source, timeout=90
            ) as response:
                data = response.read()
            break
        except Exception as exc:
            if attempt == 5:
                raise
            delay = min(120, 5 * 2 ** attempt)
            print(
                f"Overlay download retry in {delay}s: "
                f"{source} ({type(exc).__name__})",
                flush=True,
            )
            time.sleep(delay)

    if source == "upstream-overlays/modelopt.py":
        text = data.decode("utf-8")
        old = (
            'if quant_algo in '
            '("FP8_BLOCK_SCALES", "FP8_BLOCK"):'
        )
        new = (
            'if quant_algo in '
            '("FP8_BLOCK_SCALES", "FP8_BLOCK", "FP8_PB_WO"):'
        )
        if text.count(old) == 1:
            text = text.replace(old, new)
        elif new not in text:
            raise RuntimeError(
                "Expected MTP FP8 dispatch branch not found"
            )
        data = text.encode("utf-8")
        print(
            "MTP FP8_PB_WO compatibility fix prepared.",
            flush=True,
        )

    if source == "ple_layer.py":
        text = data.decode("utf-8")
        # Legacy mmap gathers synchronously in forward; no async work to start.
        layer_marker = "class Qwen4ExpPLELayer("
        layer_start = text.index(layer_marker)
        forward_start = text.index("    def forward(", layer_start)
        if "    def start_prefetch(" not in text[layer_start:forward_start]:
            text = text[:forward_start] + (
                "    def start_prefetch(self, hidden_states, input_ids, "
                "query_start_loc, ngram_context):\n"
                "        # Synchronous mmap lookup occurs in forward.\n"
                "        return None\n\n"
            ) + text[forward_start:]
        # Current model.py expects PLE to include the incoming HC residual.
        old_return = "        return gated_output\n"
        new_return = "        return gated_output + hidden_states\n"
        if text.count(old_return) == 1:
            text = text.replace(old_return, new_return)
        elif new_return not in text:
            raise RuntimeError("Unexpected legacy PLE residual contract")
        data = text.encode("utf-8")

    compile(data, str(target), "exec")
    staged.append((target, data))

for target, data in staged:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    py_compile.compile(str(target), doraise=True)
    print(f"Applied {target.relative_to(root)}", flush=True)

print("Runtime overlays and MTP fix applied.", flush=True)
PY

# The old overlays are needed for this checkpoint's disk-backed PLE.
# They do not replace hyperconnection.py, low_latency_gemm.py,
# ops/qsa_indexer.py, ops/spark.py, spark_gemm_config.py, or envs.py.
# Import in a fresh interpreter to catch overlay API mismatches early.
python3 - <<'PYCODE'
import importlib
for suffix in ("model", "model_state", "mtp", "qsa", "ops.spark"):
    importlib.import_module("vllm.models.qwen4_exp.nvidia." + suffix)
print("Fork/overlay import checks passed; starting checkpoint preparation.", flush=True)
PYCODE

######################################################################
# 2. REUSE DOWNLOADS; RESUME MISSING FILES WITH RETRY BACKOFF
#    Sequential transfers, capped at 50 decimal MB/sec.
######################################################################

python3 - <<'PY'
import email.utils
import os
import pathlib
import random
import time

import requests
from huggingface_hub import HfApi, hf_hub_url

repo = os.environ["MODEL_REPO"]
revision = os.environ["MODEL_REVISION"]
root = pathlib.Path(os.environ["MODEL_DIR"])
root.mkdir(parents=True, exist_ok=True)
marker = root / (".complete-" + revision)

cap = float(os.environ["DOWNLOAD_MB_PER_SEC"]) * 1_000_000
if cap <= 0:
    raise ValueError("Download limit must be positive")

token = os.environ.get("HF_TOKEN") or None

def pause(attempt, response=None):
    delay = min(300, 15 * 2 ** min(attempt, 5))
    if response is not None:
        value = response.headers.get("Retry-After")
        if value:
            try:
                retry_after = float(value)
            except ValueError:
                try:
                    retry_after = (
                        email.utils.parsedate_to_datetime(
                            value
                        ).timestamp() - time.time()
                    )
                except (ValueError, TypeError, OverflowError):
                    retry_after = 0
            delay = max(delay, retry_after)
    delay = max(1, delay) + random.uniform(0, 5)
    print(f"Retrying in {delay:.0f}s...", flush=True)
    time.sleep(delay)

def download(session, item):
    relative = pathlib.PurePosixPath(item.rfilename)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("Unsafe checkpoint path")

    destination = root / item.rfilename
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = pathlib.Path(str(destination) + ".part")
    size = item.size

    if size is None:
        raise RuntimeError(
            f"Missing file size: {item.rfilename}"
        )

    if destination.exists():
        if destination.stat().st_size == size:
            print(f"Skip {item.rfilename}", flush=True)
            return
        raise RuntimeError(
            f"Existing file has unexpected size: {destination}"
        )

    for attempt in range(20):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > size:
            raise RuntimeError(f"Oversized partial: {partial}")
        if offset == size and partial.exists():
            os.replace(partial, destination)
            return

        headers = {}
        if token:
            headers["Authorization"] = "Bearer " + token
        if offset:
            headers["Range"] = f"bytes={offset}-"

        url = hf_hub_url(
            repo_id=repo,
            filename=item.rfilename,
            revision=revision,
        )

        print(
            f"Download {item.rfilename}: {offset}/{size}",
            flush=True,
        )

        try:
            with session.get(
                url,
                headers=headers,
                stream=True,
                timeout=(30, 300),
            ) as response:
                response.raise_for_status()

                if response.status_code == 206:
                    expected = f"bytes {offset}-"
                    if not response.headers.get(
                        "Content-Range", ""
                    ).startswith(expected):
                        raise RuntimeError(
                            "Invalid resume response"
                        )
                elif response.status_code == 200:
                    offset = 0
                else:
                    raise RuntimeError(
                        "Unexpected download response"
                    )

                started = time.monotonic()
                last_log = started
                transferred = 0

                with partial.open(
                    "ab" if offset else "wb"
                ) as output:
                    for block in response.iter_content(
                        1024 * 1024
                    ):
                        if not block:
                            continue
                        output.write(block)
                        transferred += len(block)

                        delay = transferred / cap - (
                            time.monotonic() - started
                        )
                        if delay > 0:
                            time.sleep(delay)

                        if time.monotonic() - last_log >= 30:
                            print(
                                f"  {offset + transferred}"
                                f"/{size} bytes",
                                flush=True,
                            )
                            last_log = time.monotonic()

                    output.flush()
                    os.fsync(output.fileno())

            if partial.stat().st_size != size:
                raise requests.exceptions.ChunkedEncodingError(
                    "Incomplete download"
                )

            os.replace(partial, destination)
            return

        except requests.exceptions.RequestException as exc:
            response = getattr(exc, "response", None)
            status = (
                response.status_code
                if response is not None else None
            )
            if status is not None and status not in (
                408, 429, 500, 502, 503, 504
            ):
                raise RuntimeError(
                    f"Download failed: HTTP {status}, "
                    f"{item.rfilename}"
                ) from None

            print(
                f"Temporary download failure: "
                f"{item.rfilename}, "
                f"HTTP {status}" if status is not None else
                f"Temporary network failure: "
                f"{type(exc).__name__}",
                flush=True,
            )
            if attempt == 19:
                raise RuntimeError(
                    "Download retries exhausted; "
                    "partial file preserved"
                ) from None
            pause(attempt, response)

if marker.exists():
    print("Pinned checkpoint already downloaded.", flush=True)
else:
    for attempt in range(10):
        try:
            info = HfApi(token=token).model_info(
                repo,
                revision=revision,
                files_metadata=True,
            )
            break
        except Exception as exc:
            response = getattr(exc, "response", None)
            status = (
                response.status_code
                if response is not None else None
            )
            if status is not None and status not in (
                408, 429, 500, 502, 503, 504
            ):
                raise RuntimeError(
                    f"Model metadata failed: HTTP {status}"
                ) from None
            if attempt == 9:
                raise RuntimeError(
                    "Model metadata retries exhausted"
                ) from None
            print(
                "Temporary model metadata failure.",
                flush=True,
            )
            pause(attempt, response)

    with requests.Session() as session:
        for item in sorted(
            info.siblings, key=lambda x: x.rfilename
        ):
            download(session, item)

    marker.write_text(revision + "\n")
    print("Checkpoint download complete.", flush=True)

# Advisory release of checkpoint file cache.
# Does not require privileged host cache clearing.
if hasattr(os, "posix_fadvise"):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            with path.open("rb") as handle:
                os.posix_fadvise(
                    handle.fileno(), 0, 0,
                    os.POSIX_FADV_DONTNEED,
                )
        except OSError:
            pass
PY

######################################################################
# 3. START VLLM
######################################################################

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

nvidia-smi

echo "Starting vLLM with patched MTP FP8 expert loading."
echo "Context: ${MAX_MODEL_LEN}"
echo "Active requests: ${MAX_NUM_SEQS}"
echo "Vision: one image per request"
echo "Thinking: OFF by default"
echo "API: host port 8420"
echo "Model loading and graph preparation can take 10+ minutes."

exec vllm serve "${MODEL_DIR}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 0.0.0.0 \
  --port 8420 \
  --trust-remote-code \
  --quantization modelopt \
  --tensor-parallel-size 1 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --enable-chunked-prefill \
  --kv-cache-dtype fp8_e4m3 \
  --no-enable-prefix-caching \
  --no-enable-flashinfer-autotune \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,12,16]}' \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --default-chat-template-kwargs '{"enable_thinking":false}'
