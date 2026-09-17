# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure GB10 skinny GEMMs and export only validated winners.

Run on an otherwise idle Spark. No model weights are downloaded or modified.
The resulting profile is opt-in via VLLM_QWEN4_SPARK_GEMM_CONFIG.
"""

import argparse
import json
import math
import statistics
from functools import partial
from pathlib import Path


def parse_shape(value: str) -> tuple[int, int]:
    try:
        n, k = (int(x) for x in value.lower().split("x"))
        if n <= 0 or k <= 0:
            raise ValueError
        return n, k
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use positive NxK, e.g. 640x2560") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--shape", type=parse_shape, action="append")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.repeats < 10 or any(not 1 <= m <= 16 for m in args.rows):
        parser.error("Use at least 10 repeats and row counts in [1, 16]")

    import torch
    import torch.nn.functional as F

    from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
        SkinnyGemmConfig,
        shape_dynamic_skinny_gemm,
    )
    from vllm.models.qwen4_exp.nvidia.spark_gemm_config import (
        MIN_SPEEDUP,
        SCHEMA_VERSION,
        candidate_configs,
        validate_profile,
    )

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
        parser.error("This tuner requires a CUDA SM121 device (DGX Spark/GB10)")
    if not shape_dynamic_skinny_gemm.is_available():
        parser.error("Install the fork's CuTeDSL kernel dependencies first")
    torch.manual_seed(args.seed)
    # TP1 shapes; only matching unquantized layers consume the exported plans.
    shapes = args.shape or [
        (16384, 2560),
        (2560, 6144),
        (96, 2560),
        (13312, 2560),
        (640, 2560),
        (512, 2560),
        (1280, 2560),
        (2560, 640),
        (336, 10240),
        (320, 10240),
        (248320, 2560),
    ]
    props = torch.cuda.get_device_properties(0)
    flush_bytes = max(64 * 1024 * 1024, 2 * getattr(props, "L2_cache_size", 0))
    flush = torch.empty(flush_bytes, dtype=torch.uint8, device="cuda")

    def capture(fn):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = fn()
        return graph, result

    def measure_pair(baseline, candidate, streaming):
        samples = [[], []]
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        for repeat in range(args.repeats):
            # Alternate order to reduce thermal/clock-order bias.
            for index in (0, 1) if repeat % 2 == 0 else (1, 0):
                if streaming:
                    flush.zero_()
                starts[index].record()
                (baseline, candidate)[index].replay()
                ends[index].record()
            torch.cuda.synchronize()
            for index in range(2):
                samples[index].append(starts[index].elapsed_time(ends[index]))
        return dict(
            zip(("baseline_ms", "candidate_ms"), map(statistics.median, samples))
        )

    profile = {
        "schema_version": SCHEMA_VERSION,
        "device_capability": [12, 1],
        "dtype": "bfloat16",
        "device_name": props.name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "seed": args.seed,
        "repeats": args.repeats,
        "flush_bytes": flush_bytes,
        "entries": [],
    }
    with torch.inference_mode():
        for n, k in shapes:
            weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
            weight.mul_(1 / math.sqrt(k))
            for m in sorted(set(args.rows)):
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                reference = F.linear(x, weight)
                baseline, baseline_out = capture(partial(F.linear, x, weight))
                winner, best_speedup = None, MIN_SPEEDUP
                for config_dict in candidate_configs(m, n, k):
                    config = SkinnyGemmConfig(**config_dict)
                    actual = shape_dynamic_skinny_gemm(x, weight, config)
                    try:
                        torch.testing.assert_close(
                            actual, reference, rtol=0.01, atol=0.01
                        )
                        # Check a second scale before timing the candidate.
                        torch.testing.assert_close(
                            shape_dynamic_skinny_gemm(x * 0.01, weight, config),
                            F.linear(x * 0.01, weight),
                            rtol=0.01,
                            atol=0.0001,
                        )
                    except AssertionError:
                        continue
                    candidate, candidate_out = capture(
                        partial(shape_dynamic_skinny_gemm, x, weight, config)
                    )
                    hot = measure_pair(baseline, candidate, False)
                    streaming = measure_pair(baseline, candidate, True)
                    speedup = min(
                        t["baseline_ms"] / t["candidate_ms"] for t in (hot, streaming)
                    )
                    if speedup >= best_speedup:
                        best_speedup = speedup
                        winner = {
                            "shape": [m, n, k],
                            "config": config_dict,
                            "correctness_passed": True,
                            "hot": hot,
                            "streaming": streaming,
                        }
                    del candidate, candidate_out
                if winner is not None:
                    profile["entries"].append(winner)
                print(json.dumps({"shape": [m, n, k], "winner": winner}), flush=True)
                del baseline, baseline_out
            del weight
    validate_profile(profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
