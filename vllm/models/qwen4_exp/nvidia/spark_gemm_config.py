# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate measured GB10 GEMM plans without importing torch or CUDA."""

import json
import math
from pathlib import Path

SCHEMA_VERSION = 1
MIN_SPEEDUP = 1.05
CONFIG_FIELDS = {
    "num_rows",
    "block_size",
    "outputs_per_block",
    "k_unroll",
    "vector_width",
    "static_k",
}


def validate_profile(profile: dict) -> dict[tuple[int, int], dict[int, dict]]:
    """Accept only correct, measured winners for BF16 on SM121."""
    if not isinstance(profile, dict):
        raise ValueError("Spark GEMM profile must be an object")
    if profile.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported Spark GEMM profile version")
    if profile.get("device_capability") != [12, 1]:
        raise ValueError("Spark GEMM profile must be measured on SM121")
    if profile.get("dtype") != "bfloat16":
        raise ValueError("Spark GEMM profile must preserve BF16 precision")
    entries = profile.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Spark GEMM profile entries must be a list")
    plans: dict[tuple[int, int], dict[int, dict]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Invalid Spark GEMM entry")
        shape = entry.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or any(type(x) is not int or x <= 0 for x in shape)
        ):
            raise ValueError("GEMM shape must be positive integer [M, N, K]")
        m, n, k = shape
        config = entry.get("config")
        if not isinstance(config, dict) or set(config) != CONFIG_FIELDS:
            raise ValueError("Invalid skinny GEMM configuration fields")
        if any(
            type(config[x]) is not int or config[x] <= 0
            for x in CONFIG_FIELDS - {"static_k"}
        ):
            raise ValueError("Skinny GEMM settings must be positive integers")
        if not 1 <= m <= 16 or config["num_rows"] != m:
            raise ValueError("Skinny GEMM row count must match M in [1, 16]")
        if config["block_size"] not in (32, 64, 128, 256):
            raise ValueError("Unsupported Spark GEMM block size")
        if config["vector_width"] not in (2, 4, 8):
            raise ValueError("Unsupported Spark GEMM vector width")
        if config["outputs_per_block"] not in (1, 2, 4):
            raise ValueError("Unsupported Spark GEMM output tile")
        if config["k_unroll"] not in (1, 2, 4):
            raise ValueError("Unsupported Spark GEMM unroll")
        if n % config["outputs_per_block"]:
            raise ValueError("Output tile does not divide N")
        if k % (config["block_size"] * config["vector_width"]):
            raise ValueError("Vector tile does not divide K")
        if config["static_k"] is not None and (
            type(config["static_k"]) is not int or config["static_k"] != k
        ):
            raise ValueError("Static K does not match the measured shape")
        if entry.get("correctness_passed") is not True:
            raise ValueError("Unvalidated GEMM configuration")
        for mode in ("hot", "streaming"):
            timing = entry.get(mode)
            if not isinstance(timing, dict):
                raise ValueError("Both hot and streaming measurements are required")
            values = [timing.get(name) for name in ("baseline_ms", "candidate_ms")]
            if any(
                type(x) not in (int, float) or not math.isfinite(x) or x <= 0
                for x in values
            ):
                raise ValueError("GEMM timings must be finite and positive")
            if values[0] / values[1] < MIN_SPEEDUP:
                raise ValueError("GEMM configuration does not beat the baseline by 5%")
        shape_plans = plans.setdefault((n, k), {})
        if m in shape_plans:
            raise ValueError("Duplicate measured GEMM shape")
        shape_plans[m] = dict(config)
    return plans


def load_profile(path: str) -> dict[tuple[int, int], dict[int, dict]]:
    with Path(path).open(encoding="utf-8") as handle:
        return validate_profile(json.load(handle))


def candidate_configs(m: int, n: int, k: int) -> list[dict]:
    """Enumerate legal candidates; none is enabled without measurements."""
    if not 1 <= m <= 16 or n <= 0 or k <= 0:
        raise ValueError("Invalid skinny GEMM shape")
    configs = []
    for block in (32, 64, 128):
        for vector in (4, 8):
            if k % (block * vector):
                continue
            for outputs in (1, 2, 4):
                if n % outputs:
                    continue
                for unroll in (1, 2):
                    configs.append(
                        {
                            "num_rows": m,
                            "block_size": block,
                            "outputs_per_block": outputs,
                            "k_unroll": unroll,
                            "vector_width": vector,
                            "static_k": k,
                        }
                    )
    return configs
