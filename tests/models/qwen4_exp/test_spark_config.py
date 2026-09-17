# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import json

import pytest

from vllm.models.qwen4_exp.nvidia.spark_gemm_config import (
    candidate_configs,
    load_profile,
    validate_profile,
)


def profile():
    return {
        "schema_version": 1,
        "device_capability": [12, 1],
        "dtype": "bfloat16",
        "entries": [
            {
                "shape": [1, 640, 2560],
                "config": candidate_configs(1, 640, 2560)[0],
                "correctness_passed": True,
                "hot": {"baseline_ms": 0.02, "candidate_ms": 0.01},
                "streaming": {"baseline_ms": 0.03, "candidate_ms": 0.02},
            }
        ],
    }


def test_measured_profile_round_trip(tmp_path):
    data = profile()
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(data))
    plans = load_profile(str(path))
    assert plans[(640, 2560)][1] == data["entries"][0]["config"]
    assert (512, 2560) not in plans
    assert 4 not in plans[(640, 2560)]


@pytest.mark.parametrize(
    "key,value",
    [
        ("device_capability", [12, 0]),
        ("dtype", "float16"),
        ("schema_version", 2),
        ("entries", {}),
    ],
)
def test_reject_wrong_profile_contract(key, value):
    data = profile()
    data[key] = value
    with pytest.raises(ValueError):
        validate_profile(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_rows", 2),
        ("num_rows", True),
        ("block_size", 0),
        ("block_size", 256),
        ("outputs_per_block", 3),
        ("vector_width", 3),
        ("static_k", 1280),
        ("k_unroll", 1000000),
    ],
)
def test_reject_illegal_kernel_layout(field, value):
    data = profile()
    data["entries"][0]["config"][field] = value
    with pytest.raises(ValueError):
        validate_profile(data)


@pytest.mark.parametrize("mode", ["hot", "streaming"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), 0.04, True])
def test_reject_unmeasured_or_regressing_kernel(mode, value):
    data = profile()
    data["entries"][0][mode]["candidate_ms"] = value
    with pytest.raises(ValueError):
        validate_profile(data)


def test_reject_unvalidated_or_duplicate_winner():
    data = profile()
    data["entries"][0]["correctness_passed"] = False
    with pytest.raises(ValueError):
        validate_profile(data)
    data = profile()
    data["entries"].append(copy.deepcopy(data["entries"][0]))
    with pytest.raises(ValueError):
        validate_profile(data)


def test_no_winners_keeps_default_implementation():
    data = profile()
    data["entries"] = []
    assert validate_profile(data) == {}


@pytest.mark.parametrize("shape", [(1, 640, 2560), (4, 336, 10240), (8, 248320, 2560)])
def test_tuner_emits_only_valid_candidates(shape):
    candidates = candidate_configs(*shape)
    assert candidates
    for config in candidates:
        data = profile()
        data["entries"][0].update(shape=list(shape), config=config)
        validate_profile(data)
