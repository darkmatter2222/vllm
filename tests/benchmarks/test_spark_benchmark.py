# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import importlib.util
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[2] / "benchmarks/spark/benchmark_qwen38.py"
_spec = importlib.util.spec_from_file_location("spark_benchmark", _path)
assert _spec is not None and _spec.loader is not None
benchmark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(benchmark)


def test_speculative_bursts_count_tokens_not_messages():
    events = [
        (2.0, {"choices": [{"index": 0, "token_ids": [1, 2, 3], "text": "abc"}]}),
        (2.5, {"choices": [{"index": 0, "token_ids": [4, 5], "text": "de"}]}),
        (3.0, {"choices": [{"index": 0, "token_ids": [6], "text": "f"}]}),
        (
            3.1,
            {"choices": [], "usage": {"prompt_tokens": 4096, "completion_tokens": 6}},
        ),
    ]
    result = benchmark.summarize_stream(events, 3.1)
    assert result["decode_tokens_per_second"] == 3
    assert result["ttft_seconds"] == 2
    assert result["input_tokens_per_ttft_second"] == 2048
    assert result["text"] == "abcdef"


def test_single_burst_has_no_decode_rate():
    event = {
        "choices": [{"token_ids": [1, 2], "text": "hi"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    }
    assert (
        benchmark.summarize_stream([(1.0, event)], 1.1)["decode_tokens_per_second"]
        is None
    )


@pytest.mark.parametrize(
    "event",
    [
        {"error": "out of memory"},
        {"choices": [{"text": "missing token IDs"}]},
        {
            "choices": [{"token_ids": [1]}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        },
    ],
)
def test_invalid_measurements_are_rejected(event):
    with pytest.raises(ValueError):
        benchmark.summarize_stream([(1.0, event)], 2.0)


def report():
    return {
        "checkpoint_id": "test/bf16@revision",
        "corpus_sha256": "abc",
        "generation": {"temperature": 0},
        "repeats": 1,
        "samples": [
            {
                "id": "code",
                "mode": mode,
                "trial": 0,
                "token_ids": [1, 2],
                "ttft_seconds": 2.0,
                "decode_tokens_per_second": 10.0,
            }
            for mode in ("cold", "warm")
        ],
    }


def test_compare_reports_quality_difference_and_matched_speed_change():
    base = report()
    candidate = copy.deepcopy(base)
    candidate["samples"][0].update(
        token_ids=[1, 3], ttft_seconds=1.0, decode_tokens_per_second=15.0
    )
    result = benchmark.compare_results(base, candidate)
    assert result["greedy_token_mismatches"] == [["code", "cold", 0]]
    cold = result["performance"]["cold"]
    assert cold["ttft_seconds_change_percent"]["code"] == -50
    assert cold["decode_tokens_per_second_change_percent"]["code"] == 50


@pytest.mark.parametrize(
    "key", ["checkpoint_id", "corpus_sha256", "generation", "repeats"]
)
def test_compare_rejects_different_workloads(key):
    candidate = report()
    candidate[key] = "different"
    with pytest.raises(ValueError):
        benchmark.compare_results(report(), candidate)


def test_compare_rejects_missing_samples():
    candidate = report()
    candidate["samples"].pop()
    with pytest.raises(ValueError):
        benchmark.compare_results(report(), candidate)
