# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sequential Qwen/Spark benchmark with cold/warm separation and token checks.

Use a dedicated, otherwise idle vLLM server: this tool resets its prefix cache.
Input JSONL contains {"id": ..., "prompt": <string or token-ID list>} objects.
"""

import argparse
import hashlib
import json
import os
import statistics
import time
import urllib.request
from pathlib import Path


def summarize_stream(events: list[tuple[float, dict]], elapsed: float) -> dict:
    """Count tokens rather than SSE messages, including speculative bursts."""
    tokens, text, first, last, first_count, usage = [], [], None, None, 0, None
    metrics = None
    for timestamp, event in events:
        if "error" in event:
            raise ValueError(f"Server stream error: {event['error']}")
        if event.get("usage") is not None:
            usage = event["usage"]
        if event.get("metrics") is not None:
            metrics = event["metrics"]
        for choice in event.get("choices", []):
            if choice.get("index", 0) != 0:
                raise ValueError("Expected exactly one completion")
            ids = choice.get("token_ids") or []
            if choice.get("text") and not ids:
                raise ValueError("Server omitted token IDs; cannot measure decode")
            if ids:
                if first is None:
                    first, first_count = timestamp, len(ids)
                last = timestamp
                tokens.extend(ids)
                text.append(choice.get("text", ""))
    if not usage or first is None or last is None:
        raise ValueError("Incomplete stream: usage and generated tokens are required")
    if usage["completion_tokens"] != len(tokens):
        raise ValueError("Token deltas do not match completion usage")
    duration = last - first
    return {
        "prompt_tokens": usage["prompt_tokens"],
        "output_tokens": len(tokens),
        "ttft_seconds": first,
        "elapsed_seconds": elapsed,
        # Entire first burst is excluded, rather than assuming one token/chunk.
        "decode_tokens_per_second": (
            (len(tokens) - first_count) / duration if duration > 0 else None
        ),
        "input_tokens_per_ttft_second": usage["prompt_tokens"] / first,
        "first_burst_tokens": first_count,
        "token_ids": tokens,
        "text": "".join(text),
        "server_metrics": metrics,
        "prompt_tokens_details": usage.get("prompt_tokens_details"),
    }


def post(url: str, payload: dict, api_key: str, timeout: float):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers, method="POST"
    )
    return urllib.request.urlopen(request, timeout=timeout)


def complete(root: str, payload: dict, api_key: str, timeout: float) -> dict:
    started, events, done = time.perf_counter(), [], False
    with post(root + "/v1/completions", payload, api_key, timeout) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            events.append((time.perf_counter() - started, json.loads(data)))
    if not done:
        raise ValueError("Connection ended before the SSE completion marker")
    return summarize_stream(events, time.perf_counter() - started)


def compare_results(reference: dict, candidate: dict) -> dict:
    """Fail comparisons with different models, prompts or generation settings."""
    for key in ("checkpoint_id", "corpus_sha256", "generation", "repeats"):
        if reference[key] != candidate[key]:
            raise ValueError(f"Cannot compare different {key}")

    def keyed(report):
        items = report["samples"]
        result = {(s["id"], s["mode"], s["trial"]): s for s in items}
        if len(result) != len(items):
            raise ValueError("Duplicate benchmark sample")
        return result

    base, current = keyed(reference), keyed(candidate)
    if base.keys() != current.keys():
        raise ValueError("Benchmark sample sets do not match")
    mismatches, summary = [], {}
    for key in base:
        if base[key]["token_ids"] != current[key]["token_ids"]:
            mismatches.append(list(key))
    for mode in ("cold", "warm"):
        summary[mode] = {}
        for metric in ("ttft_seconds", "decode_tokens_per_second"):
            # Match each prompt/trial before aggregating; do not mix lengths.
            by_prompt = {}
            for key in base:
                if key[1] != mode:
                    continue
                b, c = base[key][metric], current[key][metric]
                if b is not None and c is not None and b > 0:
                    by_prompt.setdefault(key[0], []).append(100 * (c / b - 1))
            summary[mode][metric + "_change_percent"] = {
                prompt: statistics.median(values)
                for prompt, values in by_prompt.items()
            }
    return {"greedy_token_mismatches": mismatches, "performance": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True, help="Served model name")
    parser.add_argument(
        "--checkpoint-id",
        required=True,
        help="Exact checkpoint/revision/quantization identity",
    )
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument(
        "--reset-prefix-cache",
        action="store_true",
        help="Required: allow reset on a dedicated server",
    )
    args = parser.parse_args()
    if not args.reset_prefix_cache:
        parser.error("Use --reset-prefix-cache only against a dedicated idle server")
    if args.repeats < 1 or args.max_tokens < 2 or args.timeout <= 0:
        parser.error("repeats >= 1, max-tokens >= 2, timeout > 0 are required")
    corpus = args.prompts.read_bytes()
    prompts = [json.loads(line) for line in corpus.splitlines() if line.strip()]
    if not prompts or any(not isinstance(p.get("id"), str) for p in prompts):
        parser.error("Each prompt needs a string id")
    if len({p["id"] for p in prompts}) != len(prompts):
        parser.error("Prompt IDs must be unique")
    for prompt in prompts:
        value = prompt.get("prompt")
        if not (
            isinstance(value, str)
            and value
            or isinstance(value, list)
            and value
            and all(type(token) is int and token >= 0 for token in value)
        ):
            parser.error("Prompts must be nonempty strings or lists of token IDs")
    generation = {
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "seed": 42,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
    }
    report = {
        "schema_version": 1,
        "label": args.label,
        "model": args.model,
        "checkpoint_id": args.checkpoint_id,
        "corpus_sha256": hashlib.sha256(corpus).hexdigest(),
        "generation": generation,
        "repeats": args.repeats,
        "samples": [],
        "notes": [
            "Sequential requests; no concurrency throughput measurement.",
            (
                "input_tokens_per_ttft_second includes request overhead and first "
                "decode; it is not kernel-only prefill speed."
            ),
            (
                "Warm means an identical prompt after the cold request; inspect "
                "cache metrics to confirm the actual hit rate."
            ),
            "Greedy token agreement is a regression check, not an accuracy benchmark.",
        ],
    }
    root = args.base_url.rstrip("/")
    api_key = os.environ.get("VLLM_API_KEY", "")
    # Exclude lazy initialization; every measured cold request resets the cache.
    complete(
        root,
        {**generation, "model": args.model, "prompt": prompts[0]["prompt"]},
        api_key,
        args.timeout,
    )
    for trial in range(args.repeats):
        for prompt in prompts:
            with post(
                root + "/reset_prefix_cache", {}, api_key, args.timeout
            ) as response:
                response.read()
            for mode in ("cold", "warm"):
                sample = complete(
                    root,
                    {**generation, "model": args.model, "prompt": prompt["prompt"]},
                    api_key,
                    args.timeout,
                )
                sample.update(id=prompt["id"], mode=mode, trial=trial)
                report["samples"].append(sample)
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in sample.items()
                            if k not in ("token_ids", "text")
                        }
                    ),
                    flush=True,
                )
    if args.reference:
        report["comparison"] = compare_results(
            json.loads(args.reference.read_text()), report
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report.get("comparison", {}).get("greedy_token_mismatches"):
        raise SystemExit(
            "Greedy outputs differ; review quality before enabling changes"
        )


if __name__ == "__main__":
    main()
