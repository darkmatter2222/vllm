# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GB10 candidate correctness; performance is measured under benchmarks/kernels."""

import math

import pytest
import torch
import torch.nn.functional as F

from vllm.models.qwen4_exp.nvidia.ops.spark import (
    hc_project_mix,
    ple_lookup_tiled,
    qsa_all_blocks,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("tokens", [0, 1, 7, 8, 9, 17])
def test_tiled_ple_preserves_bytes_and_shard_mask(dtype, tokens):
    torch.manual_seed(7)
    weight = torch.randn(11, 37, device="cuda").to(dtype)
    ids = torch.tensor([[100, 103, 109], [100, 99, 111]], device="cuda")
    ids = ids.repeat((tokens + 1) // 2, 1)[:tokens].contiguous()
    output = torch.empty(tokens, 3, 37, dtype=dtype, device="cuda")
    ple_lookup_tiled(weight, ids, output, 100, 111)
    owned = (ids >= 100) & (ids < 111)
    index = (ids - 100).clamp(0, 10)
    expected = torch.where(owned[..., None], weight.float()[index], 0)
    expected = expected.to(dtype)
    assert torch.equal(output.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("tokens", [1, 4, 8])
@pytest.mark.parametrize("rank", [33, 320])
def test_hc_fused_projection_preserves_bf16_gate_boundary(tokens, rank):
    torch.manual_seed(5)
    # Non-contiguous outer stride exercises the merged down/injection slice.
    lora = torch.randn(tokens, rank + 16, device="cuda", dtype=torch.bfloat16)[:, :rank]
    x = torch.randn(tokens, 4 * 79, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(4 * 79, rank, device="cuda", dtype=torch.bfloat16)
    weight.mul_(1 / math.sqrt(rank))
    gate = F.linear(lora, weight).float()
    expected = (torch.sigmoid(gate) * x.float()).reshape(tokens, 4, 79).mean(1)
    actual = hc_project_mix(x, lora, weight, 4)
    torch.testing.assert_close(
        actual, expected.to(torch.bfloat16), rtol=0.01, atol=0.01
    )


def test_qsa_all_blocks_keeps_every_visible_block_and_padding():
    visible = torch.tensor([0, 1, 7, 512], dtype=torch.int32, device="cuda")
    output = torch.empty(4, 512, dtype=torch.int32, device="cuda")
    qsa_all_blocks(visible, output)
    cols = torch.arange(512, device="cuda").expand(4, -1)
    expected = torch.where(cols < visible[:, None], cols, -1).to(torch.int32)
    assert torch.equal(output, expected)


def test_hc_graph_replay_uses_updated_inputs():
    x = torch.randn(1, 4 * 64, device="cuda", dtype=torch.bfloat16)
    lora = torch.randn(1, 32, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(4 * 64, 32, device="cuda", dtype=torch.bfloat16) / 8
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        hc_project_mix(x, lora, weight, 4)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = hc_project_mix(x, lora, weight, 4)
    x.zero_()
    graph.replay()
    assert torch.count_nonzero(actual).item() == 0
