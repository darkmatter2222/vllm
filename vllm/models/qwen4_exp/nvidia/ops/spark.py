# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in GB10 kernels preserving weights, routing and cache precision."""

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _ple_lookup_tiled_kernel(
    weight_ptr,
    ids_ptr,
    out_ptr,
    tokens,
    heads,
    dim,
    vocab_start,
    vocab_end,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    head = tl.program_id(1)
    lane = tl.arange(0, BLOCK_T)
    token = tl.program_id(0) * BLOCK_T + lane
    ids = tl.load(ids_ptr + token * heads + head, token < tokens, other=-1)
    owned = (token < tokens) & (ids >= vocab_start) & (ids < vocab_end)
    # Only the first instance of each ID fetches a row from shared memory.
    same = (ids[:, None] == ids[None, :]) & owned[None, :]
    first = tl.min(tl.where(same, lane[None, :], BLOCK_T), axis=1)
    first = tl.minimum(first, BLOCK_T - 1)
    cols = tl.arange(0, BLOCK_D)
    local = tl.where(owned, ids - vocab_start, 0).to(tl.int64)
    values = tl.load(
        weight_ptr + local[:, None] * dim + cols[None, :],
        (owned & (first == lane))[:, None] & (cols[None, :] < dim),
        other=0,
    )
    values = tl.gather(
        values, tl.broadcast_to(first[:, None], (BLOCK_T, BLOCK_D)), axis=0
    )
    values = tl.where(owned[:, None], values, 0)
    tl.store(
        out_ptr + (token[:, None] * heads + head) * dim + cols[None, :],
        values,
        (token[:, None] < tokens) & (cols[None, :] < dim),
    )


def ple_lookup_tiled(
    weight: torch.Tensor,
    ids: torch.Tensor,
    output: torch.Tensor,
    vocab_start: int,
    vocab_end: int,
) -> None:
    """Gather owned rows, reusing repeated IDs within eight-token tiles."""
    assert ids.ndim == 2 and ids.is_contiguous()
    assert weight.ndim == 2 and weight.is_contiguous()
    assert output.shape == (*ids.shape, weight.shape[1])
    assert output.is_contiguous() and output.dtype == weight.dtype
    assert output.device == ids.device == weight.device
    if ids.numel() == 0:
        return
    # FP8 lookup is a byte copy, with no conversion or NaN canonicalization.
    if weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        weight, output = weight.view(torch.uint8), output.view(torch.uint8)
    _ple_lookup_tiled_kernel[(triton.cdiv(ids.shape[0], 8), ids.shape[1])](
        weight,
        ids,
        output,
        *ids.shape,
        weight.shape[1],
        vocab_start,
        vocab_end,
        BLOCK_T=8,
        BLOCK_D=triton.next_power_of_2(weight.shape[1]),
        num_warps=4,
    )


@triton.jit
def _hc_project_mix_kernel(
    x_ptr,
    lora_ptr,
    weight_ptr,
    output_ptr,
    stride_x,
    stride_lora,
    stride_w,
    stride_out,
    H: tl.constexpr,
    RANK: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    h = tile * BLOCK_H + tl.arange(0, BLOCK_H)
    r = tl.arange(0, BLOCK_R)
    lora = tl.load(lora_ptr + row * stride_lora + r, r < RANK, other=0)
    acc = tl.zeros((BLOCK_H,), tl.float32)
    for stream in tl.static_range(HC):
        cols = stream * H + h
        weight = tl.load(
            weight_ptr + cols[:, None] * stride_w + r[None, :],
            (h[:, None] < H) & (r[None, :] < RANK),
            other=0,
        )
        gate = tl.sum(weight.to(tl.float32) * lora[None, :].to(tl.float32), axis=1)
        # Match the original BF16 linear-output boundary before sigmoid.
        gate = gate.to(x_ptr.dtype.element_ty).to(tl.float32)
        x = tl.load(x_ptr + row * stride_x + cols, h < H, other=0)
        acc += tl.sigmoid(gate) * x.to(tl.float32)
    tl.store(output_ptr + row * stride_out + h, acc / HC, h < H)


def _hc_project_mix(
    x: torch.Tensor, lora: torch.Tensor, weight: torch.Tensor, hc_count: int
) -> torch.Tensor:
    assert x.ndim == lora.ndim == weight.ndim == 2
    assert x.shape[0] == lora.shape[0] and x.shape[1] % hc_count == 0
    assert weight.shape == (x.shape[1], lora.shape[1])
    assert x.dtype == lora.dtype == weight.dtype == torch.bfloat16
    assert x.device == lora.device == weight.device
    assert x.stride(1) == lora.stride(1) == weight.stride(1) == 1
    h, rank = x.shape[1] // hc_count, lora.shape[1]
    output = x.new_empty((x.shape[0], h))
    if x.shape[0] == 0:
        return output
    _hc_project_mix_kernel[(x.shape[0], triton.cdiv(h, 16))](
        x,
        lora,
        weight,
        output,
        x.stride(0),
        lora.stride(0),
        weight.stride(0),
        output.stride(0),
        H=h,
        RANK=rank,
        HC=hc_count,
        BLOCK_H=16,
        BLOCK_R=triton.next_power_of_2(rank),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


def _hc_project_mix_fake(
    x: torch.Tensor, lora: torch.Tensor, weight: torch.Tensor, hc_count: int
) -> torch.Tensor:
    return x.new_empty((x.shape[0], x.shape[1] // hc_count))


direct_register_custom_op(
    op_name="qwen4_exp_spark_hc_project_mix",
    op_func=_hc_project_mix,
    fake_impl=_hc_project_mix_fake,
)


def hc_project_mix(
    x: torch.Tensor, lora: torch.Tensor, weight: torch.Tensor, hc_count: int
) -> torch.Tensor:
    return torch.ops.vllm.qwen4_exp_spark_hc_project_mix(x, lora, weight, hc_count)


@triton.jit
def _qsa_all_blocks_kernel(
    visible_ptr,
    out_ptr,
    stride_out,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    visible = tl.load(visible_ptr + row)
    tl.store(
        out_ptr + row * stride_out + col, tl.where(col < visible, col, -1), col < TOPK
    )


def qsa_all_blocks(visible_blocks: torch.Tensor, output: torch.Tensor) -> None:
    """Enumerate all blocks when the host proved they fit the selection budget."""
    assert visible_blocks.is_contiguous()
    assert output.ndim == 2 and output.shape[0] == visible_blocks.numel()
    assert output.stride(1) == 1
    if output.shape[0] == 0:
        return
    _qsa_all_blocks_kernel[(output.shape[0],)](
        visible_blocks,
        output,
        output.stride(0),
        TOPK=output.shape[1],
        BLOCK=triton.next_power_of_2(output.shape[1]),
        num_warps=4,
    )
