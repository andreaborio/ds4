#!/usr/bin/env python3
"""Freeze the pinned llama.cpp Qwen V-head tiling contract."""

from __future__ import annotations


GGUF_TO_HF_HEAD = [0, 2, 1, 3]
HEAD_DIM = 2


def permute_head_blocks(values: list[int], block: int) -> list[int]:
    assert len(values) == len(GGUF_TO_HF_HEAD) * block
    output: list[int] = []
    for hf_head in GGUF_TO_HF_HEAD:
        start = hf_head * block
        output.extend(values[start : start + block])
    return output


def dot(row: list[int], values: list[int]) -> int:
    assert len(row) == len(values)
    return sum(weight * value for weight, value in zip(row, values))


def main() -> None:
    # Literal sentinels make direction mistakes visible: GGUF heads 0..3 read
    # HF heads 0,2,1,3.  This is the 2x miniature of Qwen's 16x2 tiling.
    value_hf = [10, 11, 20, 21, 30, 31, 40, 41]
    value_gguf = permute_head_blocks(value_hf, HEAD_DIM)
    assert value_gguf == [10, 11, 30, 31, 20, 21, 40, 41]

    q_hf = [100, 101, 110, 111]
    k_hf = [200, 201, 210, 211]
    qkv_hf = q_hf + k_hf + value_hf
    qkv_gguf = q_hf + k_hf + value_gguf
    assert qkv_gguf == [
        100, 101, 110, 111,
        200, 201, 210, 211,
        10, 11, 30, 31, 20, 21, 40, 41,
    ]
    assert qkv_gguf[: len(q_hf) + len(k_hf)] == qkv_hf[:8]

    z_hf = [300, 301, 310, 311, 320, 321, 330, 331]
    assert permute_head_blocks(z_hf, HEAD_DIM) == [
        300, 301, 320, 321, 310, 311, 330, 331,
    ]

    for controls in (
        [400, 410, 420, 430],  # alpha
        [500, 510, 520, 530],  # beta
        [600, 610, 620, 630],  # transformed A
        [700, 710, 720, 730],  # dt bias
    ):
        assert permute_head_blocks(controls, 1) == [
            controls[0], controls[2], controls[1], controls[3]
        ]

    # Depthwise conv leaves Q/K channels fixed and applies the same V-row
    # permutation independently to every kernel tap.
    conv_qk = list(range(800, 808))
    conv_v_by_tap = [
        [900 + tap * 100 + head * 10 + dim
         for head in range(4) for dim in range(HEAD_DIM)]
        for tap in range(4)
    ]
    conv_gguf = [conv_qk + permute_head_blocks(v, HEAD_DIM)
                 for v in conv_v_by_tap]
    assert all(row[:8] == conv_qk for row in conv_gguf)
    assert conv_gguf[0][8:] == [900, 901, 920, 921, 910, 911, 930, 931]

    # W_out columns and its input activations must use the same direct
    # permutation.  Their dot product therefore remains unchanged.
    output_rows_hf = [
        [1, 2, 3, 4, 5, 6, 7, 8],
        [11, 13, 17, 19, 23, 29, 31, 37],
    ]
    output_rows_gguf = [
        permute_head_blocks(row, HEAD_DIM) for row in output_rows_hf
    ]
    assert output_rows_gguf[0] == [1, 2, 5, 6, 3, 4, 7, 8]
    assert [dot(row, value_gguf) for row in output_rows_gguf] == [
        dot(row, value_hf) for row in output_rows_hf
    ]

    print("qwen GGUF V-head tiling contract: OK")


if __name__ == "__main__":
    main()
