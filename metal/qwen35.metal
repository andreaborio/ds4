// Qwen3.5-MoE / Qwen3.6 decode primitives for DS4's Metal backend.
//
// This is an original implementation of the scalar contracts in
// ds4_qwen_ref.c and ds4_qwen.c.  Buffers contain F32 values; every stride is
// expressed in bytes so the host can bind packed rows or larger workspaces.
// Stateful kernels operate on one decode token per dispatch.

#include <metal_stdlib>
using namespace metal;

struct ds4_metal_args_qwen35_split_q_gate {
    uint32_t n_token;
    uint32_t n_query_head;
    uint32_t head_dim;
    uint32_t reserved;
    uint64_t projection_token_stride;
    uint64_t projection_head_stride;
    uint64_t projection_dim_stride;
    uint64_t query_token_stride;
    uint64_t query_head_stride;
    uint64_t query_dim_stride;
    uint64_t gate_token_stride;
    uint64_t gate_head_stride;
    uint64_t gate_dim_stride;
};

struct ds4_metal_args_qwen35_sigmoid_mul {
    uint64_t n_value;
    uint64_t input_stride;
    uint64_t gate_stride;
    uint64_t output_stride;
};

struct ds4_metal_args_qwen35_sigmoid_mul_rows {
    uint32_t n_row;
    uint32_t row_width;
    uint64_t input_row_stride;
    uint64_t input_dim_stride;
    uint64_t gate_row_stride;
    uint64_t output_row_stride;
    uint64_t output_dim_stride;
};

struct ds4_metal_args_qwen35_rope {
    uint32_t n_token;
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t n_rot;
    float    theta;
    uint32_t reserved;
    uint64_t source_token_stride;
    uint64_t source_head_stride;
    uint64_t source_dim_stride;
    uint64_t output_token_stride;
    uint64_t output_head_stride;
    uint64_t output_dim_stride;
    uint64_t position_stride;
};

struct ds4_metal_args_qwen35_conv_step {
    uint32_t n_channel;
    uint32_t kernel_size;
    uint64_t input_channel_stride;
    uint64_t weight_channel_stride;
    uint64_t weight_tap_stride;
    uint64_t state_channel_stride;
    uint64_t state_tap_stride;
    uint64_t output_channel_stride;
};

struct ds4_metal_args_qwen35_conv_sequence {
    uint32_t n_token;
    uint32_t n_channel;
    uint32_t kernel_size;
    uint32_t reserved;
    uint64_t input_token_stride;
    uint64_t input_channel_stride;
    uint64_t weight_channel_stride;
    uint64_t weight_tap_stride;
    uint64_t state_channel_stride;
    uint64_t state_tap_stride;
    uint64_t output_token_stride;
    uint64_t output_channel_stride;
};

struct ds4_metal_args_qwen35_gated_delta_step {
    uint32_t n_key_head;
    uint32_t n_value_head;
    uint32_t key_dim;
    uint32_t value_dim;
    uint64_t query_head_stride;
    uint64_t query_dim_stride;
    uint64_t key_head_stride;
    uint64_t key_dim_stride;
    uint64_t value_head_stride;
    uint64_t value_dim_stride;
    uint64_t log_decay_head_stride;
    uint64_t beta_head_stride;
    uint64_t state_head_stride;
    uint64_t state_value_stride;
    uint64_t state_key_stride;
    uint64_t output_head_stride;
    uint64_t output_dim_stride;
};

struct ds4_metal_args_qwen35_gated_delta_sequence {
    uint32_t n_token;
    uint32_t n_key_head;
    uint32_t n_value_head;
    uint32_t key_dim;
    uint32_t value_dim;
    uint32_t reserved;
    uint64_t projection_token_stride;
    uint64_t query_offset;
    uint64_t key_offset;
    uint64_t value_offset;
    uint64_t query_head_stride;
    uint64_t query_dim_stride;
    uint64_t key_head_stride;
    uint64_t key_dim_stride;
    uint64_t value_head_stride;
    uint64_t value_dim_stride;
    uint64_t log_decay_token_stride;
    uint64_t log_decay_head_stride;
    uint64_t beta_token_stride;
    uint64_t beta_head_stride;
    uint64_t state_head_stride;
    uint64_t state_value_stride;
    uint64_t state_key_stride;
    uint64_t output_token_stride;
    uint64_t output_head_stride;
    uint64_t output_dim_stride;
};

struct ds4_metal_args_qwen35_rmsnorm_gated {
    uint32_t n_vector;
    uint32_t dim;
    float    epsilon;
    uint32_t reserved;
    uint64_t input_vector_stride;
    uint64_t input_dim_stride;
    uint64_t gate_vector_stride;
    uint64_t gate_dim_stride;
    uint64_t weight_dim_stride;
    uint64_t output_vector_stride;
    uint64_t output_dim_stride;
};

struct ds4_metal_args_qwen35_embedding_q8_0 {
    uint32_t row_index;
    uint32_t n_embd;
    uint32_t block_size;
    uint32_t reserved;
    uint64_t source_row_stride;
    uint64_t source_block_stride;
    uint64_t source_scale_offset;
    uint64_t source_quant_offset;
    uint64_t source_quant_stride;
    uint64_t output_dim_stride;
};

struct ds4_metal_args_qwen35_embedding_q8_0_batch {
    uint32_t n_token;
    uint32_t n_row;
    uint32_t n_embd;
    uint32_t block_size;
    uint64_t source_row_stride;
    uint64_t source_block_stride;
    uint64_t source_scale_offset;
    uint64_t source_quant_offset;
    uint64_t source_quant_stride;
    uint64_t token_id_stride;
    uint64_t output_token_stride;
    uint64_t output_dim_stride;
};

struct ds4_metal_args_qwen35_gated_delta_controls {
    uint32_t n_token;
    uint32_t n_value_head;
    uint64_t alpha_logit_token_stride;
    uint64_t alpha_logit_head_stride;
    uint64_t beta_logit_token_stride;
    uint64_t beta_logit_head_stride;
    uint64_t ssm_a_head_stride;
    uint64_t dt_bias_head_stride;
    uint64_t log_decay_token_stride;
    uint64_t log_decay_head_stride;
    uint64_t beta_token_stride;
    uint64_t beta_head_stride;
};

struct ds4_metal_args_qwen35_gqa_decode {
    uint32_t n_kv;
    uint32_t n_query_head;
    uint32_t n_kv_head;
    uint32_t head_dim;
    uint64_t query_head_stride;
    uint64_t query_dim_stride;
    uint64_t key_token_stride;
    uint64_t key_head_stride;
    uint64_t key_dim_stride;
    uint64_t value_token_stride;
    uint64_t value_head_stride;
    uint64_t value_dim_stride;
    uint64_t output_head_stride;
    uint64_t output_dim_stride;
};

struct ds4_metal_args_qwen35_gqa_prefill {
    uint32_t position0;
    uint32_t n_token;
    uint32_t n_query_head;
    uint32_t n_kv_head;
    uint32_t head_dim;
    uint32_t reserved;
    uint64_t query_token_stride;
    uint64_t query_head_stride;
    uint64_t query_dim_stride;
    uint64_t key_token_stride;
    uint64_t key_head_stride;
    uint64_t key_dim_stride;
    uint64_t value_token_stride;
    uint64_t value_head_stride;
    uint64_t value_dim_stride;
    uint64_t output_token_stride;
    uint64_t output_head_stride;
    uint64_t output_dim_stride;
};

struct ds4_metal_args_qwen35_router_top8 {
    uint32_t n_token;
    uint32_t reserved;
    uint64_t logits_token_stride;
    uint64_t logits_stride;
    uint64_t selected_token_stride;
    uint64_t selected_stride;
    uint64_t selected_weight_token_stride;
    uint64_t selected_weight_stride;
};

static inline float qwen35_metal_sigmoid(float x) {
    if (x >= 0.0f) {
        return 1.0f / (1.0f + exp(-x));
    }
    const float exponential = exp(x);
    return exponential / (1.0f + exponential);
}

static inline float qwen35_metal_silu(float x) {
    return x * qwen35_metal_sigmoid(x);
}

static inline float qwen35_metal_softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return exp(x);
    // Metal does not expose log1p.  Keep both tails stable instead of losing
    // exp(x) when it is added to 1 in ordinary float arithmetic.
    if (x < -10.0f) {
        const float exponential = exp(x);
        return exponential - 0.5f * exponential * exponential +
               (exponential * exponential * exponential) / 3.0f;
    }
    if (x > 10.0f) return x + log(1.0f + exp(-x));
    return log(1.0f + exp(x));
}

static inline float qwen35_metal_load_f32(
        device const char *base,
        uint64_t           offset) {
    return *((device const float *)(base + offset));
}

static inline void qwen35_metal_store_f32(
        device char *base,
        uint64_t     offset,
        float        value) {
    *((device float *)(base + offset)) = value;
}

// Scalar-reference stable softmax and deterministic top-8 routing for the
// fixed Qwen3.5/3.6 expert geometry.  The arithmetic order intentionally mirrors
// ds4_qwen35_cpu_softmax_top8_f32: normalize all 256 probabilities first,
// select in descending probability order (lower expert ID wins ties), then
// renormalize the eight selected probabilities.
//
// Dispatch requirements:
//   grid = n_token threadgroups
//   threads_per_threadgroup >= 256
//   threadgroup(0) scratch = 256 floats (1024 bytes)
//
// Model-produced logits are finite.  For defensive diagnostics, a non-finite
// input emits selected=-1 and weight=0 for every slot.
kernel void kernel_qwen35_router_softmax_top8_serial_f32(
        constant ds4_metal_args_qwen35_router_top8 &args [[buffer(0)]],
        device const char *logits          [[buffer(1)]],
        device       char *selected        [[buffer(2)]],
        device       char *selected_weight [[buffer(3)]],
        threadgroup float *probability [[threadgroup(0)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort3 thread_pos [[thread_position_in_threadgroup]],
        ushort3 threads [[threads_per_threadgroup]]) {
    constexpr uint n_expert = 256u;
    constexpr uint n_selected = 8u;
    const uint tid = thread_pos.x;
    const uint token = group.x;
    if (token >= args.n_token || threads.x < n_expert ||
        args.logits_stride < sizeof(float) ||
        args.selected_stride < sizeof(int32_t) ||
        args.selected_weight_stride < sizeof(float)) {
        return;
    }

    const uint64_t logits_base =
        (uint64_t)token * args.logits_token_stride;
    const uint64_t selected_base =
        (uint64_t)token * args.selected_token_stride;
    const uint64_t selected_weight_base =
        (uint64_t)token * args.selected_weight_token_stride;

    if (tid == 0u) {
        float maximum = qwen35_metal_load_f32(logits, logits_base);
        bool finite = isfinite(maximum);
        for (uint expert = 1u; expert < n_expert; expert++) {
            const float value = qwen35_metal_load_f32(
                logits,
                logits_base + (uint64_t)expert * args.logits_stride);
            finite = finite && isfinite(value);
            if (value > maximum) maximum = value;
        }
        probability[0] = maximum;
        probability[1] = finite ? 1.0f : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float maximum = probability[0];
    const bool finite = probability[1] != 0.0f;
    // Every lane must capture the two control values before lanes 0 and 1
    // reuse those scratch slots for exponentials.  Without this rendezvous,
    // a later SIMD group can observe overwritten controls and diverge around
    // the following threadgroup barrier.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!finite) {
        if (tid < n_selected) {
            *((device int32_t *)(selected +
                selected_base +
                (uint64_t)tid * args.selected_stride)) = -1;
            qwen35_metal_store_f32(
                selected_weight,
                selected_weight_base +
                    (uint64_t)tid * args.selected_weight_stride,
                0.0f);
        }
        return;
    }

    if (tid < n_expert) {
        probability[tid] = exp(qwen35_metal_load_f32(
            logits,
            logits_base + (uint64_t)tid * args.logits_stride) - maximum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid != 0u) return;

    float total = 0.0f;
    for (uint expert = 0u; expert < n_expert; expert++) {
        total += probability[expert];
    }
    for (uint expert = 0u; expert < n_expert; expert++) {
        probability[expert] /= total;
    }

    int32_t chosen[n_selected];
    float chosen_weight[n_selected];
    for (uint slot = 0u; slot < n_selected; slot++) {
        uint best = n_expert;
        for (uint expert = 0u; expert < n_expert; expert++) {
            bool used = false;
            for (uint prior = 0u; prior < slot; prior++) {
                if (chosen[prior] == (int32_t)expert) {
                    used = true;
                    break;
                }
            }
            if (used) continue;
            if (best == n_expert ||
                probability[expert] > probability[best] ||
                (probability[expert] == probability[best] && expert < best)) {
                best = expert;
            }
        }
        chosen[slot] = (int32_t)best;
        chosen_weight[slot] = probability[best];
    }

    float selected_total = 0.0f;
    for (uint slot = 0u; slot < n_selected; slot++) {
        selected_total += chosen_weight[slot];
    }
    for (uint slot = 0u; slot < n_selected; slot++) {
        *((device int32_t *)(selected +
            selected_base +
            (uint64_t)slot * args.selected_stride)) = chosen[slot];
        qwen35_metal_store_f32(
            selected_weight,
            selected_weight_base +
                (uint64_t)slot * args.selected_weight_stride,
            chosen_weight[slot] / selected_total);
    }
}

// Decode-optimized form of the router above.  Exponentials and final
// normalization retain the reference arithmetic, while eight deterministic
// two-level SIMD reductions replace the serial 8x256 selection scan.  Each
// reduction compares probability first and expert ID second, so lower IDs
// still win exact ties.  The larger scratch layout is:
//
//   256 probabilities, 32 partial values, 32 partial IDs,
//   8 selected IDs, 8 control words.
//
// Keeping the serial kernel above gives the host a precise diagnostic A/B and
// a fallback for any future device whose SIMD geometry violates this contract.
kernel void kernel_qwen35_router_softmax_top8_f32(
        constant ds4_metal_args_qwen35_router_top8 &args [[buffer(0)]],
        device const char *logits          [[buffer(1)]],
        device       char *selected        [[buffer(2)]],
        device       char *selected_weight [[buffer(3)]],
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort3 thread_pos [[thread_position_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simdgroup [[simdgroup_index_in_threadgroup]],
        ushort simd_width [[threads_per_simdgroup]],
        ushort3 threads [[threads_per_threadgroup]]) {
    constexpr uint n_expert = 256u;
    constexpr uint n_selected = 8u;
    constexpr uint max_simdgroup = 32u;
    const uint tid = thread_pos.x;
    const uint token = group.x;
    if (token >= args.n_token || threads.x != n_expert ||
        simd_width == 0u ||
        (threads.x + simd_width - 1u) / simd_width > max_simdgroup ||
        args.logits_stride < sizeof(float) ||
        args.selected_stride < sizeof(int32_t) ||
        args.selected_weight_stride < sizeof(float)) {
        return;
    }

    threadgroup float *probability = scratch;
    threadgroup float *partial_value = probability + n_expert;
    threadgroup uint *partial_id =
        (threadgroup uint *)(partial_value + max_simdgroup);
    threadgroup uint *chosen = partial_id + max_simdgroup;
    threadgroup uint *control = chosen + n_selected;
    threadgroup float *control_float =
        (threadgroup float *)(control + n_selected);
    const uint n_simdgroup =
        (threads.x + simd_width - 1u) / simd_width;

    const uint64_t logits_base =
        (uint64_t)token * args.logits_token_stride;
    const uint64_t selected_base =
        (uint64_t)token * args.selected_token_stride;
    const uint64_t selected_weight_base =
        (uint64_t)token * args.selected_weight_token_stride;
    const float logit = qwen35_metal_load_f32(
        logits, logits_base + (uint64_t)tid * args.logits_stride);

    const uint lane_finite = isfinite(logit) ? 1u : 0u;
    const uint group_finite = simd_min(lane_finite);
    const float group_maximum = simd_max(
        lane_finite != 0u ? logit : -INFINITY);
    if (lane == 0u) {
        partial_id[simdgroup] = group_finite;
        partial_value[simdgroup] = group_maximum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0u) {
        const uint finite_candidate =
            lane < n_simdgroup ? partial_id[lane] : 1u;
        const float maximum_candidate =
            lane < n_simdgroup ? partial_value[lane] : -INFINITY;
        const uint all_finite = simd_min(finite_candidate);
        const float maximum = simd_max(maximum_candidate);
        if (lane == 0u) {
            control[0] = all_finite;
            control_float[0] = maximum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const bool finite = control[0] != 0u;
    const float maximum = control_float[0];
    if (!finite) {
        if (tid < n_selected) {
            *((device int32_t *)(selected +
                selected_base +
                (uint64_t)tid * args.selected_stride)) = -1;
            qwen35_metal_store_f32(
                selected_weight,
                selected_weight_base +
                    (uint64_t)tid * args.selected_weight_stride,
                0.0f);
        }
        return;
    }

    const float probability_value = exp(logit - maximum);
    probability[tid] = probability_value;
    bool active = true;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint slot = 0u; slot < n_selected; slot++) {
        const float candidate = active ? probability_value : -INFINITY;
        const float simd_maximum = simd_max(candidate);
        const uint candidate_id =
            active && candidate == simd_maximum ? tid : UINT_MAX;
        const uint simd_id = simd_min(candidate_id);
        if (lane == 0u) {
            partial_value[simdgroup] = simd_maximum;
            partial_id[simdgroup] = simd_id;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simdgroup == 0u) {
            const float value = lane < n_simdgroup
                ? partial_value[lane]
                : -INFINITY;
            const float best_value = simd_max(value);
            const uint id = lane < n_simdgroup && value == best_value
                ? partial_id[lane]
                : UINT_MAX;
            const uint best_id = simd_min(id);
            if (lane == 0u) {
                chosen[slot] = best_id;
                control[1] = best_id;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == control[1]) active = false;
    }

    if (tid != 0u) return;

    float total = 0.0f;
    for (uint expert = 0u; expert < n_expert; expert++) {
        total += probability[expert];
    }
    for (uint expert = 0u; expert < n_expert; expert++) {
        probability[expert] /= total;
    }

    float chosen_weight[n_selected];
    float selected_total = 0.0f;
    for (uint slot = 0u; slot < n_selected; slot++) {
        chosen_weight[slot] = probability[chosen[slot]];
        selected_total += chosen_weight[slot];
    }
    for (uint slot = 0u; slot < n_selected; slot++) {
        *((device int32_t *)(selected +
            selected_base +
            (uint64_t)slot * args.selected_stride)) = (int32_t)chosen[slot];
        qwen35_metal_store_f32(
            selected_weight,
            selected_weight_base +
                (uint64_t)slot * args.selected_weight_stride,
            chosen_weight[slot] / selected_total);
    }
}

// Dequantizes one token-embedding row from GGUF Q8_0 blocks.  The standard
// encoding has block_size=32, a two-byte F16 scale, then 32 signed bytes.  All
// physical offsets and strides are explicit so padded rows remain valid.
kernel void kernel_qwen35_dequant_embedding_q8_0_f32(
        constant ds4_metal_args_qwen35_embedding_q8_0 &args [[buffer(0)]],
        device const char *embedding [[buffer(1)]],
        device       char *output    [[buffer(2)]],
        uint dim [[thread_position_in_grid]]) {
    if (dim >= args.n_embd || args.block_size != 32u) return;
    const uint block = dim / args.block_size;
    const uint within_block = dim - block * args.block_size;
    const uint64_t block_offset =
        (uint64_t)args.row_index * args.source_row_stride +
        (uint64_t)block * args.source_block_stride;
    const half scale = *((device const half *)(
        embedding + block_offset + args.source_scale_offset));
    const int8_t quant = *((device const int8_t *)(
        embedding + block_offset + args.source_quant_offset +
        (uint64_t)within_block * args.source_quant_stride));
    qwen35_metal_store_f32(
        output, (uint64_t)dim * args.output_dim_stride,
        (float)scale * (float)quant);
}

// Batched embedding gather.  Token IDs remain device-owned so one prompt
// chunk needs one dispatch instead of one model-range binding per row.
kernel void kernel_qwen35_dequant_embedding_q8_0_batch_f32(
        constant ds4_metal_args_qwen35_embedding_q8_0_batch &args
            [[buffer(0)]],
        device const char *embedding [[buffer(1)]],
        device const char *token_ids [[buffer(2)]],
        device       char *output    [[buffer(3)]],
        uint gid32 [[thread_position_in_grid]]) {
    const uint64_t gid = (uint64_t)gid32;
    const uint64_t total =
        (uint64_t)args.n_token * (uint64_t)args.n_embd;
    if (gid >= total || args.n_embd == 0u || args.block_size != 32u) return;

    const uint64_t token = gid / (uint64_t)args.n_embd;
    const uint dim = (uint)(gid - token * (uint64_t)args.n_embd);
    const int32_t row_index = *((device const int32_t *)(
        token_ids + token * args.token_id_stride));
    if (row_index < 0 || (uint32_t)row_index >= args.n_row) return;

    const uint block = dim / args.block_size;
    const uint within_block = dim - block * args.block_size;
    const uint64_t block_offset =
        (uint64_t)(uint32_t)row_index * args.source_row_stride +
        (uint64_t)block * args.source_block_stride;
    const half scale = *((device const half *)(
        embedding + block_offset + args.source_scale_offset));
    const int8_t quant = *((device const int8_t *)(
        embedding + block_offset + args.source_quant_offset +
        (uint64_t)within_block * args.source_quant_stride));
    qwen35_metal_store_f32(
        output,
        token * args.output_token_stride +
            (uint64_t)dim * args.output_dim_stride,
        (float)scale * (float)quant);
}

// Gated DeltaNet control transform.  GGUF stores ssm_a as -exp(A_log), so it
// is multiplied directly by the positive softplus timestep.  A one-token
// decode is the n_token=1 special case of the same packed-row contract.
kernel void kernel_qwen35_gated_delta_controls_f32(
        constant ds4_metal_args_qwen35_gated_delta_controls &args
            [[buffer(0)]],
        device const char *alpha_logit [[buffer(1)]],
        device const char *beta_logit  [[buffer(2)]],
        device const char *ssm_a       [[buffer(3)]],
        device const char *dt_bias     [[buffer(4)]],
        device       char *log_decay   [[buffer(5)]],
        device       char *beta        [[buffer(6)]],
        uint gid [[thread_position_in_grid]]) {
    const uint64_t total =
        (uint64_t)args.n_token * (uint64_t)args.n_value_head;
    if ((uint64_t)gid >= total || args.n_value_head == 0u) return;
    const uint token = gid / args.n_value_head;
    const uint head = gid - token * args.n_value_head;
    const float alpha = qwen35_metal_load_f32(
        alpha_logit,
        (uint64_t)token * args.alpha_logit_token_stride +
            (uint64_t)head * args.alpha_logit_head_stride);
    const float beta_value = qwen35_metal_sigmoid(qwen35_metal_load_f32(
        beta_logit,
        (uint64_t)token * args.beta_logit_token_stride +
            (uint64_t)head * args.beta_logit_head_stride));
    const float a = qwen35_metal_load_f32(
        ssm_a, (uint64_t)head * args.ssm_a_head_stride);
    const float bias = qwen35_metal_load_f32(
        dt_bias, (uint64_t)head * args.dt_bias_head_stride);
    qwen35_metal_store_f32(
        log_decay,
        (uint64_t)token * args.log_decay_token_stride +
            (uint64_t)head * args.log_decay_head_stride,
        a * qwen35_metal_softplus(alpha + bias));
    qwen35_metal_store_f32(
        beta,
        (uint64_t)token * args.beta_token_stride +
            (uint64_t)head * args.beta_head_stride,
        beta_value);
}

// Projection layout is [token][head][Q then gate].  One grid thread copies one
// element to each output.  Dispatch at least n_token*n_query_head*head_dim
// threads; surplus threads are ignored.
kernel void kernel_qwen35_split_q_gate_f32(
        constant ds4_metal_args_qwen35_split_q_gate &args [[buffer(0)]],
        device const char *projection [[buffer(1)]],
        device       char *query      [[buffer(2)]],
        device       char *gate       [[buffer(3)]],
        uint gid32 [[thread_position_in_grid]]) {
    const uint64_t gid = (uint64_t)gid32;
    const uint64_t head_values =
        (uint64_t)args.n_query_head * (uint64_t)args.head_dim;
    const uint64_t total = (uint64_t)args.n_token * head_values;
    if (gid >= total || args.head_dim == 0 || args.n_query_head == 0) return;

    const uint64_t token = gid / head_values;
    const uint64_t within_token = gid - token * head_values;
    const uint64_t head = within_token / (uint64_t)args.head_dim;
    const uint64_t dim = within_token - head * (uint64_t)args.head_dim;

    const uint64_t projection_base =
        token * args.projection_token_stride +
        head * args.projection_head_stride;
    const uint64_t query_offset =
        token * args.query_token_stride + head * args.query_head_stride +
        dim * args.query_dim_stride;
    const uint64_t gate_offset =
        token * args.gate_token_stride + head * args.gate_head_stride +
        dim * args.gate_dim_stride;
    const float q = qwen35_metal_load_f32(
        projection, projection_base + dim * args.projection_dim_stride);
    const float g = qwen35_metal_load_f32(
        projection,
        projection_base +
            ((uint64_t)args.head_dim + dim) * args.projection_dim_stride);
    qwen35_metal_store_f32(query, query_offset, q);
    qwen35_metal_store_f32(gate, gate_offset, g);
}

// Elementwise output = input * sigmoid(gate_logit).  This is the full-
// attention output gate; shared-expert scalar gating remains a separate graph
// operation because its gate is broadcast over a row.
kernel void kernel_qwen35_sigmoid_mul_f32(
        constant ds4_metal_args_qwen35_sigmoid_mul &args [[buffer(0)]],
        device const char *input      [[buffer(1)]],
        device const char *gate_logit [[buffer(2)]],
        device       char *output     [[buffer(3)]],
        uint gid32 [[thread_position_in_grid]]) {
    const uint64_t gid = (uint64_t)gid32;
    if (gid >= args.n_value) return;
    const float x = qwen35_metal_load_f32(input, gid * args.input_stride);
    const float z = qwen35_metal_load_f32(
        gate_logit, gid * args.gate_stride);
    qwen35_metal_store_f32(
        output, gid * args.output_stride, x * qwen35_metal_sigmoid(z));
}

// Row-wise scalar form used by the shared-expert gate during batched prefill.
// Each row owns one gate logit and row_width activation values.
kernel void kernel_qwen35_sigmoid_mul_rows_f32(
        constant ds4_metal_args_qwen35_sigmoid_mul_rows &args [[buffer(0)]],
        device const char *input      [[buffer(1)]],
        device const char *gate_logit [[buffer(2)]],
        device       char *output     [[buffer(3)]],
        uint gid32 [[thread_position_in_grid]]) {
    const uint64_t gid = (uint64_t)gid32;
    const uint64_t total =
        (uint64_t)args.n_row * (uint64_t)args.row_width;
    if (gid >= total || args.row_width == 0u) return;

    const uint64_t row = gid / (uint64_t)args.row_width;
    const uint64_t dim = gid - row * (uint64_t)args.row_width;
    const float x = qwen35_metal_load_f32(
        input,
        row * args.input_row_stride + dim * args.input_dim_stride);
    const float z = qwen35_metal_load_f32(
        gate_logit, row * args.gate_row_stride);
    qwen35_metal_store_f32(
        output,
        row * args.output_row_stride + dim * args.output_dim_stride,
        x * qwen35_metal_sigmoid(z));
}

// Split-half NeoX RoPE over the first n_rot dimensions.  Each first-half
// thread owns and writes one pair, which makes exact in-place operation safe.
// Dimensions [n_rot, head_dim) are copied for the out-of-place case.
kernel void kernel_qwen35_rope_prefix_f32(
        constant ds4_metal_args_qwen35_rope &args [[buffer(0)]],
        device const char     *source   [[buffer(1)]],
        device const uint8_t  *position [[buffer(2)]],
        device       char     *output   [[buffer(3)]],
        uint gid32 [[thread_position_in_grid]]) {
    const uint64_t gid = (uint64_t)gid32;
    const uint64_t head_values =
        (uint64_t)args.n_head * (uint64_t)args.head_dim;
    const uint64_t total = (uint64_t)args.n_token * head_values;
    if (gid >= total || args.n_head == 0 || args.head_dim == 0 ||
        args.n_rot == 0 || args.n_rot > args.head_dim ||
        (args.n_rot & 1u) != 0 || !(args.theta > 0.0f)) {
        return;
    }

    const uint64_t token = gid / head_values;
    const uint64_t within_token = gid - token * head_values;
    const uint64_t head = within_token / (uint64_t)args.head_dim;
    const uint64_t dim = within_token - head * (uint64_t)args.head_dim;
    const uint64_t source_base =
        token * args.source_token_stride + head * args.source_head_stride;
    const uint64_t output_base =
        token * args.output_token_stride + head * args.output_head_stride;

    if (dim >= (uint64_t)args.n_rot) {
        const float x = qwen35_metal_load_f32(
            source, source_base + dim * args.source_dim_stride);
        qwen35_metal_store_f32(
            output, output_base + dim * args.output_dim_stride, x);
        return;
    }

    const uint64_t half_rot = (uint64_t)args.n_rot / 2u;
    if (dim >= half_rot) return;

    const float exponent =
        (2.0f * (float)dim) / (float)args.n_rot;
    const uint32_t token_position = *((device const uint32_t *)(
        position + token * args.position_stride));
    const float angle =
        (float)token_position / pow(args.theta, exponent);
    const float cosine = cos(angle);
    const float sine = sin(angle);
    const float a = qwen35_metal_load_f32(
        source, source_base + dim * args.source_dim_stride);
    const float b = qwen35_metal_load_f32(
        source, source_base + (dim + half_rot) * args.source_dim_stride);
    qwen35_metal_store_f32(
        output, output_base + dim * args.output_dim_stride,
        a * cosine - b * sine);
    qwen35_metal_store_f32(
        output, output_base + (dim + half_rot) * args.output_dim_stride,
        b * cosine + a * sine);
}

// One-token causal depthwise convolution with SiLU.  State is
// [channel][kernel-1] in chronological order and is advanced in place.
kernel void kernel_qwen35_causal_conv_step_f32(
        constant ds4_metal_args_qwen35_conv_step &args [[buffer(0)]],
        device const char *input  [[buffer(1)]],
        device const char *weight [[buffer(2)]],
        device       char *state  [[buffer(3)]],
        device       char *output [[buffer(4)]],
        uint channel32 [[thread_position_in_grid]]) {
    const uint64_t channel = (uint64_t)channel32;
    if (channel >= (uint64_t)args.n_channel || args.kernel_size < 2u) return;

    const uint64_t state_base = channel * args.state_channel_stride;
    const uint64_t weight_base = channel * args.weight_channel_stride;
    const float current = qwen35_metal_load_f32(
        input, channel * args.input_channel_stride);
    float total = current * qwen35_metal_load_f32(
        weight, weight_base +
            ((uint64_t)args.kernel_size - 1u) * args.weight_tap_stride);

    const uint64_t history_len = (uint64_t)args.kernel_size - 1u;
    for (uint64_t tap = 0; tap < history_len; tap++) {
        total += qwen35_metal_load_f32(
                     state, state_base + tap * args.state_tap_stride) *
                 qwen35_metal_load_f32(
                     weight, weight_base + tap * args.weight_tap_stride);
    }
    qwen35_metal_store_f32(
        output, channel * args.output_channel_stride,
        qwen35_metal_silu(total));

    for (uint64_t tap = 0; tap + 1u < history_len; tap++) {
        qwen35_metal_store_f32(
            state, state_base + tap * args.state_tap_stride,
            qwen35_metal_load_f32(
                state, state_base + (tap + 1u) * args.state_tap_stride));
    }
    qwen35_metal_store_f32(
        state, state_base + (history_len - 1u) * args.state_tap_stride,
        current);
}

// Layer-major prefill form.  One GPU thread owns a depthwise channel, keeps
// its three-value history in registers, and advances every token in causal
// order before committing the final state once.  Qwen's supported kernel is
// fixed at four taps, but explicit strides keep the activation layout clear.
kernel void kernel_qwen35_causal_conv_sequence_f32(
        constant ds4_metal_args_qwen35_conv_sequence &args [[buffer(0)]],
        device const char *input  [[buffer(1)]],
        device const char *weight [[buffer(2)]],
        device       char *state  [[buffer(3)]],
        device       char *output [[buffer(4)]],
        uint channel [[thread_position_in_grid]]) {
    if (channel >= args.n_channel || args.n_token == 0u ||
        args.kernel_size != 4u) {
        return;
    }

    const uint64_t state_base =
        (uint64_t)channel * args.state_channel_stride;
    const uint64_t weight_base =
        (uint64_t)channel * args.weight_channel_stride;
    float history[3];
    for (uint tap = 0u; tap < 3u; tap++) {
        history[tap] = qwen35_metal_load_f32(
            state, state_base + (uint64_t)tap * args.state_tap_stride);
    }

    for (uint token = 0u; token < args.n_token; token++) {
        const uint64_t input_offset =
            (uint64_t)token * args.input_token_stride +
            (uint64_t)channel * args.input_channel_stride;
        const float current = qwen35_metal_load_f32(input, input_offset);
        float total = current * qwen35_metal_load_f32(
            weight, weight_base + 3u * args.weight_tap_stride);
        for (uint tap = 0u; tap < 3u; tap++) {
            total += history[tap] * qwen35_metal_load_f32(
                weight,
                weight_base + (uint64_t)tap * args.weight_tap_stride);
        }
        qwen35_metal_store_f32(
            output,
            (uint64_t)token * args.output_token_stride +
                (uint64_t)channel * args.output_channel_stride,
            qwen35_metal_silu(total));
        history[0] = history[1];
        history[1] = history[2];
        history[2] = current;
    }

    for (uint tap = 0u; tap < 3u; tap++) {
        qwen35_metal_store_f32(
            state,
            state_base + (uint64_t)tap * args.state_tap_stride,
            history[tap]);
    }
}

// One threadgroup updates one value head for one decode token.  The modulo
// key-head mapping is the post-GGUF V-head tiling contract, not contiguous GQA.
// State layout is [value_head][value_dim][key_dim].
kernel void kernel_qwen35_gated_delta_step_f32(
        constant ds4_metal_args_qwen35_gated_delta_step &args [[buffer(0)]],
        device const char *query      [[buffer(1)]],
        device const char *key        [[buffer(2)]],
        device const char *value      [[buffer(3)]],
        device const char *log_decay  [[buffer(4)]],
        device const char *beta       [[buffer(5)]],
        device       char *state      [[buffer(6)]],
        device       char *output     [[buffer(7)]],
        threadgroup float *scratch    [[threadgroup(0)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort3 thread_pos [[thread_position_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simdgroup [[simdgroup_index_in_threadgroup]],
        ushort simd_width [[threads_per_simdgroup]],
        ushort3 threads [[threads_per_threadgroup]]) {
    const uint value_head = group.x;
    const uint tid = thread_pos.x;
    if (value_head >= args.n_value_head || args.n_key_head == 0 ||
        args.key_dim == 0 || args.value_dim == 0 ||
        args.n_value_head % args.n_key_head != 0) {
        return;
    }

    const uint n_simdgroup =
        (threads.x + simd_width - 1u) / simd_width;
    // Two reductions plus two final inverse norms.  Host allocates
    // 2*n_simdgroup floats; DS4 dispatches no more than 32 SIMD groups.
    threadgroup float *query_partial = scratch;
    threadgroup float *key_partial = scratch + n_simdgroup;
    const uint key_head = value_head % args.n_key_head;
    const uint64_t query_base =
        (uint64_t)key_head * args.query_head_stride;
    const uint64_t key_base =
        (uint64_t)key_head * args.key_head_stride;

    float query_square = 0.0f;
    float key_square = 0.0f;
    for (uint dim = tid; dim < args.key_dim; dim += threads.x) {
        const float q = qwen35_metal_load_f32(
            query, query_base + (uint64_t)dim * args.query_dim_stride);
        const float k = qwen35_metal_load_f32(
            key, key_base + (uint64_t)dim * args.key_dim_stride);
        query_square += q * q;
        key_square += k * k;
    }
    query_square = simd_sum(query_square);
    key_square = simd_sum(key_square);
    if (lane == 0u) {
        query_partial[simdgroup] = query_square;
        key_partial[simdgroup] = key_square;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0u) {
        float q = lane < n_simdgroup ? query_partial[lane] : 0.0f;
        float k = lane < n_simdgroup ? key_partial[lane] : 0.0f;
        q = simd_sum(q);
        k = simd_sum(k);
        if (lane == 0u) {
            query_partial[0] =
                (1.0f / sqrt((float)args.key_dim)) /
                sqrt(q + 1.0e-6f);
            key_partial[0] = 1.0f / sqrt(k + 1.0e-6f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float query_inverse = query_partial[0];
    const float key_inverse = key_partial[0];
    const float decay = exp(qwen35_metal_load_f32(
        log_decay,
        (uint64_t)value_head * args.log_decay_head_stride));
    const float step = qwen35_metal_load_f32(
        beta, (uint64_t)value_head * args.beta_head_stride);
    const uint64_t value_base =
        (uint64_t)value_head * args.value_head_stride;
    const uint64_t state_head_base =
        (uint64_t)value_head * args.state_head_stride;
    const uint64_t output_base =
        (uint64_t)value_head * args.output_head_stride;

    for (uint value_dim = tid; value_dim < args.value_dim;
         value_dim += threads.x) {
        const uint64_t state_row =
            state_head_base +
            (uint64_t)value_dim * args.state_value_stride;
        float memory = 0.0f;
        for (uint key_dim = 0; key_dim < args.key_dim; key_dim++) {
            const uint64_t state_offset =
                state_row + (uint64_t)key_dim * args.state_key_stride;
            const float decayed =
                qwen35_metal_load_f32(state, state_offset) * decay;
            qwen35_metal_store_f32(state, state_offset, decayed);
            const float normalized_key =
                qwen35_metal_load_f32(
                    key,
                    key_base +
                        (uint64_t)key_dim * args.key_dim_stride) *
                key_inverse;
            memory += decayed * normalized_key;
        }

        const float target = qwen35_metal_load_f32(
            value,
            value_base +
                (uint64_t)value_dim * args.value_dim_stride);
        const float delta = (target - memory) * step;
        float result = 0.0f;
        for (uint key_dim = 0; key_dim < args.key_dim; key_dim++) {
            const uint64_t state_offset =
                state_row + (uint64_t)key_dim * args.state_key_stride;
            const float normalized_key =
                qwen35_metal_load_f32(
                    key,
                    key_base +
                        (uint64_t)key_dim * args.key_dim_stride) *
                key_inverse;
            const float updated =
                qwen35_metal_load_f32(state, state_offset) +
                normalized_key * delta;
            qwen35_metal_store_f32(state, state_offset, updated);
            const float normalized_query =
                qwen35_metal_load_f32(
                    query,
                    query_base +
                        (uint64_t)key_dim * args.query_dim_stride) *
                query_inverse;
            result += updated * normalized_query;
        }
        qwen35_metal_store_f32(
            output,
            output_base +
                (uint64_t)value_dim * args.output_dim_stride,
            result);
    }
}

// Autoregressive Qwen3.6 specializes key_dim=128.  The generic kernel above
// assigns one thread to each value row and walks the state row twice.  That
// exposes only n_value_head threadgroups and reads/writes the 128x128 state
// twice.  This path assigns one SIMD group to a value row, keeps four adjacent
// state cells per lane in registers, and commits each cell once.  A 32x4
// threadgroup therefore advances four rows and exposes 32 row blocks per value
// head to the GPU.  Query/key norms are calculated once per row block in the
// first SIMD group; their small repeated cost avoids another dispatch and
// scratch tensor on the decode timeline.
kernel void kernel_qwen35_gated_delta_step_128_f32(
        constant ds4_metal_args_qwen35_gated_delta_step &args [[buffer(0)]],
        device const char *query      [[buffer(1)]],
        device const char *key        [[buffer(2)]],
        device const char *value      [[buffer(3)]],
        device const char *log_decay  [[buffer(4)]],
        device const char *beta       [[buffer(5)]],
        device       char *state      [[buffer(6)]],
        device       char *output     [[buffer(7)]],
        threadgroup float *norm       [[threadgroup(0)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort3 thread_pos [[thread_position_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    constexpr uint dims_per_lane = 4u;
    constexpr uint rows_per_group = 4u;
    const uint value_head = group.y;
    const uint row_in_group = thread_pos.y;
    if (value_head >= args.n_value_head || args.n_key_head == 0u ||
        args.key_dim != 128u || args.value_dim == 0u ||
        args.n_value_head % args.n_key_head != 0u) {
        return;
    }

    const uint key_head = value_head % args.n_key_head;
    const uint64_t query_base =
        (uint64_t)key_head * args.query_head_stride;
    const uint64_t key_base =
        (uint64_t)key_head * args.key_head_stride;

    if (row_in_group == 0u) {
        float query_square = 0.0f;
        float key_square = 0.0f;
        for (uint j = 0u; j < dims_per_lane; j++) {
            const uint dim = (uint)lane * dims_per_lane + j;
            const float q = qwen35_metal_load_f32(
                query,
                query_base + (uint64_t)dim * args.query_dim_stride);
            const float k = qwen35_metal_load_f32(
                key,
                key_base + (uint64_t)dim * args.key_dim_stride);
            query_square += q * q;
            key_square += k * k;
        }
        query_square = simd_sum(query_square);
        key_square = simd_sum(key_square);
        if (lane == 0u) {
            norm[0] = (1.0f / sqrt((float)args.key_dim)) /
                      sqrt(query_square + 1.0e-6f);
            norm[1] = 1.0f / sqrt(key_square + 1.0e-6f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint value_dim = group.x * rows_per_group + row_in_group;
    if (value_dim >= args.value_dim) return;

    const float query_inverse = norm[0];
    const float key_inverse = norm[1];
    const float decay = exp(qwen35_metal_load_f32(
        log_decay,
        (uint64_t)value_head * args.log_decay_head_stride));
    const float step = qwen35_metal_load_f32(
        beta,
        (uint64_t)value_head * args.beta_head_stride);
    const uint64_t state_row =
        (uint64_t)value_head * args.state_head_stride +
        (uint64_t)value_dim * args.state_value_stride;

    float state_value[dims_per_lane];
    float normalized_key[dims_per_lane];
    float normalized_query[dims_per_lane];
    float memory_partial = 0.0f;
    for (uint j = 0u; j < dims_per_lane; j++) {
        const uint dim = (uint)lane * dims_per_lane + j;
        const uint64_t state_offset =
            state_row + (uint64_t)dim * args.state_key_stride;
        normalized_key[j] = qwen35_metal_load_f32(
            key,
            key_base + (uint64_t)dim * args.key_dim_stride) * key_inverse;
        normalized_query[j] = qwen35_metal_load_f32(
            query,
            query_base + (uint64_t)dim * args.query_dim_stride) *
            query_inverse;
        state_value[j] = qwen35_metal_load_f32(state, state_offset) * decay;
        memory_partial += state_value[j] * normalized_key[j];
    }

    const float memory = simd_sum(memory_partial);
    const uint64_t value_offset =
        (uint64_t)value_head * args.value_head_stride +
        (uint64_t)value_dim * args.value_dim_stride;
    const float target = qwen35_metal_load_f32(value, value_offset);
    const float delta = (target - memory) * step;
    float result_partial = 0.0f;
    for (uint j = 0u; j < dims_per_lane; j++) {
        const uint dim = (uint)lane * dims_per_lane + j;
        const uint64_t state_offset =
            state_row + (uint64_t)dim * args.state_key_stride;
        state_value[j] += normalized_key[j] * delta;
        qwen35_metal_store_f32(state, state_offset, state_value[j]);
        result_partial += state_value[j] * normalized_query[j];
    }

    const float result = simd_sum(result_partial);
    if (lane == 0u) {
        const uint64_t output_offset =
            (uint64_t)value_head * args.output_head_stride +
            (uint64_t)value_dim * args.output_dim_stride;
        qwen35_metal_store_f32(output, output_offset, result);
    }
}

// Prompt-sequence counterpart of the 128-wide decode kernel.  Each SIMD group
// owns one recurrent value row and retains its 128 state cells in registers
// across the complete token chunk.  Tokens remain strictly serial inside the
// kernel, while all value rows and heads stay parallel.  This removes one
// state read/write round trip per prompt token without changing the recurrence.
kernel void kernel_qwen35_gated_delta_sequence_128_f32(
        constant ds4_metal_args_qwen35_gated_delta_sequence &args
            [[buffer(0)]],
        device const char *projection [[buffer(1)]],
        device const char *log_decay  [[buffer(2)]],
        device const char *beta       [[buffer(3)]],
        device       char *state      [[buffer(4)]],
        device       char *output     [[buffer(5)]],
        threadgroup float *norm       [[threadgroup(0)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort3 thread_pos [[thread_position_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    constexpr uint dims_per_lane = 4u;
    constexpr uint rows_per_group = 4u;
    const uint value_head = group.y;
    const uint row_in_group = thread_pos.y;
    if (value_head >= args.n_value_head || args.n_token == 0u ||
        args.n_key_head == 0u || args.key_dim != 128u ||
        args.value_dim != 128u ||
        args.n_value_head % args.n_key_head != 0u) {
        return;
    }

    const uint value_dim = group.x * rows_per_group + row_in_group;
    const bool active_row = value_dim < args.value_dim;
    const uint key_head = value_head % args.n_key_head;
    const uint64_t state_row =
        (uint64_t)value_head * args.state_head_stride +
        (uint64_t)value_dim * args.state_value_stride;

    float state_value[dims_per_lane] = { 0.0f, 0.0f, 0.0f, 0.0f };
    if (active_row) {
        for (uint j = 0u; j < dims_per_lane; j++) {
            const uint dim = (uint)lane * dims_per_lane + j;
            state_value[j] = qwen35_metal_load_f32(
                state,
                state_row + (uint64_t)dim * args.state_key_stride);
        }
    }

    for (uint token = 0u; token < args.n_token; token++) {
        const uint64_t projection_base =
            (uint64_t)token * args.projection_token_stride;
        const uint64_t query_base =
            projection_base + args.query_offset +
            (uint64_t)key_head * args.query_head_stride;
        const uint64_t key_base =
            projection_base + args.key_offset +
            (uint64_t)key_head * args.key_head_stride;

        if (row_in_group == 0u) {
            float query_square = 0.0f;
            float key_square = 0.0f;
            for (uint j = 0u; j < dims_per_lane; j++) {
                const uint dim = (uint)lane * dims_per_lane + j;
                const float q = qwen35_metal_load_f32(
                    projection,
                    query_base + (uint64_t)dim * args.query_dim_stride);
                const float k = qwen35_metal_load_f32(
                    projection,
                    key_base + (uint64_t)dim * args.key_dim_stride);
                query_square += q * q;
                key_square += k * k;
            }
            query_square = simd_sum(query_square);
            key_square = simd_sum(key_square);
            if (lane == 0u) {
                norm[0] = (1.0f / sqrt((float)args.key_dim)) /
                          sqrt(query_square + 1.0e-6f);
                norm[1] = 1.0f / sqrt(key_square + 1.0e-6f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active_row) {
            const float query_inverse = norm[0];
            const float key_inverse = norm[1];
            const float decay = exp(qwen35_metal_load_f32(
                log_decay,
                (uint64_t)token * args.log_decay_token_stride +
                    (uint64_t)value_head * args.log_decay_head_stride));
            const float step = qwen35_metal_load_f32(
                beta,
                (uint64_t)token * args.beta_token_stride +
                    (uint64_t)value_head * args.beta_head_stride);

            float normalized_key[dims_per_lane];
            float normalized_query[dims_per_lane];
            float memory_partial = 0.0f;
            for (uint j = 0u; j < dims_per_lane; j++) {
                const uint dim = (uint)lane * dims_per_lane + j;
                normalized_key[j] = qwen35_metal_load_f32(
                    projection,
                    key_base + (uint64_t)dim * args.key_dim_stride) *
                    key_inverse;
                normalized_query[j] = qwen35_metal_load_f32(
                    projection,
                    query_base + (uint64_t)dim * args.query_dim_stride) *
                    query_inverse;
                state_value[j] *= decay;
                memory_partial += state_value[j] * normalized_key[j];
            }

            const float memory = simd_sum(memory_partial);
            const float target = qwen35_metal_load_f32(
                projection,
                projection_base + args.value_offset +
                    (uint64_t)value_head * args.value_head_stride +
                    (uint64_t)value_dim * args.value_dim_stride);
            const float delta = (target - memory) * step;
            float result_partial = 0.0f;
            for (uint j = 0u; j < dims_per_lane; j++) {
                state_value[j] += normalized_key[j] * delta;
                result_partial += state_value[j] * normalized_query[j];
            }
            const float result = simd_sum(result_partial);
            if (lane == 0u) {
                qwen35_metal_store_f32(
                    output,
                    (uint64_t)token * args.output_token_stride +
                        (uint64_t)value_head * args.output_head_stride +
                        (uint64_t)value_dim * args.output_dim_stride,
                    result);
            }
        }

        // Every row must finish consuming the shared norm before row zero
        // publishes the following token's values.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active_row) {
        for (uint j = 0u; j < dims_per_lane; j++) {
            const uint dim = (uint)lane * dims_per_lane + j;
            qwen35_metal_store_f32(
                state,
                state_row + (uint64_t)dim * args.state_key_stride,
                state_value[j]);
        }
    }
}

// One-token causal grouped-query attention over separate F32 K/V caches with
// layout [token][kv_head][head_dim].  One threadgroup owns one query head.  A
// stable online softmax avoids an O(context) score buffer while preserving the
// ordinary GQA mapping: contiguous query-head groups share each KV head.
//
// Dispatch requirements:
//   grid = n_query_head threadgroups
//   threads_per_threadgroup >= head_dim
//   scratch = n_simdgroup + 4 floats
kernel void kernel_qwen35_gqa_decode_f32(
        constant ds4_metal_args_qwen35_gqa_decode &args [[buffer(0)]],
        device const char *query       [[buffer(1)]],
        device const char *key_cache   [[buffer(2)]],
        device const char *value_cache [[buffer(3)]],
        device       char *output      [[buffer(4)]],
        threadgroup float *scratch     [[threadgroup(0)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort3 thread_pos [[thread_position_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simdgroup [[simdgroup_index_in_threadgroup]],
        ushort simd_width [[threads_per_simdgroup]],
        ushort3 threads [[threads_per_threadgroup]]) {
    const uint query_head = group.x;
    const uint tid = thread_pos.x;
    if (query_head >= args.n_query_head || args.n_kv == 0u ||
        args.n_kv_head == 0u || args.head_dim == 0u ||
        args.n_query_head % args.n_kv_head != 0u ||
        threads.x < args.head_dim) {
        return;
    }

    const uint n_simdgroup =
        (threads.x + simd_width - 1u) / simd_width;
    if (n_simdgroup > simd_width) return;
    const uint control = n_simdgroup;
    const uint query_per_kv = args.n_query_head / args.n_kv_head;
    const uint kv_head = query_head / query_per_kv;
    const uint64_t query_base =
        (uint64_t)query_head * args.query_head_stride;
    const uint64_t output_base =
        (uint64_t)query_head * args.output_head_stride;
    const float scale = 1.0f / sqrt((float)args.head_dim);
    float accumulator = 0.0f;

    if (tid == 0u) {
        scratch[control + 0u] = -INFINITY;
        scratch[control + 1u] = 0.0f;
        scratch[control + 2u] = 0.0f;
        scratch[control + 3u] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint token = 0; token < args.n_kv; token++) {
        const uint64_t key_base =
            (uint64_t)token * args.key_token_stride +
            (uint64_t)kv_head * args.key_head_stride;
        float dot_product = 0.0f;
        for (uint dim = tid; dim < args.head_dim; dim += threads.x) {
            const float q = qwen35_metal_load_f32(
                query, query_base + (uint64_t)dim * args.query_dim_stride);
            const float k = qwen35_metal_load_f32(
                key_cache,
                key_base + (uint64_t)dim * args.key_dim_stride);
            dot_product += q * k;
        }
        dot_product = simd_sum(dot_product);
        if (lane == 0u) scratch[simdgroup] = dot_product;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simdgroup == 0u) {
            float dot = lane < n_simdgroup ? scratch[lane] : 0.0f;
            dot = simd_sum(dot);
            if (lane == 0u) {
                const float score = dot * scale;
                const float previous_max = scratch[control + 0u];
                const float next_max = max(previous_max, score);
                const float previous_factor =
                    exp(previous_max - next_max);
                const float current_factor = exp(score - next_max);
                scratch[control + 0u] = next_max;
                scratch[control + 1u] =
                    scratch[control + 1u] * previous_factor +
                    current_factor;
                scratch[control + 2u] = previous_factor;
                scratch[control + 3u] = current_factor;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < args.head_dim) {
            const uint64_t value_offset =
                (uint64_t)token * args.value_token_stride +
                (uint64_t)kv_head * args.value_head_stride +
                (uint64_t)tid * args.value_dim_stride;
            accumulator =
                accumulator * scratch[control + 2u] +
                qwen35_metal_load_f32(value_cache, value_offset) *
                    scratch[control + 3u];
        }
        // The next token reuses the partial-sum slots and online-softmax
        // factors, so every lane must finish consuming them first.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < args.head_dim) {
        qwen35_metal_store_f32(
            output, output_base + (uint64_t)tid * args.output_dim_stride,
            accumulator / scratch[control + 1u]);
    }
}

// Long-context decode variant.  The scalar-token kernel above uses the whole
// threadgroup for one K dot product and therefore crosses three threadgroup
// barriers for every cached token.  Here each SIMD group scans an independent
// strided slice of the cache, retaining its query and partial softmax output in
// registers.  One final threadgroup reduction merges the independently stable
// online-softmax states.  Qwen's 256-wide head maps to eight values per lane on
// Apple GPU SIMD32, so eight cache tokens advance concurrently with no barrier
// inside the context loop.
//
// Dispatch requirements:
//   grid = n_query_head threadgroups
//   threads_per_threadgroup >= head_dim
//   scratch = n_simdgroup * (head_dim + 2) + 1 floats
kernel void kernel_qwen35_gqa_decode_parallel_f32(
        constant ds4_metal_args_qwen35_gqa_decode &args [[buffer(0)]],
        device const char *query       [[buffer(1)]],
        device const char *key_cache   [[buffer(2)]],
        device const char *value_cache [[buffer(3)]],
        device       char *output      [[buffer(4)]],
        threadgroup float *scratch     [[threadgroup(0)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort3 thread_pos [[thread_position_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simdgroup [[simdgroup_index_in_threadgroup]],
        ushort simd_width [[threads_per_simdgroup]],
        ushort3 threads [[threads_per_threadgroup]]) {
    constexpr uint max_dims_per_lane = 8u;
    const uint query_head = group.x;
    const uint tid = thread_pos.x;
    if (query_head >= args.n_query_head || args.n_kv == 0u ||
        args.n_kv_head == 0u || args.head_dim == 0u ||
        args.n_query_head % args.n_kv_head != 0u ||
        threads.x < args.head_dim || simd_width == 0u) {
        return;
    }

    const uint n_simdgroup =
        (threads.x + simd_width - 1u) / simd_width;
    const uint dims_per_lane =
        (args.head_dim + simd_width - 1u) / simd_width;
    if (n_simdgroup == 0u || n_simdgroup > simd_width ||
        dims_per_lane > max_dims_per_lane) {
        return;
    }

    const uint query_per_kv = args.n_query_head / args.n_kv_head;
    const uint kv_head = query_head / query_per_kv;
    const uint64_t query_base =
        (uint64_t)query_head * args.query_head_stride;
    const uint64_t output_base =
        (uint64_t)query_head * args.output_head_stride;
    const float scale = 1.0f / sqrt((float)args.head_dim);

    float query_lane[max_dims_per_lane];
    float accumulator[max_dims_per_lane];
    for (uint j = 0u; j < max_dims_per_lane; j++) {
        const uint dim = (uint)lane + j * (uint)simd_width;
        query_lane[j] = dim < args.head_dim
            ? qwen35_metal_load_f32(
                  query,
                  query_base + (uint64_t)dim * args.query_dim_stride)
            : 0.0f;
        accumulator[j] = 0.0f;
    }

    float local_max = -INFINITY;
    float local_sum = 0.0f;
    for (uint token = (uint)simdgroup;
         token < args.n_kv;
         token += n_simdgroup) {
        const uint64_t key_base =
            (uint64_t)token * args.key_token_stride +
            (uint64_t)kv_head * args.key_head_stride;
        float dot_product = 0.0f;
        for (uint j = 0u; j < max_dims_per_lane; j++) {
            const uint dim = (uint)lane + j * (uint)simd_width;
            if (dim < args.head_dim) {
                dot_product += query_lane[j] * qwen35_metal_load_f32(
                    key_cache,
                    key_base + (uint64_t)dim * args.key_dim_stride);
            }
        }
        const float score = simd_sum(dot_product) * scale;
        const float next_max = max(local_max, score);
        const float previous_factor = exp(local_max - next_max);
        const float current_factor = exp(score - next_max);
        local_sum = local_sum * previous_factor + current_factor;

        const uint64_t value_base =
            (uint64_t)token * args.value_token_stride +
            (uint64_t)kv_head * args.value_head_stride;
        for (uint j = 0u; j < max_dims_per_lane; j++) {
            const uint dim = (uint)lane + j * (uint)simd_width;
            if (dim < args.head_dim) {
                accumulator[j] =
                    accumulator[j] * previous_factor +
                    qwen35_metal_load_f32(
                        value_cache,
                        value_base +
                            (uint64_t)dim * args.value_dim_stride) *
                        current_factor;
            }
        }
        local_max = next_max;
    }

    const uint partial_values = n_simdgroup * args.head_dim;
    const uint max_base = partial_values;
    const uint sum_base = max_base + n_simdgroup;
    const uint denominator_index = sum_base + n_simdgroup;
    for (uint j = 0u; j < max_dims_per_lane; j++) {
        const uint dim = (uint)lane + j * (uint)simd_width;
        if (dim < args.head_dim) {
            scratch[(uint)simdgroup * args.head_dim + dim] = accumulator[j];
        }
    }
    if (lane == 0u) {
        scratch[max_base + (uint)simdgroup] = local_max;
        scratch[sum_base + (uint)simdgroup] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0u) {
        const float partial_max = lane < n_simdgroup
            ? scratch[max_base + (uint)lane]
            : -INFINITY;
        const float merged_max = simd_max(partial_max);
        const float merge_scale = lane < n_simdgroup
            ? exp(partial_max - merged_max)
            : 0.0f;
        const float partial_sum = lane < n_simdgroup
            ? scratch[sum_base + (uint)lane] * merge_scale
            : 0.0f;
        const float denominator = simd_sum(partial_sum);
        if (lane < n_simdgroup) {
            scratch[max_base + (uint)lane] = merge_scale;
        }
        if (lane == 0u) scratch[denominator_index] = denominator;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < args.head_dim) {
        float numerator = 0.0f;
        for (uint part = 0u; part < n_simdgroup; part++) {
            numerator += scratch[part * args.head_dim + tid] *
                         scratch[max_base + part];
        }
        qwen35_metal_store_f32(
            output,
            output_base + (uint64_t)tid * args.output_dim_stride,
            numerator / scratch[denominator_index]);
    }
}

// Layer-major causal GQA for a prompt chunk.  K/V rows for the complete chunk
// are written to the persistent cache before this dispatch; each query group
// limits its online-softmax scan to position0 + query_token + 1, so future
// rows remain invisible without allocating a quadratic mask.
kernel void kernel_qwen35_gqa_prefill_f32(
        constant ds4_metal_args_qwen35_gqa_prefill &args [[buffer(0)]],
        device const char *query       [[buffer(1)]],
        device const char *key_cache   [[buffer(2)]],
        device const char *value_cache [[buffer(3)]],
        device       char *output      [[buffer(4)]],
        threadgroup float *scratch     [[threadgroup(0)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort3 thread_pos [[thread_position_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simdgroup [[simdgroup_index_in_threadgroup]],
        ushort simd_width [[threads_per_simdgroup]],
        ushort3 threads [[threads_per_threadgroup]]) {
    const uint query_head = group.x;
    const uint query_token = group.y;
    const uint tid = thread_pos.x;
    if (query_head >= args.n_query_head || query_token >= args.n_token ||
        args.n_kv_head == 0u || args.head_dim == 0u ||
        args.n_query_head % args.n_kv_head != 0u ||
        threads.x < args.head_dim) {
        return;
    }

    const uint n_simdgroup =
        (threads.x + simd_width - 1u) / simd_width;
    if (n_simdgroup > simd_width) return;
    const uint control = n_simdgroup;
    const uint query_per_kv = args.n_query_head / args.n_kv_head;
    const uint kv_head = query_head / query_per_kv;
    const uint n_kv = args.position0 + query_token + 1u;
    const uint64_t query_base =
        (uint64_t)query_token * args.query_token_stride +
        (uint64_t)query_head * args.query_head_stride;
    const uint64_t output_base =
        (uint64_t)query_token * args.output_token_stride +
        (uint64_t)query_head * args.output_head_stride;
    const float scale = 1.0f / sqrt((float)args.head_dim);
    float accumulator = 0.0f;

    if (tid == 0u) {
        scratch[control + 0u] = -INFINITY;
        scratch[control + 1u] = 0.0f;
        scratch[control + 2u] = 0.0f;
        scratch[control + 3u] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint token = 0u; token < n_kv; token++) {
        const uint64_t key_base =
            (uint64_t)token * args.key_token_stride +
            (uint64_t)kv_head * args.key_head_stride;
        float dot_product = 0.0f;
        for (uint dim = tid; dim < args.head_dim; dim += threads.x) {
            const float q = qwen35_metal_load_f32(
                query,
                query_base + (uint64_t)dim * args.query_dim_stride);
            const float k = qwen35_metal_load_f32(
                key_cache,
                key_base + (uint64_t)dim * args.key_dim_stride);
            dot_product += q * k;
        }
        dot_product = simd_sum(dot_product);
        if (lane == 0u) scratch[simdgroup] = dot_product;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simdgroup == 0u) {
            float dot = lane < n_simdgroup ? scratch[lane] : 0.0f;
            dot = simd_sum(dot);
            if (lane == 0u) {
                const float score = dot * scale;
                const float previous_max = scratch[control + 0u];
                const float next_max = max(previous_max, score);
                const float previous_factor = exp(previous_max - next_max);
                const float current_factor = exp(score - next_max);
                scratch[control + 0u] = next_max;
                scratch[control + 1u] =
                    scratch[control + 1u] * previous_factor +
                    current_factor;
                scratch[control + 2u] = previous_factor;
                scratch[control + 3u] = current_factor;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < args.head_dim) {
            const uint64_t value_offset =
                (uint64_t)token * args.value_token_stride +
                (uint64_t)kv_head * args.value_head_stride +
                (uint64_t)tid * args.value_dim_stride;
            accumulator =
                accumulator * scratch[control + 2u] +
                qwen35_metal_load_f32(value_cache, value_offset) *
                    scratch[control + 3u];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < args.head_dim) {
        qwen35_metal_store_f32(
            output,
            output_base + (uint64_t)tid * args.output_dim_stride,
            accumulator / scratch[control + 1u]);
    }
}

// One threadgroup normalizes and gates one value-head row.  Scratch holds one
// partial sum per SIMD group; the host allocates n_simdgroup floats.
kernel void kernel_qwen35_rmsnorm_gated_f32(
        constant ds4_metal_args_qwen35_rmsnorm_gated &args [[buffer(0)]],
        device const char *input  [[buffer(1)]],
        device const char *gate   [[buffer(2)]],
        device const char *weight [[buffer(3)]],
        device       char *output [[buffer(4)]],
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort3 thread_pos [[thread_position_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simdgroup [[simdgroup_index_in_threadgroup]],
        ushort simd_width [[threads_per_simdgroup]],
        ushort3 threads [[threads_per_threadgroup]]) {
    const uint row = group.x;
    const uint tid = thread_pos.x;
    if (row >= args.n_vector || args.dim == 0 ||
        !(args.epsilon > 0.0f)) {
        return;
    }

    const uint n_simdgroup =
        (threads.x + simd_width - 1u) / simd_width;
    const uint64_t input_base =
        (uint64_t)row * args.input_vector_stride;
    float square = 0.0f;
    for (uint dim = tid; dim < args.dim; dim += threads.x) {
        const float x = qwen35_metal_load_f32(
            input, input_base + (uint64_t)dim * args.input_dim_stride);
        square += x * x;
    }
    square = simd_sum(square);
    if (lane == 0u) scratch[simdgroup] = square;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0u) {
        float total = lane < n_simdgroup ? scratch[lane] : 0.0f;
        total = simd_sum(total);
        if (lane == 0u) {
            scratch[0] =
                1.0f / sqrt(total / (float)args.dim + args.epsilon);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inverse_rms = scratch[0];
    const uint64_t gate_base =
        (uint64_t)row * args.gate_vector_stride;
    const uint64_t output_base =
        (uint64_t)row * args.output_vector_stride;
    for (uint dim = tid; dim < args.dim; dim += threads.x) {
        const float x = qwen35_metal_load_f32(
            input, input_base + (uint64_t)dim * args.input_dim_stride);
        const float z = qwen35_metal_load_f32(
            gate, gate_base + (uint64_t)dim * args.gate_dim_stride);
        const float w = qwen35_metal_load_f32(
            weight, (uint64_t)dim * args.weight_dim_stride);
        qwen35_metal_store_f32(
            output, output_base + (uint64_t)dim * args.output_dim_stride,
            x * inverse_rms * w * qwen35_metal_silu(z));
    }
}
