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

struct ds4_metal_args_qwen35_gated_delta_controls {
    uint32_t n_value_head;
    uint32_t reserved;
    uint64_t alpha_logit_head_stride;
    uint64_t beta_logit_head_stride;
    uint64_t ssm_a_head_stride;
    uint64_t dt_bias_head_stride;
    uint64_t log_decay_head_stride;
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

// One-token Gated DeltaNet control transform.  GGUF stores ssm_a as
// -exp(A_log), so it is multiplied directly by the positive softplus timestep.
kernel void kernel_qwen35_gated_delta_controls_f32(
        constant ds4_metal_args_qwen35_gated_delta_controls &args
            [[buffer(0)]],
        device const char *alpha_logit [[buffer(1)]],
        device const char *beta_logit  [[buffer(2)]],
        device const char *ssm_a       [[buffer(3)]],
        device const char *dt_bias     [[buffer(4)]],
        device       char *log_decay   [[buffer(5)]],
        device       char *beta        [[buffer(6)]],
        uint head [[thread_position_in_grid]]) {
    if (head >= args.n_value_head) return;
    const float alpha = qwen35_metal_load_f32(
        alpha_logit, (uint64_t)head * args.alpha_logit_head_stride);
    const float beta_value = qwen35_metal_sigmoid(qwen35_metal_load_f32(
        beta_logit, (uint64_t)head * args.beta_logit_head_stride));
    const float a = qwen35_metal_load_f32(
        ssm_a, (uint64_t)head * args.ssm_a_head_stride);
    const float bias = qwen35_metal_load_f32(
        dt_bias, (uint64_t)head * args.dt_bias_head_stride);
    qwen35_metal_store_f32(
        log_decay, (uint64_t)head * args.log_decay_head_stride,
        a * qwen35_metal_softplus(alpha + bias));
    qwen35_metal_store_f32(
        beta, (uint64_t)head * args.beta_head_stride, beta_value);
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
