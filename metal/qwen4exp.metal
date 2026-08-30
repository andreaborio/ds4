// Qwen4Exp Phase-5 resident correctness primitives.
//
// These kernels deliberately prefer bounded, serial reductions to throughput.
// All activations, controls, reductions, and persistent state are F32.  The
// layouts are contiguous and named at each entry point; production codec,
// sparse-QSA, SSD, and M5-specialized paths are intentionally absent.

#include <metal_stdlib>
using namespace metal;

enum : uint {
    DS4_Q4E_METAL_OK = 0u,
    DS4_Q4E_METAL_NONFINITE = 1u,
    DS4_Q4E_METAL_BAD_ARGUMENT = 2u,
    DS4_Q4E_METAL_BAD_INDEX = 3u,
};

struct ds4_metal_args_qwen4exp_embedding {
    uint n_token;
    uint n_row;
    uint n_stream;
    uint dim;
};

struct ds4_metal_args_qwen4exp_gr {
    uint  n_token;
    uint  n_stream;
    uint  dim;
    uint  rank;
    float epsilon;
    uint  reserved0;
    uint  reserved1;
    uint  reserved2;
};

struct ds4_metal_args_qwen4exp_conv {
    uint n_token;
    uint n_channel;
    uint kernel_size;
    uint dilation;
    uint n_sequence;
    uint reserved;
};

struct ds4_metal_args_qwen4exp_controls {
    uint n_token;
    uint n_value_head;
};

struct ds4_metal_args_qwen4exp_gdn {
    uint n_token;
    uint n_key_head;
    uint n_value_head;
    uint key_dim;
    uint value_dim;
    uint reserved;
};

struct ds4_metal_args_qwen4exp_router {
    uint n_token;
    uint n_expert;
    uint n_selected;
    uint reserved;
};

struct ds4_metal_args_qwen4exp_moe {
    uint n_token;
    uint n_expert;
    uint n_selected;
    uint input_dim;
    uint expert_dim;
    uint output_dim;
    uint reserved0;
    uint reserved1;
};

struct ds4_metal_args_qwen4exp_qsa_group {
    uint  n_group;
    uint  compression;
    uint  head_dim;
    uint  n_rot;
    float theta;
    float epsilon;
    uint  position0;
    uint  n_slot;
};

struct ds4_metal_args_qwen4exp_qsa_select {
    uint n_query;
    uint n_query_head;
    uint head_dim;
    uint n_visible_max;
    uint compression;
    uint group_budget;
    uint max_width;
    uint reserved;
};

struct ds4_metal_args_qwen4exp_qsa_attention {
    uint n_query;
    uint n_query_head;
    uint n_kv_head;
    uint head_dim;
    uint n_key;
    uint max_selected;
    uint reserved0;
    uint reserved1;
};

struct ds4_metal_args_qwen4exp_ple_gather {
    uint n_token;
    uint n_head;
    uint row_dim;
    uint n_row;
};

struct ds4_metal_args_qwen4exp_ple_gate {
    uint n_token;
    uint n_stream;
    uint dim;
    uint reserved;
};

struct ds4_metal_args_qwen4exp_head {
    uint n_token;
    uint input_dim;
    uint output_dim;
    uint reserved;
};

static_assert(sizeof(ds4_metal_args_qwen4exp_embedding) == 16,
              "qwen4exp embedding ABI drift");
static_assert(sizeof(ds4_metal_args_qwen4exp_gr) == 32,
              "qwen4exp GR ABI drift");
static_assert(sizeof(ds4_metal_args_qwen4exp_conv) == 24,
              "qwen4exp conv ABI drift");
static_assert(sizeof(ds4_metal_args_qwen4exp_controls) == 8,
              "qwen4exp controls ABI drift");
static_assert(sizeof(ds4_metal_args_qwen4exp_gdn) == 24,
              "qwen4exp GDN ABI drift");
static_assert(sizeof(ds4_metal_args_qwen4exp_router) == 16,
              "qwen4exp router ABI drift");
static_assert(sizeof(ds4_metal_args_qwen4exp_moe) == 32,
              "qwen4exp MoE ABI drift");
static_assert(sizeof(ds4_metal_args_qwen4exp_qsa_group) == 32,
              "qwen4exp QSA-group ABI drift");
static_assert(sizeof(ds4_metal_args_qwen4exp_qsa_select) == 32,
              "qwen4exp QSA-select ABI drift");
static_assert(sizeof(ds4_metal_args_qwen4exp_qsa_attention) == 32,
              "qwen4exp QSA-attention ABI drift");
static_assert(sizeof(ds4_metal_args_qwen4exp_ple_gather) == 16,
              "qwen4exp PLE-gather ABI drift");
static_assert(sizeof(ds4_metal_args_qwen4exp_ple_gate) == 16,
              "qwen4exp PLE-gate ABI drift");
static_assert(sizeof(ds4_metal_args_qwen4exp_head) == 16,
              "qwen4exp head ABI drift");

static inline float q4e_sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + exp(-x));
    const float e = exp(x);
    return e / (1.0f + e);
}

static inline float q4e_silu(float x) {
    return x * q4e_sigmoid(x);
}

static inline float q4e_softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return exp(x);
    return log(1.0f + exp(x));
}

// table [row][dim], token_id [token], residual [token][stream][dim].
kernel void kernel_qwen4exp_embedding_four_stream_f32(
        constant ds4_metal_args_qwen4exp_embedding &a [[buffer(0)]],
        device const uint  *token_id [[buffer(1)]],
        device const float *table    [[buffer(2)]],
        device       float *residual [[buffer(3)]],
        device       uint  *status   [[buffer(4)]],
        uint token [[thread_position_in_grid]]) {
    if (token >= a.n_token) return;
    if (a.n_stream == 0u || a.n_stream > 4u || a.dim == 0u ||
        token_id[token] >= a.n_row) {
        status[token] = DS4_Q4E_METAL_BAD_INDEX;
        return;
    }
    const uint row = token_id[token];
    for (uint i = 0u; i < a.dim; i++) {
        if (!isfinite(table[(ulong)row * a.dim + i])) {
            status[token] = DS4_Q4E_METAL_NONFINITE;
            return;
        }
    }
    for (uint stream = 0u; stream < a.n_stream; stream++) {
        for (uint i = 0u; i < a.dim; i++) {
            residual[((ulong)token * a.n_stream + stream) * a.dim + i] =
                table[(ulong)row * a.dim + i];
        }
    }
    status[token] = DS4_Q4E_METAL_OK;
}

// residual/norm [token][stream][dim], down [rank][stream*dim],
// up [stream*dim][rank], inject [stream][stream*dim], mixed [token][dim],
// injection [token][stream].  Both reductions divide by n_stream (four in the
// admitted profile): before SiLU and before the injection sigmoid.
kernel void kernel_qwen4exp_gr_prepare_f32(
        constant ds4_metal_args_qwen4exp_gr &a [[buffer(0)]],
        device const float *residual    [[buffer(1)]],
        device const float *norm_weight [[buffer(2)]],
        device const float *down        [[buffer(3)]],
        device const float *up          [[buffer(4)]],
        device const float *inject      [[buffer(5)]],
        device       float *mixed       [[buffer(6)]],
        device       float *injection   [[buffer(7)]],
        device       uint  *status      [[buffer(8)]],
        uint token [[thread_position_in_grid]]) {
    if (token >= a.n_token) return;
    const ulong wide = (ulong)a.n_stream * a.dim;
    if (a.n_stream == 0u || a.n_stream > 4u || a.dim == 0u ||
        a.rank == 0u || a.rank > 256u || !(a.epsilon > 0.0f) ||
        !isfinite(a.epsilon)) {
        status[token] = DS4_Q4E_METAL_BAD_ARGUMENT;
        return;
    }
    const ulong residual_base = (ulong)token * wide;
    for (ulong i = 0u; i < wide; i++) {
        if (!isfinite(residual[residual_base + i]) ||
            !isfinite(norm_weight[i])) {
            status[token] = DS4_Q4E_METAL_NONFINITE;
            return;
        }
    }
    for (ulong i = 0u; i < (ulong)a.rank * wide; i++) {
        if (!isfinite(down[i])) {
            status[token] = DS4_Q4E_METAL_NONFINITE;
            return;
        }
    }
    for (ulong i = 0u; i < wide * a.rank; i++) {
        if (!isfinite(up[i])) {
            status[token] = DS4_Q4E_METAL_NONFINITE;
            return;
        }
    }
    for (ulong i = 0u; i < (ulong)a.n_stream * wide; i++) {
        if (!isfinite(inject[i])) {
            status[token] = DS4_Q4E_METAL_NONFINITE;
            return;
        }
    }

    float stream_inverse[4];
    for (uint stream = 0u; stream < a.n_stream; stream++) {
        float sum_square = 0.0f;
        for (uint i = 0u; i < a.dim; i++) {
            const float x = residual[residual_base + (ulong)stream * a.dim + i];
            sum_square += x * x;
        }
        stream_inverse[stream] =
            1.0f / sqrt(sum_square / (float)a.dim + a.epsilon);
        if (!isfinite(stream_inverse[stream])) {
            status[token] = DS4_Q4E_METAL_NONFINITE;
            return;
        }
    }

    for (uint i = 0u; i < a.dim; i++) {
        float branch_sum = 0.0f;
        for (uint stream = 0u; stream < a.n_stream; stream++) {
            const ulong out_index = (ulong)stream * a.dim + i;
            float gate_logit = 0.0f;
            for (uint r = 0u; r < a.rank; r++) {
                float hidden_sum = 0.0f;
                for (ulong j = 0u; j < wide; j++) {
                    const uint source_stream = (uint)(j / a.dim);
                    const float normalized =
                        residual[residual_base + j] *
                        stream_inverse[source_stream] *
                        (1.0f + norm_weight[j]);
                    hidden_sum += down[(ulong)r * wide + j] * normalized;
                }
                gate_logit += up[out_index * a.rank + r] *
                    q4e_silu(hidden_sum / (float)a.n_stream);
            }
            const float normalized =
                residual[residual_base + out_index] * stream_inverse[stream] *
                (1.0f + norm_weight[out_index]);
            branch_sum += q4e_sigmoid(gate_logit) * normalized;
        }
        const float result = branch_sum / (float)a.n_stream;
        if (!isfinite(result)) {
            status[token] = DS4_Q4E_METAL_NONFINITE;
            return;
        }
    }
    for (uint stream = 0u; stream < a.n_stream; stream++) {
        float injection_sum = 0.0f;
        for (ulong j = 0u; j < wide; j++) {
            const uint source_stream = (uint)(j / a.dim);
            const float normalized = residual[residual_base + j] *
                stream_inverse[source_stream] * (1.0f + norm_weight[j]);
            injection_sum += inject[(ulong)stream * wide + j] * normalized;
        }
        const float result =
            2.0f * q4e_sigmoid(injection_sum / (float)a.n_stream);
        if (!isfinite(result)) {
            status[token] = DS4_Q4E_METAL_NONFINITE;
            return;
        }
    }

    // Recompute after the complete token preflight so output is transactional.
    for (uint i = 0u; i < a.dim; i++) {
        float branch_sum = 0.0f;
        for (uint stream = 0u; stream < a.n_stream; stream++) {
            const ulong out_index = (ulong)stream * a.dim + i;
            float gate_logit = 0.0f;
            for (uint r = 0u; r < a.rank; r++) {
                float hidden_sum = 0.0f;
                for (ulong j = 0u; j < wide; j++) {
                    const uint source_stream = (uint)(j / a.dim);
                    const float normalized = residual[residual_base + j] *
                        stream_inverse[source_stream] *
                        (1.0f + norm_weight[j]);
                    hidden_sum += down[(ulong)r * wide + j] * normalized;
                }
                gate_logit += up[out_index * a.rank + r] *
                    q4e_silu(hidden_sum / (float)a.n_stream);
            }
            const float normalized = residual[residual_base + out_index] *
                stream_inverse[stream] * (1.0f + norm_weight[out_index]);
            branch_sum += q4e_sigmoid(gate_logit) * normalized;
        }
        mixed[(ulong)token * a.dim + i] =
            branch_sum / (float)a.n_stream;
    }
    for (uint stream = 0u; stream < a.n_stream; stream++) {
        float injection_sum = 0.0f;
        for (ulong j = 0u; j < wide; j++) {
            const uint source_stream = (uint)(j / a.dim);
            const float normalized = residual[residual_base + j] *
                stream_inverse[source_stream] * (1.0f + norm_weight[j]);
            injection_sum += inject[(ulong)stream * wide + j] * normalized;
        }
        injection[(ulong)token * a.n_stream + stream] =
            2.0f * q4e_sigmoid(injection_sum / (float)a.n_stream);
    }
    status[token] = DS4_Q4E_METAL_OK;
}

// residual [token][stream][dim], block [token][dim], injection [token][stream].
kernel void kernel_qwen4exp_gr_apply_f32(
        constant ds4_metal_args_qwen4exp_gr &a [[buffer(0)]],
        device       float *residual  [[buffer(1)]],
        device const float *block     [[buffer(2)]],
        device const float *injection [[buffer(3)]],
        device       uint  *status    [[buffer(4)]],
        uint token [[thread_position_in_grid]]) {
    if (token >= a.n_token) return;
    if (a.n_stream == 0u || a.n_stream > 4u || a.dim == 0u) {
        status[token] = DS4_Q4E_METAL_BAD_ARGUMENT;
        return;
    }
    const ulong wide = (ulong)a.n_stream * a.dim;
    const ulong base = (ulong)token * wide;
    for (ulong i = 0u; i < wide; i++) {
        const uint stream = (uint)(i / a.dim);
        const float result = residual[base + i] +
            injection[(ulong)token * a.n_stream + stream] *
            block[(ulong)token * a.dim + i % a.dim];
        if (!isfinite(result)) {
            status[token] = DS4_Q4E_METAL_NONFINITE;
            return;
        }
    }
    for (ulong i = 0u; i < wide; i++) {
        const uint stream = (uint)(i / a.dim);
        residual[base + i] +=
            injection[(ulong)token * a.n_stream + stream] *
            block[(ulong)token * a.dim + i % a.dim];
    }
    status[token] = DS4_Q4E_METAL_OK;
}

// Same four-stream gated mixer as prepare, with no injection projection.
kernel void kernel_qwen4exp_gr_final_mix_f32(
        constant ds4_metal_args_qwen4exp_gr &a [[buffer(0)]],
        device const float *residual    [[buffer(1)]],
        device const float *norm_weight [[buffer(2)]],
        device const float *down        [[buffer(3)]],
        device const float *up          [[buffer(4)]],
        device       float *output      [[buffer(5)]],
        device       uint  *status      [[buffer(6)]],
        uint token [[thread_position_in_grid]]) {
    if (token >= a.n_token) return;
    const ulong wide = (ulong)a.n_stream * a.dim;
    if (a.n_stream == 0u || a.n_stream > 4u || a.dim == 0u ||
        a.rank == 0u || a.rank > 256u || !(a.epsilon > 0.0f) ||
        !isfinite(a.epsilon)) {
        status[token] = DS4_Q4E_METAL_BAD_ARGUMENT;
        return;
    }
    const ulong base = (ulong)token * wide;
    float inverse[4];
    for (uint stream = 0u; stream < a.n_stream; stream++) {
        float ss = 0.0f;
        for (uint i = 0u; i < a.dim; i++) {
            const float x = residual[base + (ulong)stream * a.dim + i];
            if (!isfinite(x) ||
                !isfinite(norm_weight[(ulong)stream * a.dim + i])) {
                status[token] = DS4_Q4E_METAL_NONFINITE;
                return;
            }
            ss += x * x;
        }
        inverse[stream] = 1.0f / sqrt(ss / (float)a.dim + a.epsilon);
    }
    for (ulong i = 0u; i < (ulong)a.rank * wide; i++) {
        if (!isfinite(down[i])) {
            status[token] = DS4_Q4E_METAL_NONFINITE;
            return;
        }
    }
    for (ulong i = 0u; i < wide * a.rank; i++) {
        if (!isfinite(up[i])) {
            status[token] = DS4_Q4E_METAL_NONFINITE;
            return;
        }
    }
    for (uint i = 0u; i < a.dim; i++) {
        float total = 0.0f;
        for (uint stream = 0u; stream < a.n_stream; stream++) {
            const ulong out_index = (ulong)stream * a.dim + i;
            float gate_logit = 0.0f;
            for (uint r = 0u; r < a.rank; r++) {
                float hidden_sum = 0.0f;
                for (ulong j = 0u; j < wide; j++) {
                    const uint source_stream = (uint)(j / a.dim);
                    const float norm = residual[base + j] * inverse[source_stream] *
                        (1.0f + norm_weight[j]);
                    hidden_sum += down[(ulong)r * wide + j] * norm;
                }
                gate_logit += up[out_index * a.rank + r] *
                    q4e_silu(hidden_sum / (float)a.n_stream);
            }
            const float norm = residual[base + out_index] * inverse[stream] *
                (1.0f + norm_weight[out_index]);
            total += q4e_sigmoid(gate_logit) * norm;
        }
        const float result = total / (float)a.n_stream;
        if (!isfinite(result)) {
            status[token] = DS4_Q4E_METAL_NONFINITE;
            return;
        }
    }
    for (uint i = 0u; i < a.dim; i++) {
        float total = 0.0f;
        for (uint stream = 0u; stream < a.n_stream; stream++) {
            const ulong out_index = (ulong)stream * a.dim + i;
            float gate_logit = 0.0f;
            for (uint r = 0u; r < a.rank; r++) {
                float hidden_sum = 0.0f;
                for (ulong j = 0u; j < wide; j++) {
                    const uint source_stream = (uint)(j / a.dim);
                    const float norm = residual[base + j] * inverse[source_stream] *
                        (1.0f + norm_weight[j]);
                    hidden_sum += down[(ulong)r * wide + j] * norm;
                }
                gate_logit += up[out_index * a.rank + r] *
                    q4e_silu(hidden_sum / (float)a.n_stream);
            }
            const float norm = residual[base + out_index] * inverse[stream] *
                (1.0f + norm_weight[out_index]);
            total += q4e_sigmoid(gate_logit) * norm;
        }
        output[(ulong)token * a.dim + i] = total / (float)a.n_stream;
    }
    status[token] = DS4_Q4E_METAL_OK;
}

// input/output [sequence][token][channel], weight [channel][kernel], state
// [sequence][channel][dilation*(kernel-1)] oldest-to-newest. One
// sequence/channel owner performs the exact contiguous scan and commits its
// final private history once; sequences never share convolution history.
kernel void kernel_qwen4exp_depthwise_conv_sequence_silu_f32(
        constant ds4_metal_args_qwen4exp_conv &a [[buffer(0)]],
        device const float *input  [[buffer(1)]],
        device const float *weight [[buffer(2)]],
        device       float *state  [[buffer(3)]],
        device       float *output [[buffer(4)]],
        device       uint  *status [[buffer(5)]],
        uint item [[thread_position_in_grid]]) {
    const uint owner_count = a.n_sequence * a.n_channel;
    if (item >= owner_count || a.n_channel == 0u) return;
    const uint sequence = item / a.n_channel;
    const uint channel = item % a.n_channel;
    const uint history_len = a.dilation * (a.kernel_size - 1u);
    if (a.n_sequence == 0u || a.n_token == 0u || a.kernel_size < 2u ||
        a.kernel_size > 8u ||
        a.dilation == 0u || history_len > 32u) {
        status[item] = DS4_Q4E_METAL_BAD_ARGUMENT;
        return;
    }
    float history[32];
    for (uint slot = 0u; slot < history_len; slot++) {
        history[slot] = state[
            ((ulong)sequence * a.n_channel + channel) * history_len + slot];
        if (!isfinite(history[slot])) {
            status[item] = DS4_Q4E_METAL_NONFINITE;
            return;
        }
    }
    for (uint tap = 0u; tap < a.kernel_size; tap++) {
        if (!isfinite(weight[(ulong)channel * a.kernel_size + tap])) {
            status[item] = DS4_Q4E_METAL_NONFINITE;
            return;
        }
    }
    for (uint token = 0u; token < a.n_token; token++) {
        if (!isfinite(input[
                ((ulong)sequence * a.n_token + token) * a.n_channel + channel])) {
            status[item] = DS4_Q4E_METAL_NONFINITE;
            return;
        }
    }
    // Preflight arithmetic against a local copy.
    float check_history[32];
    for (uint slot = 0u; slot < history_len; slot++)
        check_history[slot] = history[slot];
    for (uint token = 0u; token < a.n_token; token++) {
        const float current = input[
            ((ulong)sequence * a.n_token + token) * a.n_channel + channel];
        float total = current * weight[(ulong)channel * a.kernel_size +
                                       a.kernel_size - 1u];
        for (uint tap = 0u; tap + 1u < a.kernel_size; tap++)
            total += check_history[tap * a.dilation] *
                weight[(ulong)channel * a.kernel_size + tap];
        if (!isfinite(q4e_silu(total))) {
            status[item] = DS4_Q4E_METAL_NONFINITE;
            return;
        }
        for (uint slot = 0u; slot + 1u < history_len; slot++)
            check_history[slot] = check_history[slot + 1u];
        check_history[history_len - 1u] = current;
    }
    for (uint token = 0u; token < a.n_token; token++) {
        const float current = input[
            ((ulong)sequence * a.n_token + token) * a.n_channel + channel];
        float total = current * weight[(ulong)channel * a.kernel_size +
                                       a.kernel_size - 1u];
        for (uint tap = 0u; tap + 1u < a.kernel_size; tap++)
            total += history[tap * a.dilation] *
                weight[(ulong)channel * a.kernel_size + tap];
        output[((ulong)sequence * a.n_token + token) * a.n_channel + channel] =
            q4e_silu(total);
        for (uint slot = 0u; slot + 1u < history_len; slot++)
            history[slot] = history[slot + 1u];
        history[history_len - 1u] = current;
    }
    for (uint slot = 0u; slot < history_len; slot++)
        state[((ulong)sequence * a.n_channel + channel) * history_len + slot] =
            history[slot];
    status[item] = DS4_Q4E_METAL_OK;
}

// Controls [token][value_head].
kernel void kernel_qwen4exp_gdn_controls_f32(
        constant ds4_metal_args_qwen4exp_controls &a [[buffer(0)]],
        device const float *alpha_logit [[buffer(1)]],
        device const float *beta_logit  [[buffer(2)]],
        device const float *a_log       [[buffer(3)]],
        device const float *dt_bias     [[buffer(4)]],
        device       float *log_decay   [[buffer(5)]],
        device       float *beta        [[buffer(6)]],
        device       uint  *status      [[buffer(7)]],
        uint index [[thread_position_in_grid]]) {
    const uint count = a.n_token * a.n_value_head;
    if (index >= count) return;
    if (a.n_token == 0u || a.n_value_head == 0u) return;
    const uint head = index % a.n_value_head;
    const float alpha = alpha_logit[index];
    const float beta_input = beta_logit[index];
    if (!isfinite(alpha) || !isfinite(beta_input) || !isfinite(a_log[head]) ||
        !isfinite(dt_bias[head])) {
        status[index] = DS4_Q4E_METAL_NONFINITE;
        return;
    }
    const float decay = -exp(a_log[head]) * q4e_softplus(alpha + dt_bias[head]);
    const float step = q4e_sigmoid(beta_input);
    if (!isfinite(decay) || !isfinite(step)) {
        status[index] = DS4_Q4E_METAL_NONFINITE;
        return;
    }
    log_decay[index] = decay;
    beta[index] = step;
    status[index] = DS4_Q4E_METAL_OK;
}

// Q/K [token][key_head][key_dim], V/output
// [token][value_head][value_dim], state
// [value_head][key_dim][value_dim].  Value head h uses grouped repeat decode
// key_head = h/(value_head/key_head), giving exact 16->48 triplets.
kernel void kernel_qwen4exp_gdn_sequence_f32(
        constant ds4_metal_args_qwen4exp_gdn &a [[buffer(0)]],
        device const float *query     [[buffer(1)]],
        device const float *key       [[buffer(2)]],
        device const float *value     [[buffer(3)]],
        device const float *log_decay [[buffer(4)]],
        device const float *beta      [[buffer(5)]],
        device       float *state     [[buffer(6)]],
        device       float *output    [[buffer(7)]],
        device       uint  *status    [[buffer(8)]],
        uint column [[thread_position_in_grid]]) {
    const uint n_column = a.n_value_head * a.value_dim;
    if (column >= n_column) return;
    if (a.n_token == 0u || a.n_key_head == 0u || a.n_value_head == 0u ||
        a.n_value_head % a.n_key_head != 0u || a.key_dim == 0u ||
        a.key_dim > 128u || a.value_dim == 0u || a.value_dim > 128u) {
        status[column] = DS4_Q4E_METAL_BAD_ARGUMENT;
        return;
    }
    const uint value_head = column / a.value_dim;
    const uint value_index = column % a.value_dim;
    const uint repeat = a.n_value_head / a.n_key_head;
    const uint key_head = value_head / repeat;
    float local_state[128];
    for (uint k = 0u; k < a.key_dim; k++) {
        local_state[k] = state[((ulong)value_head * a.key_dim + k) *
                               a.value_dim + value_index];
        if (!isfinite(local_state[k])) {
            status[column] = DS4_Q4E_METAL_NONFINITE;
            return;
        }
    }
    const float query_scale = sqrt((float)a.key_dim);
    for (uint token = 0u; token < a.n_token; token++) {
        const ulong qbase = ((ulong)token * a.n_key_head + key_head) * a.key_dim;
        float qss = 0.0f;
        float kss = 0.0f;
        for (uint k = 0u; k < a.key_dim; k++) {
            const float q = query[qbase + k];
            const float kval = key[qbase + k];
            if (!isfinite(q) || !isfinite(kval)) {
                status[column] = DS4_Q4E_METAL_NONFINITE;
                return;
            }
            qss += q * q;
            kss += kval * kval;
        }
        const float qinv = 1.0f / sqrt(qss + 1.0e-6f);
        const float kinv = 1.0f / sqrt(kss + 1.0e-6f);
        const ulong control = (ulong)token * a.n_value_head + value_head;
        const float decay_log = log_decay[control];
        const float step = beta[control];
        const float target =
            value[(control * a.value_dim) + value_index];
        if (!isfinite(qinv) || !isfinite(kinv) || !isfinite(decay_log) ||
            decay_log > 0.0f || !isfinite(step) || step < 0.0f || step > 1.0f ||
            !isfinite(target)) {
            status[column] = DS4_Q4E_METAL_NONFINITE;
            return;
        }
        const float decay = exp(decay_log);
        float prediction = 0.0f;
        for (uint k = 0u; k < a.key_dim; k++) {
            local_state[k] *= decay;
            prediction += local_state[k] * (key[qbase + k] * kinv);
        }
        const float delta = (target - prediction) * step;
        float result = 0.0f;
        for (uint k = 0u; k < a.key_dim; k++) {
            const float khat = key[qbase + k] * kinv;
            local_state[k] += khat * delta;
            const float qhat = (query[qbase + k] * qinv) / query_scale;
            result += local_state[k] * qhat;
        }
        if (!isfinite(result)) {
            status[column] = DS4_Q4E_METAL_NONFINITE;
            return;
        }
    }
    // The whole column is known finite; replay to emit every token and commit.
    for (uint k = 0u; k < a.key_dim; k++)
        local_state[k] = state[((ulong)value_head * a.key_dim + k) *
                               a.value_dim + value_index];
    for (uint token = 0u; token < a.n_token; token++) {
        const ulong qbase = ((ulong)token * a.n_key_head + key_head) * a.key_dim;
        float qss = 0.0f;
        float kss = 0.0f;
        for (uint k = 0u; k < a.key_dim; k++) {
            qss += query[qbase + k] * query[qbase + k];
            kss += key[qbase + k] * key[qbase + k];
        }
        const float qinv = 1.0f / sqrt(qss + 1.0e-6f);
        const float kinv = 1.0f / sqrt(kss + 1.0e-6f);
        const ulong control = (ulong)token * a.n_value_head + value_head;
        const float decay = exp(log_decay[control]);
        for (uint k = 0u; k < a.key_dim; k++) local_state[k] *= decay;
        float prediction = 0.0f;
        for (uint k = 0u; k < a.key_dim; k++)
            prediction += local_state[k] * (key[qbase + k] * kinv);
        const float delta =
            (value[control * a.value_dim + value_index] - prediction) *
            beta[control];
        float result = 0.0f;
        for (uint k = 0u; k < a.key_dim; k++) {
            const float khat = key[qbase + k] * kinv;
            local_state[k] += khat * delta;
            const float qhat = (query[qbase + k] * qinv) / query_scale;
            result += local_state[k] * qhat;
        }
        output[(control * a.value_dim) + value_index] = result;
    }
    for (uint k = 0u; k < a.key_dim; k++)
        state[((ulong)value_head * a.key_dim + k) * a.value_dim + value_index] =
            local_state[k];
    status[column] = DS4_Q4E_METAL_OK;
}

// Full F32 softmax across exactly 512 experts in the profile, deterministic
// descending probability / ascending ID selection, then selected renormalize.
kernel void kernel_qwen4exp_router_softmax_top10_f32(
        constant ds4_metal_args_qwen4exp_router &a [[buffer(0)]],
        device const float *logits          [[buffer(1)]],
        device       uint  *selected        [[buffer(2)]],
        device       float *selected_weight [[buffer(3)]],
        device       uint  *status          [[buffer(4)]],
        uint token [[thread_position_in_grid]]) {
    if (token >= a.n_token) return;
    if (a.n_expert != 512u || a.n_selected != 10u) {
        status[token] = DS4_Q4E_METAL_BAD_ARGUMENT;
        return;
    }
    const ulong base = (ulong)token * a.n_expert;
    float maximum = logits[base];
    if (!isfinite(maximum)) {
        status[token] = DS4_Q4E_METAL_NONFINITE;
        return;
    }
    for (uint expert = 1u; expert < a.n_expert; expert++) {
        const float value = logits[base + expert];
        if (!isfinite(value)) {
            status[token] = DS4_Q4E_METAL_NONFINITE;
            return;
        }
        if (value > maximum) maximum = value;
    }
    float total = 0.0f;
    for (uint expert = 0u; expert < a.n_expert; expert++)
        total += exp(logits[base + expert] - maximum);
    if (!(total > 0.0f) || !isfinite(total)) {
        status[token] = DS4_Q4E_METAL_NONFINITE;
        return;
    }
    uint chosen[10];
    float chosen_weight[10];
    for (uint slot = 0u; slot < 10u; slot++) {
        uint best = a.n_expert;
        float best_probability = 0.0f;
        for (uint expert = 0u; expert < a.n_expert; expert++) {
            bool seen = false;
            for (uint prior = 0u; prior < slot; prior++)
                if (chosen[prior] == expert) seen = true;
            if (seen) continue;
            const float probability =
                exp(logits[base + expert] - maximum) / total;
            if (best == a.n_expert || probability > best_probability ||
                (probability == best_probability && expert < best)) {
                best = expert;
                best_probability = probability;
            }
        }
        chosen[slot] = best;
        chosen_weight[slot] = best_probability;
    }
    float selected_total = 0.0f;
    for (uint slot = 0u; slot < 10u; slot++)
        selected_total += chosen_weight[slot];
    if (!(selected_total > 0.0f) || !isfinite(selected_total)) {
        status[token] = DS4_Q4E_METAL_NONFINITE;
        return;
    }
    for (uint slot = 0u; slot < 10u; slot++) {
        selected[(ulong)token * 10u + slot] = chosen[slot];
        selected_weight[(ulong)token * 10u + slot] =
            chosen_weight[slot] / selected_total;
    }
    status[token] = DS4_Q4E_METAL_OK;
}

// Tiny resident scalar MoE.  expert gate/up [expert][inner][input], down
// [expert][output][inner]; shared tensors omit the expert dimension.  The
// shared expert is multiplied by sigmoid(dot(shared_router,input)).
kernel void kernel_qwen4exp_moe_resident_f32(
        constant ds4_metal_args_qwen4exp_moe &a [[buffer(0)]],
        device const float *input          [[buffer(1)]],
        device const uint  *selected       [[buffer(2)]],
        device const float *selected_weight[[buffer(3)]],
        device const float *expert_gate    [[buffer(4)]],
        device const float *expert_up      [[buffer(5)]],
        device const float *expert_down    [[buffer(6)]],
        device const float *shared_gate    [[buffer(7)]],
        device const float *shared_up      [[buffer(8)]],
        device const float *shared_down    [[buffer(9)]],
        device const float *shared_router  [[buffer(10)]],
        device       float *output         [[buffer(11)]],
        device       uint  *status         [[buffer(12)]],
        uint token [[thread_position_in_grid]]) {
    if (token >= a.n_token) return;
    if (a.n_expert == 0u || a.n_selected == 0u || a.n_selected > 10u ||
        a.input_dim == 0u || a.expert_dim == 0u || a.expert_dim > 256u ||
        a.output_dim == 0u) {
        status[token] = DS4_Q4E_METAL_BAD_ARGUMENT;
        return;
    }
    const ulong input_base = (ulong)token * a.input_dim;
    float route_total = 0.0f;
    for (uint route = 0u; route < a.n_selected; route++) {
        const uint expert = selected[(ulong)token * a.n_selected + route];
        const float weight =
            selected_weight[(ulong)token * a.n_selected + route];
        if (expert >= a.n_expert || !isfinite(weight) || weight < 0.0f) {
            status[token] = DS4_Q4E_METAL_BAD_INDEX;
            return;
        }
        route_total += weight;
    }
    if (!isfinite(route_total)) {
        status[token] = DS4_Q4E_METAL_NONFINITE;
        return;
    }
    float shared_router_logit = 0.0f;
    for (uint i = 0u; i < a.input_dim; i++) {
        const float x = input[input_base + i];
        if (!isfinite(x) || !isfinite(shared_router[i])) {
            status[token] = DS4_Q4E_METAL_NONFINITE;
            return;
        }
        shared_router_logit += x * shared_router[i];
    }
    const float shared_scale = q4e_sigmoid(shared_router_logit);
    for (uint out = 0u; out < a.output_dim; out++) {
        float routed_sum = 0.0f;
        for (uint route = 0u; route < a.n_selected; route++) {
            const uint expert = selected[(ulong)token * a.n_selected + route];
            const float route_weight =
                selected_weight[(ulong)token * a.n_selected + route];
            float expert_output = 0.0f;
            for (uint inner = 0u; inner < a.expert_dim; inner++) {
                float gate_sum = 0.0f;
                float up_sum = 0.0f;
                const ulong matrix =
                    ((ulong)expert * a.expert_dim + inner) * a.input_dim;
                for (uint i = 0u; i < a.input_dim; i++) {
                    gate_sum += expert_gate[matrix + i] * input[input_base + i];
                    up_sum += expert_up[matrix + i] * input[input_base + i];
                }
                expert_output += expert_down[
                    ((ulong)expert * a.output_dim + out) * a.expert_dim + inner]
                    * q4e_silu(gate_sum) * up_sum;
            }
            routed_sum += route_weight * expert_output;
        }
        float shared_sum = 0.0f;
        for (uint inner = 0u; inner < a.expert_dim; inner++) {
            float gate_sum = 0.0f;
            float up_sum = 0.0f;
            for (uint i = 0u; i < a.input_dim; i++) {
                gate_sum += shared_gate[(ulong)inner * a.input_dim + i] *
                    input[input_base + i];
                up_sum += shared_up[(ulong)inner * a.input_dim + i] *
                    input[input_base + i];
            }
            shared_sum += shared_down[(ulong)out * a.expert_dim + inner] *
                q4e_silu(gate_sum) * up_sum;
        }
        const float result = routed_sum + shared_scale * shared_sum;
        if (!isfinite(result)) {
            status[token] = DS4_Q4E_METAL_NONFINITE;
            return;
        }
    }
    // Recompute after preflight to keep the token output unchanged on failure.
    for (uint out = 0u; out < a.output_dim; out++) {
        float routed_sum = 0.0f;
        for (uint route = 0u; route < a.n_selected; route++) {
            const uint expert = selected[(ulong)token * a.n_selected + route];
            float expert_output = 0.0f;
            for (uint inner = 0u; inner < a.expert_dim; inner++) {
                float gate_sum = 0.0f;
                float up_sum = 0.0f;
                const ulong matrix =
                    ((ulong)expert * a.expert_dim + inner) * a.input_dim;
                for (uint i = 0u; i < a.input_dim; i++) {
                    gate_sum += expert_gate[matrix + i] * input[input_base + i];
                    up_sum += expert_up[matrix + i] * input[input_base + i];
                }
                expert_output += expert_down[
                    ((ulong)expert * a.output_dim + out) * a.expert_dim + inner]
                    * q4e_silu(gate_sum) * up_sum;
            }
            routed_sum +=
                selected_weight[(ulong)token * a.n_selected + route] *
                expert_output;
        }
        float shared_sum = 0.0f;
        for (uint inner = 0u; inner < a.expert_dim; inner++) {
            float gate_sum = 0.0f;
            float up_sum = 0.0f;
            for (uint i = 0u; i < a.input_dim; i++) {
                gate_sum += shared_gate[(ulong)inner * a.input_dim + i] *
                    input[input_base + i];
                up_sum += shared_up[(ulong)inner * a.input_dim + i] *
                    input[input_base + i];
            }
            shared_sum += shared_down[(ulong)out * a.expert_dim + inner] *
                q4e_silu(gate_sum) * up_sum;
        }
        output[(ulong)token * a.output_dim + out] =
            routed_sum + shared_scale * shared_sum;
    }
    status[token] = DS4_Q4E_METAL_OK;
}

// raw_key [physical_slot][head_dim]. logical_slot/logical_position describe the
// attention cache's position line; the index cache must consume those same
// slots rather than allocate independently. Complete groups are mean-pooled,
// zero-centered RMS-normalized, then partial-RoPE rotated at the group's first
// logical position.
kernel void kernel_qwen4exp_qsa_group_keys_f32(
        constant ds4_metal_args_qwen4exp_qsa_group &a [[buffer(0)]],
        device const float *raw_key     [[buffer(1)]],
        device const uint  *logical_slot[[buffer(2)]],
        device const uint  *logical_pos [[buffer(3)]],
        device const float *norm_weight [[buffer(4)]],
        device       float *group_key   [[buffer(5)]],
        device       uint  *status      [[buffer(6)]],
        uint group [[thread_position_in_grid]]) {
    if (group >= a.n_group) return;
    if (a.compression != 4u || a.head_dim == 0u ||
        a.head_dim > 256u || a.n_slot == 0u || a.n_rot == 0u ||
        a.n_rot > a.head_dim ||
        (a.n_rot & 1u) != 0u || !(a.theta > 0.0f) ||
        !isfinite(a.theta) || !(a.epsilon > 0.0f) ||
        !isfinite(a.epsilon)) {
        status[group] = DS4_Q4E_METAL_BAD_ARGUMENT;
        return;
    }
    float pooled[256];
    float ss = 0.0f;
    for (uint i = 0u; i < a.head_dim; i++) {
        float sum = 0.0f;
        for (uint token = 0u; token < a.compression; token++) {
            const uint logical = group * a.compression + token;
            const uint slot = logical_slot[logical];
            const uint expected_position = a.position0 + logical;
            if (slot >= a.n_slot || logical_pos[logical] != expected_position) {
                status[group] = DS4_Q4E_METAL_BAD_INDEX;
                return;
            }
            const float x = raw_key[(ulong)slot * a.head_dim + i];
            if (!isfinite(x)) {
                status[group] = DS4_Q4E_METAL_NONFINITE;
                return;
            }
            sum += x;
            if (!isfinite(sum)) {
                status[group] = DS4_Q4E_METAL_NONFINITE;
                return;
            }
        }
        if (!isfinite(norm_weight[i])) {
            status[group] = DS4_Q4E_METAL_NONFINITE;
            return;
        }
        pooled[i] = sum / (float)a.compression;
        const float square = pooled[i] * pooled[i];
        if (!isfinite(pooled[i]) || !isfinite(square)) {
            status[group] = DS4_Q4E_METAL_NONFINITE;
            return;
        }
        ss += square;
        if (!isfinite(ss)) {
            status[group] = DS4_Q4E_METAL_NONFINITE;
            return;
        }
    }
    const float inverse = 1.0f / sqrt(ss / (float)a.head_dim + a.epsilon);
    if (!isfinite(inverse)) {
        status[group] = DS4_Q4E_METAL_NONFINITE;
        return;
    }
    for (uint i = 0u; i < a.head_dim; i++) {
        pooled[i] *= inverse * (1.0f + norm_weight[i]);
        if (!isfinite(pooled[i])) {
            status[group] = DS4_Q4E_METAL_NONFINITE;
            return;
        }
    }
    const uint half_rot = a.n_rot / 2u;
    const float position = (float)(a.position0 + group * a.compression);
    for (uint pair = 0u; pair < half_rot; pair++) {
        const float exponent = (2.0f * (float)pair) / (float)a.n_rot;
        const float angle = position / pow(a.theta, exponent);
        const float c = cos(angle);
        const float s = sin(angle);
        const float first = pooled[pair];
        const float second = pooled[half_rot + pair];
        const float rotated_first = first * c - second * s;
        const float rotated_second = second * c + first * s;
        if (!isfinite(angle) || !isfinite(c) || !isfinite(s) ||
            !isfinite(rotated_first) || !isfinite(rotated_second)) {
            status[group] = DS4_Q4E_METAL_NONFINITE;
            return;
        }
        pooled[pair] = rotated_first;
        pooled[half_rot + pair] = rotated_second;
    }
    /* Commit the entire group only after every finite-input reduction and
     * rotation has been proven finite. */
    for (uint i = 0u; i < a.head_dim; i++)
        group_key[(ulong)group * a.head_dim + i] = pooled[i];
    status[group] = DS4_Q4E_METAL_OK;
}

// query [query][query_head][head_dim], group_key [group][head_dim].  For each
// query, score is sum_head(ReLU(dot_head))/sqrt(head_dim), not ReLU(sum).
// Selected complete groups expand in score/ID order, then the visible raw tail
// is appended.  n_visible <= 2051 is a hard Phase-5 bound.
kernel void kernel_qwen4exp_qsa_select_positions_f32(
        constant ds4_metal_args_qwen4exp_qsa_select &a [[buffer(0)]],
        device const float *query       [[buffer(1)]],
        device const float *group_key   [[buffer(2)]],
        device const uint  *n_visible   [[buffer(3)]],
        device       uint  *position    [[buffer(4)]],
        device       uint  *n_position  [[buffer(5)]],
        device       uint  *status      [[buffer(6)]],
        uint query_index [[thread_position_in_grid]]) {
    if (query_index >= a.n_query) return;
    const uint visible = n_visible[query_index];
    if (a.n_query_head == 0u || a.head_dim == 0u || a.head_dim > 256u ||
        a.compression != 4u || a.group_budget > 512u ||
        a.max_width == 0u || visible > a.n_visible_max || visible > 2051u) {
        status[query_index] = DS4_Q4E_METAL_BAD_ARGUMENT;
        return;
    }
    const uint n_group = visible / a.compression;
    const uint select_group = min(n_group, a.group_budget);
    const uint tail = visible - n_group * a.compression;
    const uint width = select_group * a.compression + tail;
    if (width > a.max_width) {
        status[query_index] = DS4_Q4E_METAL_BAD_ARGUMENT;
        return;
    }
    float score[512];
    const float scale = 1.0f / sqrt((float)a.head_dim);
    for (uint group = 0u; group < n_group; group++) {
        float sum = 0.0f;
        for (uint head = 0u; head < a.n_query_head; head++) {
            float dot = 0.0f;
            for (uint i = 0u; i < a.head_dim; i++) {
                const float q = query[
                    ((ulong)query_index * a.n_query_head + head) * a.head_dim + i];
                const float k = group_key[(ulong)group * a.head_dim + i];
                if (!isfinite(q) || !isfinite(k)) {
                    status[query_index] = DS4_Q4E_METAL_NONFINITE;
                    return;
                }
                dot += q * k;
            }
            if (dot > 0.0f) sum += dot;
        }
        score[group] = sum * scale;
        if (!isfinite(score[group])) {
            status[query_index] = DS4_Q4E_METAL_NONFINITE;
            return;
        }
    }
    uint chosen[512];
    for (uint slot = 0u; slot < select_group; slot++) {
        uint best = n_group;
        for (uint group = 0u; group < n_group; group++) {
            bool seen = false;
            for (uint prior = 0u; prior < slot; prior++)
                if (chosen[prior] == group) seen = true;
            if (!seen && (best == n_group || score[group] > score[best] ||
                (score[group] == score[best] && group < best))) best = group;
        }
        chosen[slot] = best;
    }
    ulong write = (ulong)query_index * a.max_width;
    for (uint slot = 0u; slot < select_group; slot++)
        for (uint offset = 0u; offset < a.compression; offset++)
            position[write++] = chosen[slot] * a.compression + offset;
    const uint tail_start = n_group * a.compression;
    for (uint offset = 0u; offset < tail; offset++)
        position[write++] = tail_start + offset;
    n_position[query_index] = width;
    status[query_index] = DS4_Q4E_METAL_OK;
}

// Bounded dense gather/attention control. query [query][query_head][head_dim],
// key/value [token][kv_head][head_dim], selected positions are produced above.
kernel void kernel_qwen4exp_qsa_attention_f32(
        constant ds4_metal_args_qwen4exp_qsa_attention &a [[buffer(0)]],
        device const float *query      [[buffer(1)]],
        device const float *key        [[buffer(2)]],
        device const float *value      [[buffer(3)]],
        device const uint  *position   [[buffer(4)]],
        device const uint  *n_position [[buffer(5)]],
        device       float *output     [[buffer(6)]],
        device       uint  *status     [[buffer(7)]],
        uint lane [[thread_position_in_grid]]) {
    const uint count = a.n_query * a.n_query_head;
    if (lane >= count) return;
    const uint query_index = lane / a.n_query_head;
    const uint query_head = lane % a.n_query_head;
    if (a.n_query_head == 0u || a.n_kv_head == 0u ||
        a.n_query_head % a.n_kv_head != 0u || a.head_dim == 0u ||
        a.head_dim > 256u || a.n_key > 2051u || a.max_selected > 2051u ||
        n_position[query_index] == 0u ||
        n_position[query_index] > a.max_selected) {
        status[lane] = DS4_Q4E_METAL_BAD_ARGUMENT;
        return;
    }
    const uint kv_head = query_head / (a.n_query_head / a.n_kv_head);
    const uint n = n_position[query_index];
    const ulong selected_base = (ulong)query_index * a.max_selected;
    const ulong query_base =
        ((ulong)query_index * a.n_query_head + query_head) * a.head_dim;
    float maximum = -INFINITY;
    for (uint slot = 0u; slot < n; slot++) {
        const uint token = position[selected_base + slot];
        if (token >= a.n_key) {
            status[lane] = DS4_Q4E_METAL_BAD_INDEX;
            return;
        }
        const ulong key_base = ((ulong)token * a.n_kv_head + kv_head) * a.head_dim;
        float dot = 0.0f;
        for (uint i = 0u; i < a.head_dim; i++) {
            const float q = query[query_base + i];
            const float k = key[key_base + i];
            const float v = value[key_base + i];
            if (!isfinite(q) || !isfinite(k) || !isfinite(v)) {
                status[lane] = DS4_Q4E_METAL_NONFINITE;
                return;
            }
            dot += q * k;
        }
        const float score = dot / sqrt((float)a.head_dim);
        if (score > maximum) maximum = score;
    }
    float denominator = 0.0f;
    for (uint slot = 0u; slot < n; slot++) {
        const uint token = position[selected_base + slot];
        const ulong key_base = ((ulong)token * a.n_kv_head + kv_head) * a.head_dim;
        float dot = 0.0f;
        for (uint i = 0u; i < a.head_dim; i++)
            dot += query[query_base + i] * key[key_base + i];
        denominator += exp(dot / sqrt((float)a.head_dim) - maximum);
    }
    if (!(denominator > 0.0f) || !isfinite(denominator)) {
        status[lane] = DS4_Q4E_METAL_NONFINITE;
        return;
    }
    for (uint i = 0u; i < a.head_dim; i++) {
        float sum = 0.0f;
        for (uint slot = 0u; slot < n; slot++) {
            const uint token = position[selected_base + slot];
            const ulong kv_base = ((ulong)token * a.n_kv_head + kv_head) * a.head_dim;
            float dot = 0.0f;
            for (uint j = 0u; j < a.head_dim; j++)
                dot += query[query_base + j] * key[kv_base + j];
            const float probability =
                exp(dot / sqrt((float)a.head_dim) - maximum) / denominator;
            const float v = value[kv_base + i];
            if (!isfinite(v)) {
                status[lane] = DS4_Q4E_METAL_NONFINITE;
                return;
            }
            sum += probability * v;
        }
        output[query_base + i] = sum;
    }
    status[lane] = DS4_Q4E_METAL_OK;
}

// resident rows [n_row][row_dim], row_id [token][head], gathered
// [token][head][row_dim].
kernel void kernel_qwen4exp_ple_gather_f32(
        constant ds4_metal_args_qwen4exp_ple_gather &a [[buffer(0)]],
        device const uint  *row_id   [[buffer(1)]],
        device const float *rows     [[buffer(2)]],
        device       float *gathered [[buffer(3)]],
        device       uint  *status   [[buffer(4)]],
        uint item [[thread_position_in_grid]]) {
    const uint count = a.n_token * a.n_head;
    if (item >= count) return;
    const uint row = row_id[item];
    if (a.row_dim == 0u || a.n_row == 0u || row >= a.n_row) {
        status[item] = DS4_Q4E_METAL_BAD_INDEX;
        return;
    }
    for (uint i = 0u; i < a.row_dim; i++) {
        const float value = rows[(ulong)row * a.row_dim + i];
        if (!isfinite(value)) {
            status[item] = DS4_Q4E_METAL_NONFINITE;
            return;
        }
    }
    for (uint i = 0u; i < a.row_dim; i++)
        gathered[(ulong)item * a.row_dim + i] =
            rows[(ulong)row * a.row_dim + i];
    status[item] = DS4_Q4E_METAL_OK;
}

// query/key [token][stream][dim], value [token][dim], output same as query.
kernel void kernel_qwen4exp_ple_gate_f32(
        constant ds4_metal_args_qwen4exp_ple_gate &a [[buffer(0)]],
        device const float *query  [[buffer(1)]],
        device const float *key    [[buffer(2)]],
        device const float *value  [[buffer(3)]],
        device       float *output [[buffer(4)]],
        device       uint  *status [[buffer(5)]],
        uint item [[thread_position_in_grid]]) {
    const uint count = a.n_token * a.n_stream;
    if (item >= count) return;
    if (a.dim == 0u || a.n_stream == 0u) {
        status[item] = DS4_Q4E_METAL_BAD_ARGUMENT;
        return;
    }
    float dot = 0.0f;
    const ulong base = (ulong)item * a.dim;
    for (uint i = 0u; i < a.dim; i++) {
        const float q = query[base + i];
        const float k = key[base + i];
        const float v = value[((ulong)item / a.n_stream) * a.dim + i];
        if (!isfinite(q) || !isfinite(k) || !isfinite(v)) {
            status[item] = DS4_Q4E_METAL_NONFINITE;
            return;
        }
        dot += q * k;
    }
    const float scaled = dot / sqrt((float)a.dim);
    const float signed_root = scaled > 0.0f
        ? sqrt(max(scaled, 1.0e-6f))
        : scaled < 0.0f ? -sqrt(max(-scaled, 1.0e-6f)) : 0.0f;
    const float gate = q4e_sigmoid(signed_root);
    for (uint i = 0u; i < a.dim; i++)
        output[base + i] = gate *
            value[((ulong)item / a.n_stream) * a.dim + i];
    status[item] = DS4_Q4E_METAL_OK;
}

// Final unnormalized dense head: input [token][input_dim], weight
// [output_dim][input_dim], output [token][output_dim].
kernel void kernel_qwen4exp_head_f32(
        constant ds4_metal_args_qwen4exp_head &a [[buffer(0)]],
        device const float *input  [[buffer(1)]],
        device const float *weight [[buffer(2)]],
        device       float *output [[buffer(3)]],
        device       uint  *status [[buffer(4)]],
        uint item [[thread_position_in_grid]]) {
    const uint count = a.n_token * a.output_dim;
    if (item >= count) return;
    if (a.input_dim == 0u || a.output_dim == 0u) return;
    const uint token = item / a.output_dim;
    const uint out = item % a.output_dim;
    float sum = 0.0f;
    for (uint i = 0u; i < a.input_dim; i++) {
        const float x = input[(ulong)token * a.input_dim + i];
        const float w = weight[(ulong)out * a.input_dim + i];
        if (!isfinite(x) || !isfinite(w)) {
            status[item] = DS4_Q4E_METAL_NONFINITE;
            return;
        }
        sum += x * w;
    }
    if (!isfinite(sum)) {
        status[item] = DS4_Q4E_METAL_NONFINITE;
        return;
    }
    output[item] = sum;
    status[item] = DS4_Q4E_METAL_OK;
}
