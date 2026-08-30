#include "ds4_qwen4exp_ref.h"

#include <limits.h>
#include <math.h>
#include <string.h>

static bool q4e_size_mul(size_t a, size_t b, size_t *result) {
    if (!result || (a != 0 && b > SIZE_MAX / a)) return false;
    *result = a * b;
    return true;
}

static bool q4e_size_add(size_t a, size_t b, size_t *result) {
    if (!result || a > SIZE_MAX - b) return false;
    *result = a + b;
    return true;
}

static bool q4e_size_mul3(size_t a, size_t b, size_t c, size_t *result) {
    size_t ab = 0;
    return q4e_size_mul(a, b, &ab) && q4e_size_mul(ab, c, result);
}

static bool q4e_finite_array(const float *values, size_t count) {
    if (!values) return false;
    for (size_t i = 0; i < count; i++) {
        if (!isfinite(values[i])) return false;
    }
    return true;
}

static float q4e_sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    const float e = expf(x);
    return e / (1.0f + e);
}

static float q4e_silu(float x) {
    return x * q4e_sigmoid(x);
}

static float q4e_softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

static bool q4e_rms_inverse(const float *input, size_t dim, float epsilon,
                            float *inverse) {
    float sum_square = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        sum_square += input[i] * input[i];
    }
    const float mean_square = sum_square / (float)dim;
    const float denominator = mean_square + epsilon;
    if (!isfinite(sum_square) || !isfinite(denominator) ||
        !(denominator > 0.0f)) {
        return false;
    }
    *inverse = 1.0f / sqrtf(denominator);
    return isfinite(*inverse);
}

static bool q4e_validate_norm_shape(size_t n_vector, size_t dim,
                                    size_t *element_count) {
    return n_vector > 0 && dim > 0 &&
           q4e_size_mul(n_vector, dim, element_count);
}

bool ds4_qwen4exp_ref_zero_centered_rmsnorm_f32(
        float       *output,
        const float *input,
        const float *weight,
        size_t       n_vector,
        size_t       dim,
        float        epsilon) {
    size_t count = 0;
    if (!output || !input || !weight || !(epsilon > 0.0f) ||
        !isfinite(epsilon) ||
        !q4e_validate_norm_shape(n_vector, dim, &count) ||
        !q4e_finite_array(input, count) ||
        !q4e_finite_array(weight, dim)) {
        return false;
    }

    for (size_t vector = 0; vector < n_vector; vector++) {
        float inverse = 0.0f;
        if (!q4e_rms_inverse(input + vector * dim, dim, epsilon, &inverse)) {
            return false;
        }
        for (size_t i = 0; i < dim; i++) {
            const float value = input[vector * dim + i] * inverse *
                                (1.0f + weight[i]);
            if (!isfinite(value)) return false;
        }
    }

    for (size_t vector = 0; vector < n_vector; vector++) {
        float inverse = 0.0f;
        (void)q4e_rms_inverse(input + vector * dim, dim, epsilon, &inverse);
        for (size_t i = 0; i < dim; i++) {
            output[vector * dim + i] = input[vector * dim + i] * inverse *
                                       (1.0f + weight[i]);
        }
    }
    return true;
}

bool ds4_qwen4exp_ref_sigmoid_gated_rmsnorm_f32(
        float       *output,
        const float *input,
        const float *gate_logit,
        const float *weight,
        size_t       n_vector,
        size_t       dim,
        float        epsilon) {
    size_t count = 0;
    if (!output || !input || !gate_logit || !weight || !(epsilon > 0.0f) ||
        !isfinite(epsilon) ||
        !q4e_validate_norm_shape(n_vector, dim, &count) ||
        !q4e_finite_array(input, count) ||
        !q4e_finite_array(gate_logit, count) ||
        !q4e_finite_array(weight, dim)) {
        return false;
    }

    for (size_t vector = 0; vector < n_vector; vector++) {
        float inverse = 0.0f;
        if (!q4e_rms_inverse(input + vector * dim, dim, epsilon, &inverse)) {
            return false;
        }
        for (size_t i = 0; i < dim; i++) {
            const size_t offset = vector * dim + i;
            const float value = input[offset] * inverse * weight[i] *
                                q4e_sigmoid(gate_logit[offset]);
            if (!isfinite(value)) return false;
        }
    }

    for (size_t vector = 0; vector < n_vector; vector++) {
        float inverse = 0.0f;
        (void)q4e_rms_inverse(input + vector * dim, dim, epsilon, &inverse);
        for (size_t i = 0; i < dim; i++) {
            const size_t offset = vector * dim + i;
            output[offset] = input[offset] * inverse * weight[i] *
                             q4e_sigmoid(gate_logit[offset]);
        }
    }
    return true;
}

static bool q4e_gr_inputs(
        const float *residual,
        const float *norm_weight,
        const float *down,
        const float *up,
        size_t       n_stream,
        size_t       dim,
        size_t       rank,
        float        epsilon,
        size_t      *wide,
        size_t      *down_count,
        size_t      *up_count) {
    if (!residual || !norm_weight || !down || !up || n_stream == 0 ||
        dim == 0 || rank == 0 || n_stream > DS4_QWEN4EXP_RESIDUAL_STREAMS ||
        dim > DS4_QWEN4EXP_HIDDEN_SIZE || rank > DS4_QWEN4EXP_GR_LOW_RANK ||
        !(epsilon > 0.0f) || !isfinite(epsilon) ||
        !q4e_size_mul(n_stream, dim, wide) ||
        !q4e_size_mul(rank, *wide, down_count) ||
        !q4e_size_mul(*wide, rank, up_count)) {
        return false;
    }
    return q4e_finite_array(residual, *wide) &&
           q4e_finite_array(norm_weight, *wide) &&
           q4e_finite_array(down, *down_count) &&
           q4e_finite_array(up, *up_count);
}

static bool q4e_gr_build(
        float       *mixed,
        float       *injection,
        const float *residual,
        const float *norm_weight,
        const float *down,
        const float *up,
        const float *inject,
        size_t       n_stream,
        size_t       dim,
        size_t       rank,
        float        epsilon,
        bool         write_output) {
    const size_t wide = n_stream * dim;
    float normalized[DS4_QWEN4EXP_WIDE_SIZE];
    float hidden[DS4_QWEN4EXP_GR_LOW_RANK];

    for (size_t stream = 0; stream < n_stream; stream++) {
        float inverse = 0.0f;
        if (!q4e_rms_inverse(residual + stream * dim, dim, epsilon, &inverse)) {
            return false;
        }
        for (size_t i = 0; i < dim; i++) {
            const size_t offset = stream * dim + i;
            normalized[offset] = residual[offset] * inverse *
                                 (1.0f + norm_weight[offset]);
            if (!isfinite(normalized[offset])) return false;
        }
    }

    for (size_t r = 0; r < rank; r++) {
        float sum = 0.0f;
        for (size_t i = 0; i < wide; i++) {
            sum += down[r * wide + i] * normalized[i];
        }
        hidden[r] = q4e_silu(sum / (float)n_stream);
        if (!isfinite(hidden[r])) return false;
    }

    for (size_t i = 0; i < dim; i++) {
        float sum = 0.0f;
        for (size_t stream = 0; stream < n_stream; stream++) {
            const size_t output_index = stream * dim + i;
            float gate_logit = 0.0f;
            for (size_t r = 0; r < rank; r++) {
                gate_logit += up[output_index * rank + r] * hidden[r];
            }
            sum += q4e_sigmoid(gate_logit) * normalized[output_index];
        }
        const float value = sum / (float)n_stream;
        if (!isfinite(value)) return false;
        if (write_output) mixed[i] = value;
    }

    if (injection) {
        for (size_t stream = 0; stream < n_stream; stream++) {
            float sum = 0.0f;
            for (size_t i = 0; i < wide; i++) {
                sum += inject[stream * wide + i] * normalized[i];
            }
            const float value = 2.0f * q4e_sigmoid(sum / (float)n_stream);
            if (!isfinite(value)) return false;
            if (write_output) injection[stream] = value;
        }
    }
    return true;
}

bool ds4_qwen4exp_ref_gr_prepare_f32(
        float       *mixed,
        float       *injection,
        const float *residual,
        const float *norm_weight,
        const float *down,
        const float *up,
        const float *inject,
        size_t       n_stream,
        size_t       dim,
        size_t       rank,
        float        epsilon) {
    size_t wide = 0;
    size_t down_count = 0;
    size_t up_count = 0;
    size_t inject_count = 0;
    if (!mixed || !injection || !inject ||
        !q4e_gr_inputs(residual, norm_weight, down, up, n_stream, dim, rank,
                       epsilon, &wide, &down_count, &up_count) ||
        !q4e_size_mul(n_stream, wide, &inject_count) ||
        !q4e_finite_array(inject, inject_count)) {
        return false;
    }
    if (!q4e_gr_build(mixed, injection, residual, norm_weight, down, up,
                      inject, n_stream, dim, rank, epsilon, false)) {
        return false;
    }
    return q4e_gr_build(mixed, injection, residual, norm_weight, down, up,
                        inject, n_stream, dim, rank, epsilon, true);
}

bool ds4_qwen4exp_ref_gr_apply_f32(
        float       *residual,
        const float *block_output,
        const float *injection,
        size_t       n_stream,
        size_t       dim) {
    size_t wide = 0;
    if (!residual || !block_output || !injection || n_stream == 0 || dim == 0 ||
        !q4e_size_mul(n_stream, dim, &wide) ||
        !q4e_finite_array(residual, wide) ||
        !q4e_finite_array(block_output, dim) ||
        !q4e_finite_array(injection, n_stream)) {
        return false;
    }
    for (size_t stream = 0; stream < n_stream; stream++) {
        for (size_t i = 0; i < dim; i++) {
            const float value = residual[stream * dim + i] +
                                injection[stream] * block_output[i];
            if (!isfinite(value)) return false;
        }
    }
    for (size_t stream = 0; stream < n_stream; stream++) {
        for (size_t i = 0; i < dim; i++) {
            residual[stream * dim + i] +=
                injection[stream] * block_output[i];
        }
    }
    return true;
}

bool ds4_qwen4exp_ref_gr_final_mix_f32(
        float       *output,
        const float *residual,
        const float *norm_weight,
        const float *down,
        const float *up,
        size_t       n_stream,
        size_t       dim,
        size_t       rank,
        float        epsilon) {
    size_t wide = 0;
    size_t down_count = 0;
    size_t up_count = 0;
    if (!output ||
        !q4e_gr_inputs(residual, norm_weight, down, up, n_stream, dim, rank,
                       epsilon, &wide, &down_count, &up_count)) {
        return false;
    }
    if (!q4e_gr_build(output, NULL, residual, norm_weight, down, up, NULL,
                      n_stream, dim, rank, epsilon, false)) {
        return false;
    }
    return q4e_gr_build(output, NULL, residual, norm_weight, down, up, NULL,
                        n_stream, dim, rank, epsilon, true);
}

static float q4e_conv_source(
        const float *state,
        const float *input,
        size_t       token,
        size_t       channel,
        size_t       tap,
        size_t       n_channel,
        size_t       history_len,
        size_t       spacing) {
    const size_t history_index = token + tap * spacing;
    if (history_index < history_len) {
        return state[channel * history_len + history_index];
    }
    return input[(history_index - history_len) * n_channel + channel];
}

static bool q4e_conv_validate_and_run(
        float       *output,
        float       *state,
        const float *input,
        const float *weight,
        size_t       n_token,
        size_t       n_channel,
        size_t       kernel,
        size_t       dilation,
        bool         write_output) {
    const size_t history_len = dilation * (kernel - 1u);
    for (size_t token = 0; token < n_token; token++) {
        for (size_t channel = 0; channel < n_channel; channel++) {
            float sum = input[token * n_channel + channel] *
                        weight[channel * kernel + kernel - 1u];
            for (size_t tap = 0; tap + 1u < kernel; tap++) {
                sum += q4e_conv_source(state, input, token, channel, tap,
                                       n_channel, history_len, dilation) *
                       weight[channel * kernel + tap];
            }
            const float value = q4e_silu(sum);
            if (!isfinite(value)) return false;
            if (write_output) output[token * n_channel + channel] = value;
        }
    }
    return true;
}

static void q4e_conv_commit_state(
        float       *state,
        const float *input,
        size_t       n_token,
        size_t       n_channel,
        size_t       history_len) {
    for (size_t channel = 0; channel < n_channel; channel++) {
        for (size_t slot = 0; slot < history_len; slot++) {
            const size_t combined_index = n_token + slot;
            if (combined_index < history_len) {
                state[channel * history_len + slot] =
                    state[channel * history_len + combined_index];
            } else {
                state[channel * history_len + slot] =
                    input[(combined_index - history_len) * n_channel + channel];
            }
        }
    }
}

static bool q4e_conv_f32(
        float       *output,
        float       *state,
        const float *input,
        const float *weight,
        size_t       n_token,
        size_t       n_channel,
        size_t       kernel,
        size_t       dilation) {
    size_t input_count = 0;
    size_t weight_count = 0;
    size_t history_len = 0;
    size_t state_count = 0;
    size_t combined_len = 0;
    if (!output || !state || !input || !weight || n_token == 0 ||
        n_channel == 0 || kernel < 2 || dilation == 0 ||
        !q4e_size_mul(n_token, n_channel, &input_count) ||
        !q4e_size_mul(n_channel, kernel, &weight_count) ||
        !q4e_size_mul(dilation, kernel - 1u, &history_len) ||
        !q4e_size_mul(n_channel, history_len, &state_count) ||
        !q4e_size_add(n_token, history_len, &combined_len) ||
        !q4e_finite_array(input, input_count) ||
        !q4e_finite_array(weight, weight_count) ||
        !q4e_finite_array(state, state_count)) {
        return false;
    }
    (void)combined_len;
    if (!q4e_conv_validate_and_run(output, state, input, weight, n_token,
                                   n_channel, kernel, dilation, false)) {
        return false;
    }
    (void)q4e_conv_validate_and_run(output, state, input, weight, n_token,
                                    n_channel, kernel, dilation, true);
    q4e_conv_commit_state(state, input, n_token, n_channel, history_len);
    return true;
}

bool ds4_qwen4exp_ref_causal_conv1d_silu_f32(
        float       *output,
        float       *state,
        const float *input,
        const float *weight,
        size_t       n_token,
        size_t       n_channel,
        size_t       kernel) {
    return q4e_conv_f32(output, state, input, weight, n_token, n_channel,
                        kernel, 1u);
}

bool ds4_qwen4exp_ref_gdn_controls_f32(
        float       *log_decay,
        float       *beta,
        const float *alpha_logit,
        const float *beta_logit,
        const float *a_log,
        const float *dt_bias,
        size_t       n_token,
        size_t       n_value_head) {
    size_t count = 0;
    if (!log_decay || !beta || !alpha_logit || !beta_logit || !a_log ||
        !dt_bias || n_token == 0 || n_value_head == 0 ||
        !q4e_size_mul(n_token, n_value_head, &count) ||
        !q4e_finite_array(alpha_logit, count) ||
        !q4e_finite_array(beta_logit, count) ||
        !q4e_finite_array(a_log, n_value_head) ||
        !q4e_finite_array(dt_bias, n_value_head)) {
        return false;
    }
    for (size_t token = 0; token < n_token; token++) {
        for (size_t head = 0; head < n_value_head; head++) {
            const size_t index = token * n_value_head + head;
            const float decay = -expf(a_log[head]) *
                                q4e_softplus(alpha_logit[index] +
                                             dt_bias[head]);
            const float step = q4e_sigmoid(beta_logit[index]);
            if (!isfinite(decay) || !isfinite(step)) return false;
        }
    }
    for (size_t token = 0; token < n_token; token++) {
        for (size_t head = 0; head < n_value_head; head++) {
            const size_t index = token * n_value_head + head;
            log_decay[index] = -expf(a_log[head]) *
                               q4e_softplus(alpha_logit[index] +
                                            dt_bias[head]);
            beta[index] = q4e_sigmoid(beta_logit[index]);
        }
    }
    return true;
}

static bool q4e_gdn_shapes(
        size_t n_token,
        size_t n_key_head,
        size_t n_value_head,
        size_t key_dim,
        size_t value_dim,
        size_t *qk_count,
        size_t *value_count,
        size_t *state_count,
        size_t *control_count) {
    return n_token > 0 && n_key_head > 0 && n_value_head > 0 && key_dim > 0 &&
           value_dim > 0 && n_value_head % n_key_head == 0 &&
           q4e_size_mul3(n_token, n_key_head, key_dim, qk_count) &&
           q4e_size_mul3(n_token, n_value_head, value_dim, value_count) &&
           q4e_size_mul3(n_value_head, key_dim, value_dim, state_count) &&
           q4e_size_mul(n_token, n_value_head, control_count);
}

static bool q4e_gdn_norms(
        const float *query,
        const float *key,
        size_t       token,
        size_t       key_head,
        size_t       n_key_head,
        size_t       key_dim,
        float       *query_inverse,
        float       *key_inverse) {
    const float *q = query + (token * n_key_head + key_head) * key_dim;
    const float *k = key + (token * n_key_head + key_head) * key_dim;
    float q_square = 0.0f;
    float k_square = 0.0f;
    for (size_t i = 0; i < key_dim; i++) {
        q_square += q[i] * q[i];
        k_square += k[i] * k[i];
    }
    if (!isfinite(q_square) || !isfinite(k_square)) return false;
    *query_inverse = 1.0f / sqrtf(q_square + 1.0e-6f);
    *key_inverse = 1.0f / sqrtf(k_square + 1.0e-6f);
    return isfinite(*query_inverse) && isfinite(*key_inverse);
}

/* Simulate one independent [key] column of a value head.  A stride of one is
 * used for preflight scratch; value_dim is used for the real [key][value]
 * state.  Keeping normalization and query scaling as separate F32 operations
 * matches the pinned fallback and deliberately avoids reassociation. */
static bool q4e_gdn_column_run(
        float       *column,
        size_t       column_stride,
        float       *output,
        const float *query,
        const float *key,
        const float *value,
        const float *log_decay,
        const float *beta,
        size_t       n_token,
        size_t       n_key_head,
        size_t       n_value_head,
        size_t       value_head,
        size_t       key_dim,
        size_t       value_dim,
        size_t       value_index) {
    const size_t repeat_ratio = n_value_head / n_key_head;
    const size_t key_head = value_head / repeat_ratio;
    const float query_scale = sqrtf((float)key_dim);
    float qhat[DS4_QWEN4EXP_GDN_HEAD_DIM];
    float khat[DS4_QWEN4EXP_GDN_HEAD_DIM];

    if (!isfinite(query_scale) || !(query_scale > 0.0f)) return false;
    for (size_t token = 0; token < n_token; token++) {
        const float *q = query +
            (token * n_key_head + key_head) * key_dim;
        const float *k = key +
            (token * n_key_head + key_head) * key_dim;
        const size_t control = token * n_value_head + value_head;
        const size_t value_offset =
            (token * n_value_head + value_head) * value_dim + value_index;
        float query_inverse = 0.0f;
        float key_inverse = 0.0f;
        if (!q4e_gdn_norms(query, key, token, key_head, n_key_head,
                           key_dim, &query_inverse, &key_inverse)) {
            return false;
        }
        for (size_t key_index = 0; key_index < key_dim; key_index++) {
            qhat[key_index] = q[key_index] * query_inverse;
            qhat[key_index] = qhat[key_index] / query_scale;
            khat[key_index] = k[key_index] * key_inverse;
            if (!isfinite(qhat[key_index]) || !isfinite(khat[key_index])) {
                return false;
            }
        }

        const float decay = expf(log_decay[control]);
        if (!isfinite(decay)) return false;
        for (size_t key_index = 0; key_index < key_dim; key_index++) {
            float *cell = column + key_index * column_stride;
            *cell = *cell * decay;
            if (!isfinite(*cell)) return false;
        }

        float prediction = 0.0f;
        for (size_t key_index = 0; key_index < key_dim; key_index++) {
            const float product =
                column[key_index * column_stride] * khat[key_index];
            if (!isfinite(product)) return false;
            prediction += product;
            if (!isfinite(prediction)) return false;
        }
        const float difference = value[value_offset] - prediction;
        const float delta = difference * beta[control];
        if (!isfinite(difference) || !isfinite(delta)) return false;

        for (size_t key_index = 0; key_index < key_dim; key_index++) {
            float *cell = column + key_index * column_stride;
            const float correction = khat[key_index] * delta;
            if (!isfinite(correction)) return false;
            *cell = *cell + correction;
            if (!isfinite(*cell)) return false;
        }

        float result = 0.0f;
        for (size_t key_index = 0; key_index < key_dim; key_index++) {
            const float product =
                column[key_index * column_stride] * qhat[key_index];
            if (!isfinite(product)) return false;
            result += product;
            if (!isfinite(result)) return false;
        }
        if (output) output[value_offset] = result;
    }
    return true;
}

bool ds4_qwen4exp_ref_gdn_f32(
        float       *output,
        float       *state,
        const float *query,
        const float *key,
        const float *value,
        const float *log_decay,
        const float *beta,
        size_t       n_token,
        size_t       n_key_head,
        size_t       n_value_head,
        size_t       key_dim,
        size_t       value_dim) {
    size_t qk_count = 0;
    size_t value_count = 0;
    size_t state_count = 0;
    size_t control_count = 0;
    if (!output || !state || !query || !key || !value || !log_decay || !beta ||
        n_key_head > DS4_QWEN4EXP_GDN_KEY_HEADS ||
        n_value_head > DS4_QWEN4EXP_GDN_VALUE_HEADS ||
        key_dim > DS4_QWEN4EXP_GDN_HEAD_DIM ||
        value_dim > DS4_QWEN4EXP_GDN_HEAD_DIM ||
        !q4e_gdn_shapes(n_token, n_key_head, n_value_head, key_dim, value_dim,
                        &qk_count, &value_count, &state_count, &control_count) ||
        !q4e_finite_array(query, qk_count) ||
        !q4e_finite_array(key, qk_count) ||
        !q4e_finite_array(value, value_count) ||
        !q4e_finite_array(state, state_count) ||
        !q4e_finite_array(log_decay, control_count) ||
        !q4e_finite_array(beta, control_count)) {
        return false;
    }
    for (size_t i = 0; i < control_count; i++) {
        if (log_decay[i] > 0.0f || beta[i] < 0.0f || beta[i] > 1.0f) {
            return false;
        }
    }

    const size_t head_state_count = key_dim * value_dim;
    float scratch[DS4_QWEN4EXP_GDN_HEAD_DIM];

    /* Preflight every independent state column.  No public state or output is
     * touched until every arithmetic step for the complete call is finite. */
    for (size_t value_head = 0; value_head < n_value_head; value_head++) {
        const float *head_state = state + value_head * head_state_count;
        for (size_t value_index = 0; value_index < value_dim; value_index++) {
            for (size_t key_index = 0; key_index < key_dim; key_index++) {
                scratch[key_index] =
                    head_state[key_index * value_dim + value_index];
            }
            if (!q4e_gdn_column_run(
                    scratch, 1u, NULL, query, key, value, log_decay, beta,
                    n_token, n_key_head, n_value_head, value_head, key_dim,
                    value_dim, value_index)) {
                return false;
            }
        }
    }

    for (size_t value_head = 0; value_head < n_value_head; value_head++) {
        float *head_state = state + value_head * head_state_count;
        for (size_t value_index = 0; value_index < value_dim; value_index++) {
            (void)q4e_gdn_column_run(
                head_state + value_index, value_dim, output, query, key, value,
                log_decay, beta, n_token, n_key_head, n_value_head, value_head,
                key_dim, value_dim, value_index);
        }
    }
    return true;
}

bool ds4_qwen4exp_ref_softmax_topk_f32(
        uint32_t    *selected,
        float       *selected_weight,
        const float *logits,
        size_t       n_expert,
        size_t       n_selected) {
    size_t bytes = 0;
    if (!selected || !selected_weight || !logits || n_expert == 0 ||
        n_selected == 0 || n_selected > n_expert || n_expert > UINT32_MAX ||
        !q4e_size_mul(n_expert, sizeof(logits[0]), &bytes) ||
        !q4e_finite_array(logits, n_expert)) {
        return false;
    }
    (void)bytes;

    float max_logit = logits[0];
    for (size_t i = 1; i < n_expert; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    float total = 0.0f;
    for (size_t i = 0; i < n_expert; i++) {
        total += expf(logits[i] - max_logit);
    }
    if (!(total > 0.0f) || !isfinite(total)) return false;

    for (size_t slot = 0; slot < n_selected; slot++) {
        size_t best = n_expert;
        float best_probability = 0.0f;
        for (size_t expert = 0; expert < n_expert; expert++) {
            bool seen = false;
            for (size_t prior = 0; prior < slot; prior++) {
                if (selected[prior] == (uint32_t)expert) {
                    seen = true;
                    break;
                }
            }
            if (seen) continue;
            const float probability = expf(logits[expert] - max_logit) / total;
            if (best == n_expert || probability > best_probability ||
                (probability == best_probability && expert < best)) {
                best = expert;
                best_probability = probability;
            }
        }
        selected[slot] = (uint32_t)best;
        selected_weight[slot] = best_probability;
    }
    float selected_total = 0.0f;
    for (size_t i = 0; i < n_selected; i++) {
        selected_total += selected_weight[i];
    }
    if (!(selected_total > 0.0f) || !isfinite(selected_total)) return false;
    for (size_t i = 0; i < n_selected; i++) {
        selected_weight[i] /= selected_total;
    }
    return true;
}

static bool q4e_rope_value(
        const float *values,
        size_t       base,
        size_t       pair,
        size_t       n_rot,
        uint32_t     position,
        float        theta,
        float       *first,
        float       *second) {
    const size_t half = n_rot / 2u;
    const float exponent = (2.0f * (float)pair) / (float)n_rot;
    const float frequency = 1.0f / powf(theta, exponent);
    const float angle = (float)position * frequency;
    const float cosine = cosf(angle);
    const float sine = sinf(angle);
    const float x_first = values[base + pair];
    const float x_second = values[base + half + pair];
    *first = x_first * cosine - x_second * sine;
    *second = x_second * cosine + x_first * sine;
    return isfinite(*first) && isfinite(*second);
}

bool ds4_qwen4exp_ref_partial_rope_f32(
        float          *values,
        const uint32_t *position,
        size_t          n_token,
        size_t          n_head,
        size_t          head_dim,
        size_t          n_rot,
        float           theta) {
    size_t count = 0;
    if (!values || !position || n_token == 0 || n_head == 0 || head_dim == 0 ||
        n_rot == 0 || n_rot > head_dim || (n_rot & 1u) != 0 ||
        !(theta > 0.0f) || !isfinite(theta) ||
        !q4e_size_mul3(n_token, n_head, head_dim, &count) ||
        !q4e_finite_array(values, count)) {
        return false;
    }
    for (size_t token = 0; token < n_token; token++) {
        for (size_t head = 0; head < n_head; head++) {
            const size_t base = (token * n_head + head) * head_dim;
            for (size_t pair = 0; pair < n_rot / 2u; pair++) {
                float first = 0.0f;
                float second = 0.0f;
                if (!q4e_rope_value(values, base, pair, n_rot,
                                    position[token], theta, &first, &second)) {
                    return false;
                }
            }
        }
    }
    for (size_t token = 0; token < n_token; token++) {
        for (size_t head = 0; head < n_head; head++) {
            const size_t base = (token * n_head + head) * head_dim;
            for (size_t pair = 0; pair < n_rot / 2u; pair++) {
                float first = 0.0f;
                float second = 0.0f;
                (void)q4e_rope_value(values, base, pair, n_rot,
                                     position[token], theta, &first, &second);
                values[base + pair] = first;
                values[base + n_rot / 2u + pair] = second;
            }
        }
    }
    return true;
}

static bool q4e_group_key_one(
        float       *group_key,
        const float *raw_key,
        const float *norm_weight,
        size_t       group,
        size_t       compression,
        size_t       head_dim,
        size_t       n_rot,
        float        theta,
        float        epsilon,
        bool         write_output) {
    float pooled[DS4_QWEN4EXP_QSA_HEAD_DIM];
    for (size_t i = 0; i < head_dim; i++) {
        float sum = 0.0f;
        for (size_t token = 0; token < compression; token++) {
            sum += raw_key[(group * compression + token) * head_dim + i];
        }
        pooled[i] = sum / (float)compression;
        if (!isfinite(pooled[i])) return false;
    }
    float inverse = 0.0f;
    if (!q4e_rms_inverse(pooled, head_dim, epsilon, &inverse)) return false;
    for (size_t i = 0; i < head_dim; i++) {
        pooled[i] *= inverse * (1.0f + norm_weight[i]);
        if (!isfinite(pooled[i])) return false;
    }

    const uint32_t position = (uint32_t)(group * compression);
    for (size_t pair = 0; pair < n_rot / 2u; pair++) {
        float first = 0.0f;
        float second = 0.0f;
        if (!q4e_rope_value(pooled, 0, pair, n_rot, position, theta,
                            &first, &second)) {
            return false;
        }
        if (write_output) {
            group_key[group * head_dim + pair] = first;
            group_key[group * head_dim + n_rot / 2u + pair] = second;
        }
    }
    if (write_output) {
        for (size_t i = n_rot; i < head_dim; i++) {
            group_key[group * head_dim + i] = pooled[i];
        }
    }
    return true;
}

bool ds4_qwen4exp_ref_qsa_group_keys_f32(
        float       *group_key,
        const float *raw_key,
        const float *norm_weight,
        size_t       n_group,
        size_t       compression,
        size_t       head_dim,
        size_t       n_rot,
        float        theta,
        float        epsilon) {
    size_t raw_count = 0;
    size_t group_count = 0;
    size_t position_end = 0;
    if (!group_key || !raw_key || !norm_weight || n_group == 0 ||
        compression == 0 || head_dim == 0 ||
        head_dim > DS4_QWEN4EXP_QSA_HEAD_DIM || n_rot == 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 || !(theta > 0.0f) ||
        !isfinite(theta) || !(epsilon > 0.0f) || !isfinite(epsilon) ||
        !q4e_size_mul3(n_group, compression, head_dim, &raw_count) ||
        !q4e_size_mul(n_group, head_dim, &group_count) ||
        !q4e_size_mul(n_group - 1u, compression, &position_end) ||
        position_end > UINT32_MAX ||
        !q4e_finite_array(raw_key, raw_count) ||
        !q4e_finite_array(norm_weight, head_dim)) {
        return false;
    }
    (void)group_count;
    for (size_t group = 0; group < n_group; group++) {
        if (!q4e_group_key_one(group_key, raw_key, norm_weight, group,
                               compression, head_dim, n_rot, theta, epsilon,
                               false)) {
            return false;
        }
    }
    for (size_t group = 0; group < n_group; group++) {
        (void)q4e_group_key_one(group_key, raw_key, norm_weight, group,
                                compression, head_dim, n_rot, theta, epsilon,
                                true);
    }
    return true;
}

bool ds4_qwen4exp_ref_qsa_scores_f32(
        float       *score,
        const float *query,
        const float *group_key,
        size_t       n_group,
        size_t       n_query_head,
        size_t       head_dim) {
    size_t query_count = 0;
    size_t group_count = 0;
    if (!score || !query || !group_key || n_group == 0 || n_query_head == 0 ||
        head_dim == 0 ||
        !q4e_size_mul(n_query_head, head_dim, &query_count) ||
        !q4e_size_mul(n_group, head_dim, &group_count) ||
        !q4e_finite_array(query, query_count) ||
        !q4e_finite_array(group_key, group_count)) {
        return false;
    }
    const float scale = 1.0f / sqrtf((float)head_dim);
    for (size_t group = 0; group < n_group; group++) {
        float sum = 0.0f;
        for (size_t head = 0; head < n_query_head; head++) {
            float dot = 0.0f;
            for (size_t i = 0; i < head_dim; i++) {
                dot += query[head * head_dim + i] *
                       group_key[group * head_dim + i];
            }
            if (dot > 0.0f) sum += dot;
        }
        const float value = sum * scale;
        if (!isfinite(value)) return false;
    }
    for (size_t group = 0; group < n_group; group++) {
        float sum = 0.0f;
        for (size_t head = 0; head < n_query_head; head++) {
            float dot = 0.0f;
            for (size_t i = 0; i < head_dim; i++) {
                dot += query[head * head_dim + i] *
                       group_key[group * head_dim + i];
            }
            if (dot > 0.0f) sum += dot;
        }
        score[group] = sum * scale;
    }
    return true;
}

bool ds4_qwen4exp_ref_qsa_select_positions(
        uint32_t    *position,
        size_t       position_capacity,
        size_t      *n_position,
        const float *score,
        size_t       visible_tokens,
        size_t       compression,
        size_t       group_budget) {
    if (!position || !n_position || !score || compression == 0 ||
        compression > DS4_QWEN4EXP_QSA_COMPRESSION ||
        visible_tokens > UINT32_MAX) {
        return false;
    }
    const size_t n_complete = visible_tokens / compression;
    const size_t selected_groups =
        n_complete < group_budget ? n_complete : group_budget;
    const size_t tail = visible_tokens - n_complete * compression;
    size_t selected_width = 0;
    if (!q4e_size_mul(selected_groups, compression, &selected_width) ||
        selected_width > SIZE_MAX - tail) {
        return false;
    }
    selected_width += tail;
    if (position_capacity < selected_width ||
        !q4e_finite_array(score, n_complete)) {
        return false;
    }

    size_t write = 0;
    for (size_t slot = 0; slot < selected_groups; slot++) {
        size_t best = n_complete;
        for (size_t group = 0; group < n_complete; group++) {
            bool seen = false;
            for (size_t prior = 0; prior < slot; prior++) {
                if (position[prior * compression] /
                        (uint32_t)compression == group) {
                    seen = true;
                    break;
                }
            }
            if (seen) continue;
            if (best == n_complete || score[group] > score[best] ||
                (score[group] == score[best] && group < best)) {
                best = group;
            }
        }
        for (size_t offset = 0; offset < compression; offset++) {
            position[write++] = (uint32_t)(best * compression + offset);
        }
    }
    const size_t tail_start = n_complete * compression;
    for (size_t offset = 0; offset < tail; offset++) {
        position[write++] = (uint32_t)(tail_start + offset);
    }
    *n_position = write;
    return true;
}

void ds4_qwen4exp_ref_ple_history_reset(ds4_qwen4exp_ple_history *history) {
    if (!history) return;
    history->token[0] = DS4_QWEN4EXP_PLE_PAD_TOKEN;
    history->token[1] = DS4_QWEN4EXP_PLE_PAD_TOKEN;
    history->count = 0;
}

bool ds4_qwen4exp_ref_ple_history_advance(
        ds4_qwen4exp_ple_history       *after,
        const ds4_qwen4exp_ple_history *before,
        uint32_t                         token) {
    if (!after || !before || before->count > 2u) return false;
    ds4_qwen4exp_ple_history next = {
        { DS4_QWEN4EXP_PLE_PAD_TOKEN, DS4_QWEN4EXP_PLE_PAD_TOKEN }, 0
    };
    if (token != DS4_QWEN4EXP_PLE_PAD_TOKEN) {
        next.token[0] = token;
        next.token[1] = before->count > 0u ? before->token[0]
                                           : DS4_QWEN4EXP_PLE_PAD_TOKEN;
        next.count = before->count < 2u ? before->count + 1u : 2u;
    }
    *after = next;
    return true;
}

bool ds4_qwen4exp_ref_ple_rows(
        uint32_t                         row[DS4_QWEN4EXP_PLE_HEADS],
        uint32_t                         current_token,
        const ds4_qwen4exp_ple_history *history,
        const ds4_qwen4exp_profile     *profile) {
    if (!row || !history || !profile || history->count > 2u ||
        !ds4_qwen4exp_profile_validate(profile)) {
        return false;
    }
    const uint64_t previous1 = history->count >= 1u
        ? history->token[0] : DS4_QWEN4EXP_PLE_PAD_TOKEN;
    const uint64_t previous2 = history->count >= 2u
        ? history->token[1] : DS4_QWEN4EXP_PLE_PAD_TOKEN;
    const uint64_t bigram =
        (uint64_t)current_token * profile->ple_multiplier[0] ^
        previous1 * profile->ple_multiplier[1];
    const uint64_t trigram =
        bigram ^ previous2 * profile->ple_multiplier[2];
    uint32_t planned[DS4_QWEN4EXP_PLE_HEADS];
    for (size_t head = 0; head < DS4_QWEN4EXP_PLE_HEADS; head++) {
        const uint32_t prime = profile->ple_head_prime[head];
        if (prime == 0u) return false;
        const uint64_t hash = head < DS4_QWEN4EXP_PLE_HEADS_PER_NGRAM
            ? bigram : trigram;
        const uint64_t result = profile->ple_head_offset[head] + hash % prime;
        if (result > UINT32_MAX || result >= profile->ple_rows) return false;
        planned[head] = (uint32_t)result;
    }
    memcpy(row, planned, sizeof(planned));
    return true;
}

bool ds4_qwen4exp_ref_ple_gate_f32(
        float       *output,
        const float *query,
        const float *key,
        const float *value,
        size_t       n_stream,
        size_t       dim) {
    size_t wide = 0;
    if (!output || !query || !key || !value || n_stream == 0 || dim == 0 ||
        !q4e_size_mul(n_stream, dim, &wide) ||
        !q4e_finite_array(query, wide) || !q4e_finite_array(key, wide) ||
        !q4e_finite_array(value, dim)) {
        return false;
    }
    const float scale = 1.0f / sqrtf((float)dim);
    for (size_t stream = 0; stream < n_stream; stream++) {
        float dot = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            dot += query[stream * dim + i] * key[stream * dim + i];
        }
        const float scaled = dot * scale;
        const float signed_root = scaled > 0.0f
            ? sqrtf(fmaxf(scaled, 1.0e-6f))
            : scaled < 0.0f ? -sqrtf(fmaxf(-scaled, 1.0e-6f)) : 0.0f;
        const float gate = q4e_sigmoid(signed_root);
        for (size_t i = 0; i < dim; i++) {
            if (!isfinite(gate * value[i])) return false;
        }
    }
    for (size_t stream = 0; stream < n_stream; stream++) {
        float dot = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            dot += query[stream * dim + i] * key[stream * dim + i];
        }
        const float scaled = dot * scale;
        const float signed_root = scaled > 0.0f
            ? sqrtf(fmaxf(scaled, 1.0e-6f))
            : scaled < 0.0f ? -sqrtf(fmaxf(-scaled, 1.0e-6f)) : 0.0f;
        const float gate = q4e_sigmoid(signed_root);
        for (size_t i = 0; i < dim; i++) {
            output[stream * dim + i] = gate * value[i];
        }
    }
    return true;
}

bool ds4_qwen4exp_ref_state_reset_f32(float *state, size_t n_value) {
    size_t bytes = 0;
    if (!state || n_value == 0 ||
        !q4e_size_mul(n_value, sizeof(state[0]), &bytes)) {
        return false;
    }
    memset(state, 0, bytes);
    return true;
}

static bool q4e_state_copy_f32(
        float       *destination,
        const float *source,
        size_t       n_value) {
    size_t bytes = 0;
    if (!destination || !source || n_value == 0 ||
        !q4e_size_mul(n_value, sizeof(source[0]), &bytes) ||
        !q4e_finite_array(source, n_value)) {
        return false;
    }
    if (destination == source) return true;

    const uintptr_t destination_begin = (uintptr_t)destination;
    const uintptr_t source_begin = (uintptr_t)source;
    if (destination_begin > UINTPTR_MAX - bytes ||
        source_begin > UINTPTR_MAX - bytes) {
        return false;
    }
    const uintptr_t destination_end = destination_begin + bytes;
    const uintptr_t source_end = source_begin + bytes;
    if (destination_begin < source_end && source_begin < destination_end) {
        return false;
    }
    memcpy(destination, source, bytes);
    return true;
}

bool ds4_qwen4exp_ref_state_copy_f32(
        float       *destination,
        const float *source,
        size_t       n_value) {
    return q4e_state_copy_f32(destination, source, n_value);
}

bool ds4_qwen4exp_ref_state_rewind_f32(
        float       *state,
        const float *checkpoint,
        size_t       n_value) {
    return q4e_state_copy_f32(state, checkpoint, n_value);
}

bool ds4_qwen4exp_ref_dilated_conv1d_silu_f32(
        float       *output,
        float       *state,
        const float *input,
        const float *weight,
        size_t       n_token,
        size_t       n_channel,
        size_t       kernel,
        size_t       dilation) {
    return q4e_conv_f32(output, state, input, weight, n_token, n_channel,
                        kernel, dilation);
}
