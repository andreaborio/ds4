#include "ds4_qwen_ref.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static float qwen_ref_sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    const float e = expf(x);
    return e / (1.0f + e);
}

static float qwen_ref_silu(float x) {
    return x * qwen_ref_sigmoid(x);
}

static float qwen_ref_softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

void ds4_qwen_ref_causal_conv1d_silu_f32(
        float       *output,
        float       *state,
        const float *input,
        const float *weight,
        size_t       n_token,
        size_t       n_channel,
        size_t       kernel) {
    if (!output || !state || !input || !weight || kernel < 2) return;

    for (size_t t = 0; t < n_token; t++) {
        for (size_t c = 0; c < n_channel; c++) {
            const size_t history_len = kernel - 1;
            float *history = state + c * history_len;
            const float *filter = weight + c * kernel;
            const float current = input[t * n_channel + c];
            float sum = current * filter[kernel - 1];
            for (size_t k = 0; k < history_len; k++) {
                sum += history[k] * filter[k];
            }
            output[t * n_channel + c] = qwen_ref_silu(sum);

            if (history_len > 1) {
                memmove(history, history + 1,
                        (history_len - 1) * sizeof(history[0]));
            }
            history[history_len - 1] = current;
        }
    }
}

bool ds4_qwen_ref_gated_delta_rule_f32(
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
    if (!output || !state || !query || !key || !value || !log_decay || !beta ||
        n_key_head == 0 || n_value_head == 0 || key_dim == 0 || value_dim == 0 ||
        n_value_head % n_key_head != 0) {
        return false;
    }

    const float query_scale = 1.0f / sqrtf((float)key_dim);

    for (size_t t = 0; t < n_token; t++) {
        for (size_t vh = 0; vh < n_value_head; vh++) {
            /* llama.cpp's Qwen GGUF conversion tiles the V-major quantities;
             * this is the runtime order, not HF's contiguous repeat order. */
            const size_t kh = vh % n_key_head;
            const float *q = query + (t * n_key_head + kh) * key_dim;
            const float *k = key + (t * n_key_head + kh) * key_dim;
            const float *v = value + (t * n_value_head + vh) * value_dim;
            float *head_state = state + vh * key_dim * value_dim;
            float *head_output = output + (t * n_value_head + vh) * value_dim;

            float q_sq = 0.0f;
            float k_sq = 0.0f;
            for (size_t i = 0; i < key_dim; i++) {
                q_sq += q[i] * q[i];
                k_sq += k[i] * k[i];
            }
            const float q_inv = query_scale / sqrtf(q_sq + 1.0e-6f);
            const float k_inv = 1.0f / sqrtf(k_sq + 1.0e-6f);
            const float decay = expf(log_decay[t * n_value_head + vh]);
            const float step = beta[t * n_value_head + vh];

            for (size_t i = 0; i < key_dim * value_dim; i++) {
                head_state[i] *= decay;
            }

            for (size_t j = 0; j < value_dim; j++) {
                float memory = 0.0f;
                for (size_t i = 0; i < key_dim; i++) {
                    memory += head_state[j * key_dim + i] * (k[i] * k_inv);
                }
                const float delta = (v[j] - memory) * step;
                for (size_t i = 0; i < key_dim; i++) {
                    head_state[j * key_dim + i] += (k[i] * k_inv) * delta;
                }
            }

            for (size_t j = 0; j < value_dim; j++) {
                float sum = 0.0f;
                for (size_t i = 0; i < key_dim; i++) {
                    sum += head_state[j * key_dim + i] * (q[i] * q_inv);
                }
                head_output[j] = sum;
            }
        }
    }
    return true;
}

void ds4_qwen_ref_gated_delta_controls_f32(
        float       *log_decay,
        float       *beta,
        const float *alpha_logit,
        const float *beta_logit,
        const float *ssm_a,
        const float *dt_bias,
        size_t       n_token,
        size_t       n_value_head) {
    if (!log_decay || !beta || !alpha_logit || !beta_logit || !ssm_a ||
        !dt_bias) {
        return;
    }
    for (size_t t = 0; t < n_token; t++) {
        for (size_t h = 0; h < n_value_head; h++) {
            const size_t i = t * n_value_head + h;
            beta[i] = qwen_ref_sigmoid(beta_logit[i]);
            log_decay[i] = ssm_a[h] * qwen_ref_softplus(
                alpha_logit[i] + dt_bias[h]);
        }
    }
}

void ds4_qwen_ref_rmsnorm_gated_f32(
        float       *output,
        const float *input,
        const float *gate,
        const float *weight,
        size_t       n_vector,
        size_t       dim,
        float        epsilon) {
    if (!output || !input || !gate || !weight || dim == 0) return;

    for (size_t v = 0; v < n_vector; v++) {
        const float *x = input + v * dim;
        const float *z = gate + v * dim;
        float *y = output + v * dim;
        float variance = 0.0f;
        for (size_t i = 0; i < dim; i++) variance += x[i] * x[i];
        const float inv_rms = 1.0f / sqrtf(variance / (float)dim + epsilon);
        for (size_t i = 0; i < dim; i++) {
            y[i] = x[i] * inv_rms * weight[i] * qwen_ref_silu(z[i]);
        }
    }
}

bool ds4_qwen_ref_softmax_topk_f32(
        int32_t     *selected,
        float       *selected_weight,
        const float *logits,
        size_t       n_expert,
        size_t       n_selected) {
    if (!selected || !selected_weight || !logits || n_expert == 0 ||
        n_selected == 0 || n_selected > n_expert || n_expert > INT32_MAX) {
        return false;
    }

    float *probability = malloc(n_expert * sizeof(probability[0]));
    if (!probability) return false;

    float max_logit = logits[0];
    if (!isfinite(max_logit)) {
        free(probability);
        return false;
    }
    for (size_t i = 1; i < n_expert; i++) {
        if (!isfinite(logits[i])) {
            free(probability);
            return false;
        }
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    float total = 0.0f;
    for (size_t i = 0; i < n_expert; i++) {
        probability[i] = expf(logits[i] - max_logit);
        total += probability[i];
    }
    if (!(total > 0.0f) || !isfinite(total)) {
        free(probability);
        return false;
    }
    for (size_t i = 0; i < n_expert; i++) probability[i] /= total;

    for (size_t slot = 0; slot < n_selected; slot++) {
        size_t best = n_expert;
        for (size_t expert = 0; expert < n_expert; expert++) {
            bool already_selected = false;
            for (size_t prior = 0; prior < slot; prior++) {
                if (selected[prior] == (int32_t)expert) {
                    already_selected = true;
                    break;
                }
            }
            if (already_selected) continue;
            if (best == n_expert || probability[expert] > probability[best] ||
                (probability[expert] == probability[best] && expert < best)) {
                best = expert;
            }
        }
        selected[slot] = (int32_t)best;
        selected_weight[slot] = probability[best];
    }

    float selected_total = 0.0f;
    for (size_t i = 0; i < n_selected; i++) selected_total += selected_weight[i];
    if (!(selected_total > 0.0f) || !isfinite(selected_total)) {
        free(probability);
        return false;
    }
    for (size_t i = 0; i < n_selected; i++) selected_weight[i] /= selected_total;
    free(probability);
    return true;
}

void ds4_qwen_ref_sigmoid_gate_f32(
        float       *output,
        const float *input,
        const float *gate_logit,
        size_t       n_vector,
        size_t       dim) {
    if (!output || !input || !gate_logit) return;
    for (size_t v = 0; v < n_vector; v++) {
        const float gate = qwen_ref_sigmoid(gate_logit[v]);
        for (size_t i = 0; i < dim; i++) {
            output[v * dim + i] = input[v * dim + i] * gate;
        }
    }
}
