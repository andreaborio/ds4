#ifndef DS4_QWEN_REF_H
#define DS4_QWEN_REF_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Scalar post-GGUF-conversion Qwen reference operators.  These deliberately
 * favor a direct match to the published model equations over throughput;
 * Metal kernels are checked against them before entering the production graph. */

/* input/output: [token][channel], weight: [channel][kernel], state:
 * [channel][kernel - 1] from oldest to newest. */
void ds4_qwen_ref_causal_conv1d_silu_f32(
        float       *output,
        float       *state,
        const float *input,
        const float *weight,
        size_t       n_token,
        size_t       n_channel,
        size_t       kernel);

/* q/key: [token][key_head][key_dim], value/output:
 * [token][value_head][value_dim], log_decay/beta: [token][value_head],
 * state: [value_head][value_dim][key_dim].  The GGUF converter tiles V heads,
 * so runtime value head h uses key head h % n_key_head.  This transposed
 * physical state layout matches llama.cpp's contiguous CPU state. */
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
        size_t       value_dim);

/* GGUF stores ssm_a = -exp(HF A_log), already in V-head tiled order.  Do not
 * exponentiate it again: it directly scales softplus(alpha + dt). */
void ds4_qwen_ref_gated_delta_controls_f32(
        float       *log_decay,
        float       *beta,
        const float *alpha_logit,
        const float *beta_logit,
        const float *ssm_a,
        const float *dt_bias,
        size_t       n_token,
        size_t       n_value_head);

/* input/gate/output: [vector][dim], weight: [dim].  Qwen normalizes before
 * applying the SiLU gate. */
void ds4_qwen_ref_rmsnorm_gated_f32(
        float       *output,
        const float *input,
        const float *gate,
        const float *weight,
        size_t       n_vector,
        size_t       dim,
        float        epsilon);

/* Qwen router semantics: full F32 softmax, deterministic descending top-k,
 * then renormalization over only the selected experts.  Equal scores prefer
 * the lower expert id so model-free tests are stable across backends. */
bool ds4_qwen_ref_softmax_topk_f32(
        int32_t     *selected,
        float       *selected_weight,
        const float *logits,
        size_t       n_expert,
        size_t       n_selected);

/* Qwen's shared-expert gate produces one scalar per vector and broadcasts it
 * across the full hidden dimension before the routed and shared paths add. */
void ds4_qwen_ref_sigmoid_gate_f32(
        float       *output,
        const float *input,
        const float *gate_logit,
        size_t       n_vector,
        size_t       dim);

/* Full-attention reference helpers use the post-conversion GGUF layout.  The
 * fused Q projection stores [query, gate] inside every query-head block. */
bool ds4_qwen_ref_split_q_gate_f32(
        float       *query,
        float       *gate,
        const float *projection,
        size_t       n_token,
        size_t       n_query_head,
        size_t       head_dim);

/* Qwen3.5/3.6 RMSNorm weights are already shifted by +1 by the pinned GGUF
 * converter, so this helper multiplies by weight directly. */
bool ds4_qwen_ref_head_rms_norm_f32(
        float       *output,
        const float *input,
        const float *weight,
        size_t       n_token,
        size_t       n_head,
        size_t       head_dim,
        float        eps);

/* Text-only Qwen positions have identical T/H/W coordinates.  Interleaved
 * MRoPE therefore reduces to split-half RoPE on the first n_rot channels. */
bool ds4_qwen_ref_text_rope_f32(
        float          *values,
        const uint32_t *position,
        size_t          n_token,
        size_t          n_head,
        size_t          head_dim,
        size_t          n_rot,
        float           theta);

/* Scalar causal GQA over a complete sequence.  Query heads map to contiguous
 * repeated KV groups, matching repeat_interleave in the official fallback. */
bool ds4_qwen_ref_causal_gqa_f32(
        float       *output,
        const float *query,
        const float *key,
        const float *value,
        size_t       n_token,
        size_t       n_query_head,
        size_t       n_kv_head,
        size_t       head_dim);

/* Full-attention output gating is elementwise, unlike the shared-expert gate
 * above, which broadcasts one scalar across a hidden vector. */
void ds4_qwen_ref_sigmoid_gate_elements_f32(
        float       *output,
        const float *input,
        const float *gate_logit,
        size_t       n_value);

#endif
