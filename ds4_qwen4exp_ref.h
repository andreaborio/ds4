#ifndef DS4_QWEN4EXP_REF_H
#define DS4_QWEN4EXP_REF_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ds4_qwen4exp.h"

/* Scalar, allocation-free primitives for the pinned Qwen4Exp equations.
 * They favor explicit layouts and deterministic failure over throughput.
 * Unless documented otherwise, output buffers must not overlap inputs. */

bool ds4_qwen4exp_ref_zero_centered_rmsnorm_f32(
        float       *output,
        const float *input,
        const float *weight,
        size_t       n_vector,
        size_t       dim,
        float        epsilon);

bool ds4_qwen4exp_ref_sigmoid_gated_rmsnorm_f32(
        float       *output,
        const float *input,
        const float *gate_logit,
        const float *weight,
        size_t       n_vector,
        size_t       dim,
        float        epsilon);

/* Matrices are row-major [output][input].  residual/norm_weight are
 * [stream][dim], down is [rank][stream*dim], up is [stream*dim][rank], and
 * inject is [stream][stream*dim]. */
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
        float        epsilon);

bool ds4_qwen4exp_ref_gr_apply_f32(
        float       *residual,
        const float *block_output,
        const float *injection,
        size_t       n_stream,
        size_t       dim);

bool ds4_qwen4exp_ref_gr_final_mix_f32(
        float       *output,
        const float *residual,
        const float *norm_weight,
        const float *down,
        const float *up,
        size_t       n_stream,
        size_t       dim,
        size_t       rank,
        float        epsilon);

/* input/output: [token][channel], weight: [channel][kernel], state:
 * [channel][kernel-1], oldest to newest. */
bool ds4_qwen4exp_ref_causal_conv1d_silu_f32(
        float       *output,
        float       *state,
        const float *input,
        const float *weight,
        size_t       n_token,
        size_t       n_channel,
        size_t       kernel);

bool ds4_qwen4exp_ref_gdn_controls_f32(
        float       *log_decay,
        float       *beta,
        const float *alpha_logit,
        const float *beta_logit,
        const float *a_log,
        const float *dt_bias,
        size_t       n_token,
        size_t       n_value_head);

/* Q/K are [token][key_head][key_dim], V/output are
 * [token][value_head][value_dim], and state is
 * [value_head][key_dim][value_dim].  Repeat-interleave maps value head h to
 * key head h/(value_head/key_head). */
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
        size_t       value_dim);

bool ds4_qwen4exp_ref_softmax_topk_f32(
        uint32_t    *selected,
        float       *selected_weight,
        const float *logits,
        size_t       n_expert,
        size_t       n_selected);

bool ds4_qwen4exp_ref_partial_rope_f32(
        float          *values,
        const uint32_t *position,
        size_t          n_token,
        size_t          n_head,
        size_t          head_dim,
        size_t          n_rot,
        float           theta);

/* raw_key is [group*compression][head_dim].  Each complete group is averaged,
 * zero-centered-normalized, then partially rotated at its first logical
 * position. */
bool ds4_qwen4exp_ref_qsa_group_keys_f32(
        float       *group_key,
        const float *raw_key,
        const float *norm_weight,
        size_t       n_group,
        size_t       compression,
        size_t       head_dim,
        size_t       n_rot,
        float        theta,
        float        epsilon);

/* query is [query_head][head_dim], group_key is [group][head_dim].  ReLU is
 * applied to each head dot before the head sum. */
bool ds4_qwen4exp_ref_qsa_scores_f32(
        float       *score,
        const float *query,
        const float *group_key,
        size_t       n_group,
        size_t       n_query_head,
        size_t       head_dim);

/* Select descending score/ascending group ID, expand complete groups, then
 * append the visible incomplete tail. */
bool ds4_qwen4exp_ref_qsa_select_positions(
        uint32_t    *position,
        size_t       position_capacity,
        size_t      *n_position,
        const float *score,
        size_t       visible_tokens,
        size_t       compression,
        size_t       group_budget);

typedef struct {
    uint32_t token[2];
    uint32_t count;
} ds4_qwen4exp_ple_history;

void ds4_qwen4exp_ref_ple_history_reset(ds4_qwen4exp_ple_history *history);

bool ds4_qwen4exp_ref_ple_history_advance(
        ds4_qwen4exp_ple_history       *after,
        const ds4_qwen4exp_ple_history *before,
        uint32_t                         token);

/* Computes rows without mutating history, so the caller can commit the
 * separately computed successor only after the whole token succeeds. */
bool ds4_qwen4exp_ref_ple_rows(
        uint32_t                         row[DS4_QWEN4EXP_PLE_HEADS],
        uint32_t                         current_token,
        const ds4_qwen4exp_ple_history *history,
        const ds4_qwen4exp_profile     *profile);

/* key/query are [stream][dim], value is [dim], output is [stream][dim]. */
bool ds4_qwen4exp_ref_ple_gate_f32(
        float       *output,
        const float *query,
        const float *key,
        const float *value,
        size_t       n_stream,
        size_t       dim);

/* Dilation state is [channel][dilation*(kernel-1)], oldest to newest.
 * Weight is [channel][kernel].  A successful call advances the state. */
bool ds4_qwen4exp_ref_dilated_conv1d_silu_f32(
        float       *output,
        float       *state,
        const float *input,
        const float *weight,
        size_t       n_token,
        size_t       n_channel,
        size_t       kernel,
        size_t       dilation);

#endif
