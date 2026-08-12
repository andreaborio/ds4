// DS4 Metal routed-MoE matvec kernels.

#define QK_K 256
#define N_R0_Q2_K 4
#define N_R0_GLM_Q2_PAIR2_K 1
#define N_R0_Q4_K 2
#define N_R0_MLX_AFFINE4 4
#define N_R0_GLM_Q4_PAIR2_K 1
#define N_R0_GLM_Q4_PAIR_K 4
#define N_R0_Q5_PAIR_K 4
#define N_R0_Q5_K 4
#define N_R0_Q6_K 2
#define N_R0_IQ2_XXS 4
#define N_R0_IQ2_XS 4
#define N_R0_IQ3_XXS 4
#define N_R0_IQ4_XS 2

static constant uchar ds4_metal_kmask_iq2xs[8] = {
    1, 2, 4, 8, 16, 32, 64, 128
};

static constant uchar ds4_metal_ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

static constant ulong ds4_metal_iq2xxs_grid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

#define kmask_iq2xs ds4_metal_kmask_iq2xs
#define ksigns_iq2xs ds4_metal_ksigns_iq2xs
#define iq2xxs_grid ds4_metal_iq2xxs_grid
#define iq2xs_grid ds4_metal_iq2xs_grid
#define iq3xxs_grid ds4_metal_iq3xxs_grid
#define kvalues_iq4nl_f ds4_metal_kvalues_iq4nl_f

struct block_q2_K {
    uchar scales[QK_K/16];
    uchar qs[QK_K/4];
    half d;
    half dmin;
};

struct block_q4_K {
    half d;
    half dmin;
    uchar scales[12];
    uchar qs[QK_K/2];
};

// MLX affine 4-bit, group size 64.
struct block_mlx_affine4_64 {
    uchar qs[32];
    ushort scale_bf16;
    ushort bias_bf16;
};

struct block_q5_K {
    half d;
    half dmin;
    uchar scales[12];
    uchar qh[QK_K/8];
    uchar qs[QK_K/2];
};

struct block_q6_K {
    uchar ql[QK_K/2];
    uchar qh[QK_K/4];
    char scales[QK_K/16];
    half d;
};

struct block_iq2_xxs {
    half d;
    ushort qs[QK_K/8];
};

struct block_iq2_xs {
    half d;
    ushort qs[QK_K/8];
    uchar scales[QK_K/32];
};

struct block_iq3_xxs {
    half d;
    uchar qs[3*QK_K/8];
};

struct block_iq4_xs {
    half d;
    ushort scales_h;
    uchar scales_l[QK_K/64];
    uchar qs[QK_K/2];
};

struct ds4_metal_glm_routed_moe_args {
    uint32_t in_dim;
    uint32_t mid_dim;
    uint32_t out_dim;
    uint32_t n_total_expert;
    uint32_t n_expert_used;
    uint32_t n_tokens;
    uint32_t mid_token_stride;
    uint32_t down_type;
    uint32_t pad0;
    uint64_t gate_expert_bytes;
    uint64_t gate_row_bytes;
    uint64_t up_expert_bytes;
    uint64_t up_row_bytes;
    uint64_t down_expert_bytes;
    uint64_t down_row_bytes;
};

struct ds4_metal_dsv4_moe_swiglu_weight_args {
    uint32_t width;
    uint32_t rows;
    uint64_t gate_row_stride;
    uint64_t up_row_stride;
    uint64_t mid_row_stride;
    uint64_t weight_stride;
    uint32_t write_clamped;
    float clamp_value;
};

struct ds4_metal_dsv4_moe_sum_args {
    uint32_t width;
    uint32_t tokens;
    uint64_t src_token_stride;
    uint64_t dst_token_stride;
};

// Routed-MoE activation for the selected experts:
// clamp(gate), clamp(up), silu(gate) * up * route_weight.  Normal inference
// does not consume gate/up after this point, so the fast path avoids writing the
// clamped intermediates back.  A diagnostic env switch can restore those writes
// when comparing the old multi-kernel intermediate tensors.
kernel void kernel_dsv4_moe_swiglu_weight(
        constant ds4_metal_dsv4_moe_swiglu_weight_args &args,
        device char *gate,
        device char *up,
        device char *mid,
        device const char *weights,
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    if (row >= args.rows) return;

    device float *gate_row = (device float *)(gate + (uint64_t)row * args.gate_row_stride);
    device float *up_row   = (device float *)(up   + (uint64_t)row * args.up_row_stride);
    device float *mid_row  = (device float *)(mid  + (uint64_t)row * args.mid_row_stride);
    device const float *w  = (device const float *)(weights + (uint64_t)row * args.weight_stride);
    const float route_weight = w[0];
    const float c = args.clamp_value;

    for (uint i = tid; i < args.width; i += ntg) {
        float g = gate_row[i];
        float u = up_row[i];
        if (c > 1.0e-6f) {
            g = min(g, c);
            u = clamp(u, -c, c);
            if (args.write_clamped != 0) {
                gate_row[i] = g;
                up_row[i] = u;
            }
        }
        const float silu = g / (1.0f + exp(-g));
        mid_row[i] = silu * u * route_weight;
    }
}

// Same routed-MoE activation as above, but stores the down-projection input in
// half precision. The grouped matmul path converts F32 activations to half
// before MMA anyway, so this cuts the large mid write/read traffic without
// changing the effective matmul input precision.
kernel void kernel_dsv4_moe_swiglu_weight_f16(
        constant ds4_metal_dsv4_moe_swiglu_weight_args &args,
        device char *gate,
        device char *up,
        device char *mid,
        device const char *weights,
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    if (row >= args.rows) return;

    device float *gate_row = (device float *)(gate + (uint64_t)row * args.gate_row_stride);
    device float *up_row   = (device float *)(up   + (uint64_t)row * args.up_row_stride);
    device half  *mid_row  = (device half  *)(mid  + (uint64_t)row * args.mid_row_stride);
    device const float *w  = (device const float *)(weights + (uint64_t)row * args.weight_stride);
    const float route_weight = w[0];
    const float c = args.clamp_value;

    for (uint i = tid; i < args.width; i += ntg) {
        float g = gate_row[i];
        float u = up_row[i];
        if (c > 1.0e-6f) {
            g = min(g, c);
            u = clamp(u, -c, c);
            if (args.write_clamped != 0) {
                gate_row[i] = g;
                up_row[i] = u;
            }
        }
        const float silu = g / (1.0f + exp(-g));
        mid_row[i] = (half)(silu * u * route_weight);
    }
}

kernel void kernel_dsv4_moe_sum6_f32(
        constant ds4_metal_dsv4_moe_sum_args &args,
        device const char *src,
        device       char *dst,
        uint token[[threadgroup_position_in_grid]],
        uint tid[[thread_position_in_threadgroup]],
        uint ntg[[threads_per_threadgroup]]) {
    if (token >= args.tokens) return;

    device const float *s =
        (device const float *)(src + (uint64_t)token * args.src_token_stride);
    device float *d =
        (device float *)(dst + (uint64_t)token * args.dst_token_stride);

    for (uint col = tid; col < args.width; col += ntg) {
        float v = s[col];
        v += s[args.width + col];
        v += s[2u * args.width + col];
        v += s[3u * args.width + col];
        v += s[4u * args.width + col];
        v += s[5u * args.width + col];
        d[col] = v;
    }
}

/* Qwen3.6 routes every token to eight experts.  Keep the reduction in one
 * dispatch so the resident path does not serialize seven generic add kernels
 * after the down projection.  The accumulation order matches the historical
 * two-top-4 path: ((0+1)+2)+3, ((4+5)+6)+7, then left + right. */
kernel void kernel_qwen35_moe_sum8_f32(
        constant ds4_metal_dsv4_moe_sum_args &args,
        device const char *src,
        device       char *dst,
        uint token[[threadgroup_position_in_grid]],
        uint tid[[thread_position_in_threadgroup]],
        uint ntg[[threads_per_threadgroup]]) {
    if (token >= args.tokens) return;

    device const float *s =
        (device const float *)(src + (uint64_t)token * args.src_token_stride);
    device float *d =
        (device float *)(dst + (uint64_t)token * args.dst_token_stride);

    for (uint col = tid; col < args.width; col += ntg) {
        float left = s[col] + s[args.width + col];
        left += s[2u * args.width + col];
        left += s[3u * args.width + col];
        float right = s[4u * args.width + col] +
                      s[5u * args.width + col];
        right += s[6u * args.width + col];
        right += s[7u * args.width + col];
        d[col] = left + right;
    }
}

template <typename type4x4>
void dequantize_q2_K(device const block_q2_K *xb, short il, thread type4x4 & reg) {
    const float d = xb->d;
    const float min = xb->dmin;
    device const uint8_t * q = (device const uint8_t *)xb->qs;
    float dl, ml;
    uint8_t sc = xb->scales[il];

    q = q + 32*(il/8) + 16*(il&1);
    il = (il/2)%4;

    half  coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);
    uchar mask = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);
    dl = d * (sc & 0xF) * coef, ml = min * (sc >> 4);
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

static inline float ds4_glm_q2_K_value(device const block_q2_K *blocks, uint k) {
    const uint block = k / QK_K;
    const uint idx = k - block * QK_K;
    device const block_q2_K *xb = blocks + block;
    const uint group = idx / 16u;
    const uint l = idx - group * 16u;
    const uint q_base = 32u * (group / 8u) + 16u * (group & 1u);
    const uint shift = ((group / 2u) & 3u) * 2u;
    const uint q = ((uint)xb->qs[q_base + l] >> shift) & 0x03u;
    const uint sc = (uint)xb->scales[group];
    return (float)xb->d * (float)(sc & 0x0fu) * (float)q -
           (float)xb->dmin * (float)(sc >> 4u);
}

static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)),
                          uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

static inline float ds4_glm_q4_K_value(device const block_q4_K *blocks, uint k) {
    const uint block = k / QK_K;
    const uint idx = k - block * QK_K;
    device const block_q4_K *xb = blocks + block;
    const uint group = idx / 32u;
    const uint l = idx - group * 32u;
    const uchar2 sm = get_scale_min_k4_just2((int)group, 0, xb->scales);
    const uint byte_off = (group >> 1u) * 32u + l;
    const uint shift = (group & 1u) * 4u;
    const uint q = (xb->qs[byte_off] >> shift) & 0x0Fu;
    return (float)xb->d * (float)sm.x * (float)q -
           (float)xb->dmin * (float)sm.y;
}

static inline float ds4_glm_q5_K_value(device const block_q5_K *blocks, uint k) {
    const uint block = k / QK_K;
    const uint idx = k - block * QK_K;
    device const block_q5_K *xb = blocks + block;
    const uint group = idx / 32u;
    const uint l = idx - group * 32u;
    const uchar2 sm = get_scale_min_k4_just2((int)group, 0, xb->scales);
    const uint ql_base = (group >> 1u) * 32u + l;
    const uint shift = (group & 1u) * 4u;
    uint q = (xb->qs[ql_base] >> shift) & 0x0Fu;
    q += (xb->qh[l] & (uchar)(1u << group)) ? 16u : 0u;
    return (float)xb->d * (float)sm.x * (float)q -
           (float)xb->dmin * (float)sm.y;
}

static inline float ds4_glm_q6_K_value(device const block_q6_K *blocks, uint k) {
    const uint block = k / QK_K;
    const uint idx = k - block * QK_K;
    device const block_q6_K *xb = blocks + block;
    const uint n128 = idx >> 7u;
    const uint r = idx & 127u;
    const uint l = r & 31u;
    const uint quarter = r >> 5u;
    const uint ql_base = n128 * 64u;
    const uint qh_base = n128 * 32u;
    const uint sc_base = n128 * 8u;
    uint q;
    int sc;

    if (quarter == 0u) {
        q = (xb->ql[ql_base + l] & 0x0Fu) | (((xb->qh[qh_base + l] >> 0u) & 3u) << 4u);
        sc = (int)xb->scales[sc_base + l / 16u + 0u];
    } else if (quarter == 1u) {
        q = (xb->ql[ql_base + 32u + l] & 0x0Fu) | (((xb->qh[qh_base + l] >> 2u) & 3u) << 4u);
        sc = (int)xb->scales[sc_base + l / 16u + 2u];
    } else if (quarter == 2u) {
        q = (xb->ql[ql_base + l] >> 4u) | (((xb->qh[qh_base + l] >> 4u) & 3u) << 4u);
        sc = (int)xb->scales[sc_base + l / 16u + 4u];
    } else {
        q = (xb->ql[ql_base + 32u + l] >> 4u) | (((xb->qh[qh_base + l] >> 6u) & 3u) << 4u);
        sc = (int)xb->scales[sc_base + l / 16u + 6u];
    }

    return (float)xb->d * (float)sc * (float)((int)q - 32);
}

kernel void kernel_glm_q4_K_pair_swiglu_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *gate,
        device const char *up,
        device const float *x,
        device const int32_t *selected,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    const uint ntg = 256u;
    const uint row = tgpig.x;
    const uint slot = tgpig.y;
    const uint token = tgpig.z;
    if (row >= args.mid_dim || slot >= args.n_expert_used || token >= args.n_tokens) return;

    const uint64_t selected_off = (uint64_t)token * args.n_expert_used + slot;
    const uint64_t mid_off = (uint64_t)token * args.mid_token_stride +
                             (uint64_t)slot * args.mid_dim + row;
    const int expert = selected[selected_off];
    if (expert < 0 || (uint)expert >= args.n_total_expert) {
        if (tid == 0u) mid[mid_off] = 0.0f;
        return;
    }

    device const block_q4_K *gate_row =
        (device const block_q4_K *)(gate +
            (uint64_t)(uint)expert * args.gate_expert_bytes +
            (uint64_t)row * args.gate_row_bytes);
    device const block_q4_K *up_row =
        (device const block_q4_K *)(up +
            (uint64_t)(uint)expert * args.up_expert_bytes +
            (uint64_t)row * args.up_row_bytes);

    float acc_gate = 0.0f;
    float acc_up = 0.0f;
    device const float *token_x = x + (uint64_t)token * args.in_dim;
    for (uint k = tid; k < args.in_dim; k += ntg) {
        const float xv = token_x[k];
        acc_gate += ds4_glm_q4_K_value(gate_row, k) * xv;
        acc_up += ds4_glm_q4_K_value(up_row, k) * xv;
    }

    scratch[tid] = acc_gate;
    scratch[ntg + tid] = acc_up;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = ntg >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
            scratch[ntg + tid] += scratch[ntg + tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        const float g = scratch[0];
        const float u = scratch[ntg];
        const float sw = g / (1.0f + exp(-g));
        mid[mid_off] = sw * u * weights[selected_off];
    }
}

template <short N_R0>
static inline void glm_q2_K_pair_swiglu_simd_f32_impl(
        ds4_metal_glm_routed_moe_args args,
        device const char *gate,
        device const char *up,
        device const float *x,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch,
        uint3 tgpig,
        uint slot,
        uint token,
        uint64_t selected_off,
        int expert,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = 2;
    const uint row0 = ((uint)tgpig.x * (uint)NSG + (uint)sgitg) * N_R0;
    if (row0 >= args.mid_dim || slot >= args.n_expert_used || token >= args.n_tokens) return;

    const uint64_t mid_base = (uint64_t)token * args.mid_token_stride +
                              (uint64_t)slot * args.mid_dim;
    if (expert < 0 || (uint)expert >= args.n_total_expert) {
        if (tiisg == 0u) {
            for (short row = 0;
                 row < N_R0 && row0 + (uint)row < args.mid_dim;
                 row++) {
                mid[mid_base + row0 + (uint)row] = 0.0f;
            }
        }
        return;
    }

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const short is = (8 * ir) / 16;
    const int nb = args.in_dim / QK_K;
    const uint64_t expert_gate = (uint64_t)(uint)expert * args.gate_expert_bytes;
    const uint64_t expert_up = (uint64_t)(uint)expert * args.up_expert_bytes;
    device const block_q2_K *xg =
        (device const block_q2_K *)(gate + expert_gate +
            (uint64_t)row0 * args.gate_row_bytes);
    device const block_q2_K *xu =
        (device const block_q2_K *)(up + expert_up +
            (uint64_t)row0 * args.up_row_bytes);
    device const float *y = x + (uint64_t)token * args.in_dim;
    device const float *y4 = y + ix * QK_K + 128 * iq + 8 * ir;

    float yl[32];
    float sumg[N_R0] = {0.f};
    float sumu[N_R0] = {0.f};

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i +  0] = y4[i +  0]; sumy[0] += yl[i +  0];
            yl[i +  8] = y4[i + 32]; sumy[1] += yl[i +  8];
            yl[i + 16] = y4[i + 64]; sumy[2] += yl[i + 16];
            yl[i + 24] = y4[i + 96]; sumy[3] += yl[i + 24];
        }

        device const uint8_t *scg = (device const uint8_t *)xg[ib].scales + 8 * iq + is;
        device const uint16_t *qsg = (device const uint16_t *)xg[ib].qs + 16 * iq + 4 * ir;
        device const half *dhg = &xg[ib].d;
        device const uint8_t *scu = (device const uint8_t *)xu[ib].scales + 8 * iq + is;
        device const uint16_t *qsu = (device const uint16_t *)xu[ib].qs + 16 * iq + 4 * ir;
        device const half *dhu = &xu[ib].d;

        for (short row = 0;
             row < N_R0 && row0 + (uint)row < args.mid_dim;
             row++) {
            float4 acc1g = {0.f, 0.f, 0.f, 0.f};
            float4 acc2g = {0.f, 0.f, 0.f, 0.f};
            float4 acc1u = {0.f, 0.f, 0.f, 0.f};
            float4 acc2u = {0.f, 0.f, 0.f, 0.f};

            for (int i = 0; i < 8; i += 2) {
                acc1g[0] += yl[i +  0] * (qsg[i / 2] & 0x0003);
                acc2g[0] += yl[i +  1] * (qsg[i / 2] & 0x0300);
                acc1g[1] += yl[i +  8] * (qsg[i / 2] & 0x000c);
                acc2g[1] += yl[i +  9] * (qsg[i / 2] & 0x0c00);
                acc1g[2] += yl[i + 16] * (qsg[i / 2] & 0x0030);
                acc2g[2] += yl[i + 17] * (qsg[i / 2] & 0x3000);
                acc1g[3] += yl[i + 24] * (qsg[i / 2] & 0x00c0);
                acc2g[3] += yl[i + 25] * (qsg[i / 2] & 0xc000);

                acc1u[0] += yl[i +  0] * (qsu[i / 2] & 0x0003);
                acc2u[0] += yl[i +  1] * (qsu[i / 2] & 0x0300);
                acc1u[1] += yl[i +  8] * (qsu[i / 2] & 0x000c);
                acc2u[1] += yl[i +  9] * (qsu[i / 2] & 0x0c00);
                acc1u[2] += yl[i + 16] * (qsu[i / 2] & 0x0030);
                acc2u[2] += yl[i + 17] * (qsu[i / 2] & 0x3000);
                acc1u[3] += yl[i + 24] * (qsu[i / 2] & 0x00c0);
                acc2u[3] += yl[i + 25] * (qsu[i / 2] & 0xc000);
            }

            const float dg = dhg[0];
            const float mg = dhg[1] * 1.f / 16.f;
            sumg[row] += dg * ((acc1g[0] + 1.f / 256.f * acc2g[0]) * (scg[0] & 0xF) * 1.f /  1.f +
                               (acc1g[1] + 1.f / 256.f * acc2g[1]) * (scg[2] & 0xF) * 1.f /  4.f +
                               (acc1g[2] + 1.f / 256.f * acc2g[2]) * (scg[4] & 0xF) * 1.f / 16.f +
                               (acc1g[3] + 1.f / 256.f * acc2g[3]) * (scg[6] & 0xF) * 1.f / 64.f) -
                         mg * (sumy[0] * (scg[0] & 0xF0) + sumy[1] * (scg[2] & 0xF0) +
                               sumy[2] * (scg[4] & 0xF0) + sumy[3] * (scg[6] & 0xF0));

            const float du = dhu[0];
            const float mu = dhu[1] * 1.f / 16.f;
            sumu[row] += du * ((acc1u[0] + 1.f / 256.f * acc2u[0]) * (scu[0] & 0xF) * 1.f /  1.f +
                               (acc1u[1] + 1.f / 256.f * acc2u[1]) * (scu[2] & 0xF) * 1.f /  4.f +
                               (acc1u[2] + 1.f / 256.f * acc2u[2]) * (scu[4] & 0xF) * 1.f / 16.f +
                               (acc1u[3] + 1.f / 256.f * acc2u[3]) * (scu[6] & 0xF) * 1.f / 64.f) -
                         mu * (sumy[0] * (scu[0] & 0xF0) + sumy[1] * (scu[2] & 0xF0) +
                               sumy[2] * (scu[4] & 0xF0) + sumy[3] * (scu[6] & 0xF0));

            qsg += args.gate_row_bytes / 2;
            scg += args.gate_row_bytes;
            dhg += args.gate_row_bytes / 2;
            qsu += args.up_row_bytes / 2;
            scu += args.up_row_bytes;
            dhu += args.up_row_bytes / 2;
        }

        y4 += 4 * QK_K;
    }

    for (short row = 0;
         row < N_R0 && row0 + (uint)row < args.mid_dim;
         row++) {
        const float g = simd_sum(sumg[row]);
        const float u = simd_sum(sumu[row]);
        if (tiisg == 0u) {
            const float sw = g / (1.0f + exp(-g));
            mid[mid_base + row0 + (uint)row] = sw * u * weights[selected_off];
        }
    }

    (void)scratch;
}

kernel void kernel_glm_q2_K_pair_swiglu_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *gate,
        device const char *up,
        device const float *x,
        device const int32_t *selected,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint slot = tgpig.y;
    const uint token = tgpig.z;
    if (slot >= args.n_expert_used || token >= args.n_tokens) return;
    const uint64_t selected_off = (uint64_t)token * args.n_expert_used + slot;
    const int expert = selected[selected_off];
    glm_q2_K_pair_swiglu_simd_f32_impl<N_R0_Q2_K>(
        args, gate, up, x, weights, mid, scratch,
        tgpig, slot, token, selected_off, expert, tiisg, sgitg);
}

kernel void kernel_glm_q2_K_addr_pair_swiglu2_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const uint64_t *gate_addrs,
        device const uint64_t *up_addrs,
        device const float *x,
        device const int32_t *selected,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint slot = tgpig.y;
    const uint token = tgpig.z;
    if (slot >= args.n_expert_used || token >= args.n_tokens) return;

    const uint64_t selected_off = (uint64_t)token * args.n_expert_used + slot;
    const int expert = selected[selected_off];
    const short NSG = 2;
    const uint row0 = ((uint)tgpig.x * (uint)NSG + (uint)sgitg) * N_R0_GLM_Q2_PAIR2_K;
    const uint64_t mid_base = (uint64_t)token * args.mid_token_stride +
                              (uint64_t)slot * args.mid_dim;
    if (row0 >= args.mid_dim) return;

    if (expert < 0 || (uint)expert >= args.n_total_expert) {
        if (tiisg == 0u) {
            for (short row = 0;
                 row < N_R0_GLM_Q2_PAIR2_K && row0 + (uint)row < args.mid_dim;
                 row++) {
                mid[mid_base + row0 + (uint)row] = 0.0f;
            }
        }
        return;
    }
    const uint64_t gate_addr = gate_addrs[(uint)expert];
    const uint64_t up_addr = up_addrs[(uint)expert];
    if (gate_addr == 0 || up_addr == 0) {
        if (tiisg == 0u) {
            for (short row = 0;
                 row < N_R0_GLM_Q2_PAIR2_K && row0 + (uint)row < args.mid_dim;
                 row++) {
                mid[mid_base + row0 + (uint)row] = 0.0f;
            }
        }
        return;
    }

    ds4_metal_glm_routed_moe_args local = args;
    local.n_total_expert = 1;
    local.gate_expert_bytes = 0;
    local.up_expert_bytes = 0;
    glm_q2_K_pair_swiglu_simd_f32_impl<N_R0_GLM_Q2_PAIR2_K>(
        local,
        reinterpret_cast<device const char *>(gate_addr),
        reinterpret_cast<device const char *>(up_addr),
        x, weights, mid, scratch,
        tgpig, slot, token, selected_off, 0, tiisg, sgitg);
}

kernel void kernel_glm_q2_K_addr_pair_swiglu2_f32_masked(
        constant ds4_metal_glm_routed_moe_args &args,
        constant uint32_t &active_mask,
        device const uint64_t *gate_addrs,
        device const uint64_t *up_addrs,
        device const float *x,
        device const int32_t *selected,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint slot = tgpig.y;
    const uint token = tgpig.z;
    if (slot >= args.n_expert_used || token >= args.n_tokens) return;
    if ((active_mask & (1u << slot)) == 0u) return;

    const uint64_t selected_off = (uint64_t)token * args.n_expert_used + slot;
    const int expert = selected[selected_off];
    const short NSG = 2;
    const uint row0 = ((uint)tgpig.x * (uint)NSG + (uint)sgitg) * N_R0_GLM_Q2_PAIR2_K;
    const uint64_t mid_base = (uint64_t)token * args.mid_token_stride +
                              (uint64_t)slot * args.mid_dim;
    if (row0 >= args.mid_dim) return;

    if (expert < 0 || (uint)expert >= args.n_total_expert) {
        if (tiisg == 0u) {
            for (short row = 0;
                 row < N_R0_GLM_Q2_PAIR2_K && row0 + (uint)row < args.mid_dim;
                 row++) {
                mid[mid_base + row0 + (uint)row] = 0.0f;
            }
        }
        return;
    }
    const uint64_t gate_addr = gate_addrs[(uint)expert];
    const uint64_t up_addr = up_addrs[(uint)expert];
    if (gate_addr == 0 || up_addr == 0) {
        if (tiisg == 0u) {
            for (short row = 0;
                 row < N_R0_GLM_Q2_PAIR2_K && row0 + (uint)row < args.mid_dim;
                 row++) {
                mid[mid_base + row0 + (uint)row] = 0.0f;
            }
        }
        return;
    }

    ds4_metal_glm_routed_moe_args local = args;
    local.n_total_expert = 1;
    local.gate_expert_bytes = 0;
    local.up_expert_bytes = 0;
    glm_q2_K_pair_swiglu_simd_f32_impl<N_R0_GLM_Q2_PAIR2_K>(
        local,
        reinterpret_cast<device const char *>(gate_addr),
        reinterpret_cast<device const char *>(up_addr),
        x, weights, mid, scratch,
        tgpig, slot, token, selected_off, 0, tiisg, sgitg);
}

template <short N_R0>
static inline void glm_q4_K_pair_swiglu_simd_f32_impl(
        ds4_metal_glm_routed_moe_args args,
        device const char *gate,
        device const char *up,
        device const float *x,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch,
        uint3 tgpig,
        uint slot,
        uint token,
        uint64_t selected_off,
        int expert,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = 2;
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const uint row0 = ((uint)tgpig.x * (uint)NSG + (uint)sgitg) * N_R0;
    if (row0 >= args.mid_dim || slot >= args.n_expert_used || token >= args.n_tokens) return;

    const uint64_t mid_base = (uint64_t)token * args.mid_token_stride +
                              (uint64_t)slot * args.mid_dim;
    if (expert < 0 || (uint)expert >= args.n_total_expert) {
        if (tiisg == 0u) {
            for (short row = 0;
                 row < N_R0 && row0 + (uint)row < args.mid_dim;
                 row++) {
                mid[mid_base + row0 + (uint)row] = 0.0f;
            }
        }
        return;
    }

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const int nb = args.in_dim / QK_K;
    const uint64_t expert_gate = (uint64_t)(uint)expert * args.gate_expert_bytes;
    const uint64_t expert_up = (uint64_t)(uint)expert * args.up_expert_bytes;
    device const block_q4_K *xg =
        (device const block_q4_K *)(gate + expert_gate +
            (uint64_t)row0 * args.gate_row_bytes);
    device const block_q4_K *xu =
        (device const block_q4_K *)(up + expert_up +
            (uint64_t)row0 * args.up_row_bytes);
    device const float *y = x + (uint64_t)token * args.in_dim;
    device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    float sumg[N_R0] = {0.f};
    float sumu[N_R0] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        float yl[16];
        float yh[16];
        float4 sumy = {0.f, 0.f, 0.f, 0.f};

        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
            yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
            yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
        }

        device const uint16_t *scg = (device const uint16_t *)xg[ib].scales + iq;
        device const uint16_t *qg1 = (device const uint16_t *)xg[ib].qs + 16 * iq + 4 * ir;
        device const half *dhg = &xg[ib].d;
        device const uint16_t *scu = (device const uint16_t *)xu[ib].scales + iq;
        device const uint16_t *qu1 = (device const uint16_t *)xu[ib].qs + 16 * iq + 4 * ir;
        device const half *dhu = &xu[ib].d;

        for (short row = 0;
             row < N_R0 && row0 + (uint)row < args.mid_dim;
             row++) {
            sc16[0] = scg[0] & kmask1;
            sc16[1] = scg[2] & kmask1;
            sc16[2] = ((scg[4] >> 0) & kmask2) | ((scg[0] & kmask3) >> 2);
            sc16[3] = ((scg[4] >> 4) & kmask2) | ((scg[2] & kmask3) >> 2);

            device const uint16_t *qg2 = qg1 + 32;
            float4 acc1g = {0.f, 0.f, 0.f, 0.f};
            float4 acc2g = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                acc1g[0] += yl[2 * i + 0] * (qg1[i] & 0x000F);
                acc1g[1] += yl[2 * i + 1] * (qg1[i] & 0x0F00);
                acc1g[2] += yl[2 * i + 8] * (qg1[i] & 0x00F0);
                acc1g[3] += yl[2 * i + 9] * (qg1[i] & 0xF000);
                acc2g[0] += yh[2 * i + 0] * (qg2[i] & 0x000F);
                acc2g[1] += yh[2 * i + 1] * (qg2[i] & 0x0F00);
                acc2g[2] += yh[2 * i + 8] * (qg2[i] & 0x00F0);
                acc2g[3] += yh[2 * i + 9] * (qg2[i] & 0xF000);
            }

            sumg[row] += dhg[0] * ((acc1g[0] + 1.f / 256.f * acc1g[1]) * sc8[0] +
                                   (acc1g[2] + 1.f / 256.f * acc1g[3]) * sc8[1] * 1.f / 16.f +
                                   (acc2g[0] + 1.f / 256.f * acc2g[1]) * sc8[4] +
                                   (acc2g[2] + 1.f / 256.f * acc2g[3]) * sc8[5] * 1.f / 16.f) -
                         dhg[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                   sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            sc16[0] = scu[0] & kmask1;
            sc16[1] = scu[2] & kmask1;
            sc16[2] = ((scu[4] >> 0) & kmask2) | ((scu[0] & kmask3) >> 2);
            sc16[3] = ((scu[4] >> 4) & kmask2) | ((scu[2] & kmask3) >> 2);

            device const uint16_t *qu2 = qu1 + 32;
            float4 acc1u = {0.f, 0.f, 0.f, 0.f};
            float4 acc2u = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                acc1u[0] += yl[2 * i + 0] * (qu1[i] & 0x000F);
                acc1u[1] += yl[2 * i + 1] * (qu1[i] & 0x0F00);
                acc1u[2] += yl[2 * i + 8] * (qu1[i] & 0x00F0);
                acc1u[3] += yl[2 * i + 9] * (qu1[i] & 0xF000);
                acc2u[0] += yh[2 * i + 0] * (qu2[i] & 0x000F);
                acc2u[1] += yh[2 * i + 1] * (qu2[i] & 0x0F00);
                acc2u[2] += yh[2 * i + 8] * (qu2[i] & 0x00F0);
                acc2u[3] += yh[2 * i + 9] * (qu2[i] & 0xF000);
            }

            sumu[row] += dhu[0] * ((acc1u[0] + 1.f / 256.f * acc1u[1]) * sc8[0] +
                                   (acc1u[2] + 1.f / 256.f * acc1u[3]) * sc8[1] * 1.f / 16.f +
                                   (acc2u[0] + 1.f / 256.f * acc2u[1]) * sc8[4] +
                                   (acc2u[2] + 1.f / 256.f * acc2u[3]) * sc8[5] * 1.f / 16.f) -
                         dhu[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                   sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            qg1 += args.gate_row_bytes / 2;
            scg += args.gate_row_bytes / 2;
            dhg += args.gate_row_bytes / 2;
            qu1 += args.up_row_bytes / 2;
            scu += args.up_row_bytes / 2;
            dhu += args.up_row_bytes / 2;
        }

        y4 += 4 * QK_K;
    }

    for (int row = 0;
         row < N_R0 && row0 + (uint)row < args.mid_dim;
         ++row) {
        const float g = simd_sum(sumg[row]);
        const float u = simd_sum(sumu[row]);
        if (tiisg == 0u) {
            const float sw = g / (1.0f + exp(-g));
            mid[mid_base + row0 + (uint)row] = sw * u * weights[selected_off];
        }
    }

    (void)scratch;
}

kernel void kernel_glm_q4_K_pair_swiglu2_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *gate,
        device const char *up,
        device const float *x,
        device const int32_t *selected,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint slot = tgpig.y;
    const uint token = tgpig.z;
    if (slot >= args.n_expert_used || token >= args.n_tokens) return;
    const uint64_t selected_off = (uint64_t)token * args.n_expert_used + slot;
    const int expert = selected[selected_off];
    glm_q4_K_pair_swiglu_simd_f32_impl<N_R0_GLM_Q4_PAIR2_K>(
        args, gate, up, x, weights, mid, scratch,
        tgpig, slot, token, selected_off, expert, tiisg, sgitg);
}

kernel void kernel_glm_q4_K_addr_pair_swiglu_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const uint64_t *gate_addrs,
        device const uint64_t *up_addrs,
        device const float *x,
        device const int32_t *selected,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint slot = tgpig.y;
    const uint token = tgpig.z;
    if (slot >= args.n_expert_used || token >= args.n_tokens) return;

    const uint64_t selected_off = (uint64_t)token * args.n_expert_used + slot;
    const int expert = selected[selected_off];
    const short NSG = 2;
    const uint row0 = ((uint)tgpig.x * (uint)NSG + (uint)sgitg) * N_R0_Q4_K;
    const uint64_t mid_base = (uint64_t)token * args.mid_token_stride +
                              (uint64_t)slot * args.mid_dim;
    if (row0 >= args.mid_dim) return;

    if (expert < 0 || (uint)expert >= args.n_total_expert) {
        if (tiisg == 0u) {
            for (short row = 0;
                 row < N_R0_Q4_K && row0 + (uint)row < args.mid_dim;
                 row++) {
                mid[mid_base + row0 + (uint)row] = 0.0f;
            }
        }
        return;
    }
    const uint64_t gate_addr = gate_addrs[(uint)expert];
    const uint64_t up_addr = up_addrs[(uint)expert];
    if (gate_addr == 0 || up_addr == 0) {
        if (tiisg == 0u) {
            for (short row = 0;
                 row < N_R0_Q4_K && row0 + (uint)row < args.mid_dim;
                 row++) {
                mid[mid_base + row0 + (uint)row] = 0.0f;
            }
        }
        return;
    }

    ds4_metal_glm_routed_moe_args local = args;
    local.n_total_expert = 1;
    local.gate_expert_bytes = 0;
    local.up_expert_bytes = 0;
    glm_q4_K_pair_swiglu_simd_f32_impl<N_R0_Q4_K>(
        local,
        reinterpret_cast<device const char *>(gate_addr),
        reinterpret_cast<device const char *>(up_addr),
        x, weights, mid, scratch,
        tgpig, slot, token, selected_off, 0, tiisg, sgitg);
}

kernel void kernel_glm_q4_K_addr_pair_swiglu_f32_masked(
        constant ds4_metal_glm_routed_moe_args &args,
        constant uint32_t &active_mask,
        device const uint64_t *gate_addrs,
        device const uint64_t *up_addrs,
        device const float *x,
        device const int32_t *selected,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint slot = tgpig.y;
    const uint token = tgpig.z;
    if (slot >= args.n_expert_used || token >= args.n_tokens) return;
    if ((active_mask & (1u << slot)) == 0u) return;

    const uint64_t selected_off = (uint64_t)token * args.n_expert_used + slot;
    const int expert = selected[selected_off];
    const short NSG = 2;
    const uint row0 = ((uint)tgpig.x * (uint)NSG + (uint)sgitg) * N_R0_Q4_K;
    const uint64_t mid_base = (uint64_t)token * args.mid_token_stride +
                              (uint64_t)slot * args.mid_dim;
    if (row0 >= args.mid_dim) return;

    if (expert < 0 || (uint)expert >= args.n_total_expert) {
        if (tiisg == 0u) {
            for (short row = 0;
                 row < N_R0_Q4_K && row0 + (uint)row < args.mid_dim;
                 row++) {
                mid[mid_base + row0 + (uint)row] = 0.0f;
            }
        }
        return;
    }
    const uint64_t gate_addr = gate_addrs[(uint)expert];
    const uint64_t up_addr = up_addrs[(uint)expert];
    if (gate_addr == 0 || up_addr == 0) {
        if (tiisg == 0u) {
            for (short row = 0;
                 row < N_R0_Q4_K && row0 + (uint)row < args.mid_dim;
                 row++) {
                mid[mid_base + row0 + (uint)row] = 0.0f;
            }
        }
        return;
    }

    ds4_metal_glm_routed_moe_args local = args;
    local.n_total_expert = 1;
    local.gate_expert_bytes = 0;
    local.up_expert_bytes = 0;
    glm_q4_K_pair_swiglu_simd_f32_impl<N_R0_Q4_K>(
        local,
        reinterpret_cast<device const char *>(gate_addr),
        reinterpret_cast<device const char *>(up_addr),
        x, weights, mid, scratch,
        tgpig, slot, token, selected_off, 0, tiisg, sgitg);
}

kernel void kernel_glm_q4_K_slots6_pair_swiglu_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *gate0,
        device const char *gate1,
        device const char *gate2,
        device const char *gate3,
        device const char *gate4,
        device const char *gate5,
        device const char *up0,
        device const char *up1,
        device const char *up2,
        device const char *up3,
        device const char *up4,
        device const char *up5,
        device const float *x,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint slot = tgpig.y;
    const uint token = tgpig.z;
    if (slot >= args.n_expert_used || token >= args.n_tokens) return;

    device const char *gate_cur = gate0;
    device const char *up_cur = up0;
    switch (slot) {
    case 1: gate_cur = gate1; up_cur = up1; break;
    case 2: gate_cur = gate2; up_cur = up2; break;
    case 3: gate_cur = gate3; up_cur = up3; break;
    case 4: gate_cur = gate4; up_cur = up4; break;
    case 5: gate_cur = gate5; up_cur = up5; break;
    default: break;
    }

    const uint64_t selected_off = (uint64_t)token * args.n_expert_used + slot;
    glm_q4_K_pair_swiglu_simd_f32_impl<N_R0_Q4_K>(
        args, gate_cur, up_cur, x, weights, mid, scratch,
        tgpig, slot, token, selected_off, 0, tiisg, sgitg);
}

kernel void kernel_glm_q4_K_slots8_pair_swiglu_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *gate0,
        device const char *gate1,
        device const char *gate2,
        device const char *gate3,
        device const char *gate4,
        device const char *gate5,
        device const char *gate6,
        device const char *gate7,
        device const char *up0,
        device const char *up1,
        device const char *up2,
        device const char *up3,
        device const char *up4,
        device const char *up5,
        device const char *up6,
        device const char *up7,
        device const float *x,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint slot = tgpig.y;
    const uint token = tgpig.z;
    if (slot >= args.n_expert_used || token >= args.n_tokens) return;

    device const char *gate_cur = gate0;
    device const char *up_cur = up0;
    switch (slot) {
    case 1: gate_cur = gate1; up_cur = up1; break;
    case 2: gate_cur = gate2; up_cur = up2; break;
    case 3: gate_cur = gate3; up_cur = up3; break;
    case 4: gate_cur = gate4; up_cur = up4; break;
    case 5: gate_cur = gate5; up_cur = up5; break;
    case 6: gate_cur = gate6; up_cur = up6; break;
    case 7: gate_cur = gate7; up_cur = up7; break;
    default: break;
    }

    const uint64_t selected_off = (uint64_t)token * args.n_expert_used + slot;
    glm_q4_K_pair_swiglu_simd_f32_impl<N_R0_Q4_K>(
        args, gate_cur, up_cur, x, weights, mid, scratch,
        tgpig, slot, token, selected_off, 0, tiisg, sgitg);
}

kernel void kernel_glm_q4_K_slots6_pair_swiglu4_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *gate0,
        device const char *gate1,
        device const char *gate2,
        device const char *gate3,
        device const char *gate4,
        device const char *gate5,
        device const char *up0,
        device const char *up1,
        device const char *up2,
        device const char *up3,
        device const char *up4,
        device const char *up5,
        device const float *x,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint slot = tgpig.y;
    const uint token = tgpig.z;
    if (slot >= args.n_expert_used || token >= args.n_tokens) return;

    device const char *gate_cur = gate0;
    device const char *up_cur = up0;
    switch (slot) {
    case 1: gate_cur = gate1; up_cur = up1; break;
    case 2: gate_cur = gate2; up_cur = up2; break;
    case 3: gate_cur = gate3; up_cur = up3; break;
    case 4: gate_cur = gate4; up_cur = up4; break;
    case 5: gate_cur = gate5; up_cur = up5; break;
    default: break;
    }

    const uint64_t selected_off = (uint64_t)token * args.n_expert_used + slot;
    glm_q4_K_pair_swiglu_simd_f32_impl<N_R0_GLM_Q4_PAIR_K>(
        args, gate_cur, up_cur, x, weights, mid, scratch,
        tgpig, slot, token, selected_off, 0, tiisg, sgitg);
}

kernel void kernel_glm_q4_K_slots8_pair_swiglu4_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *gate0,
        device const char *gate1,
        device const char *gate2,
        device const char *gate3,
        device const char *gate4,
        device const char *gate5,
        device const char *gate6,
        device const char *gate7,
        device const char *up0,
        device const char *up1,
        device const char *up2,
        device const char *up3,
        device const char *up4,
        device const char *up5,
        device const char *up6,
        device const char *up7,
        device const float *x,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint slot = tgpig.y;
    const uint token = tgpig.z;
    if (slot >= args.n_expert_used || token >= args.n_tokens) return;

    device const char *gate_cur = gate0;
    device const char *up_cur = up0;
    switch (slot) {
    case 1: gate_cur = gate1; up_cur = up1; break;
    case 2: gate_cur = gate2; up_cur = up2; break;
    case 3: gate_cur = gate3; up_cur = up3; break;
    case 4: gate_cur = gate4; up_cur = up4; break;
    case 5: gate_cur = gate5; up_cur = up5; break;
    case 6: gate_cur = gate6; up_cur = up6; break;
    case 7: gate_cur = gate7; up_cur = up7; break;
    default: break;
    }

    const uint64_t selected_off = (uint64_t)token * args.n_expert_used + slot;
    glm_q4_K_pair_swiglu_simd_f32_impl<N_R0_GLM_Q4_PAIR_K>(
        args, gate_cur, up_cur, x, weights, mid, scratch,
        tgpig, slot, token, selected_off, 0, tiisg, sgitg);
}

kernel void kernel_glm_q4_K_pair_swiglu4_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *gate,
        device const char *up,
        device const float *x,
        device const int32_t *selected,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint slot = tgpig.y;
    const uint token = tgpig.z;
    if (slot >= args.n_expert_used || token >= args.n_tokens) return;
    const uint64_t selected_off = (uint64_t)token * args.n_expert_used + slot;
    const int expert = selected[selected_off];
    glm_q4_K_pair_swiglu_simd_f32_impl<N_R0_GLM_Q4_PAIR_K>(
        args, gate, up, x, weights, mid, scratch,
        tgpig, slot, token, selected_off, expert, tiisg, sgitg);
}

kernel void kernel_glm_q4_K_pair_swiglu2_mapped_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *gate,
        device const char *up,
        device const float *x,
        device const uint32_t *htpe,
        device const int32_t *hids,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint expert = tgpig.z;
    if (expert >= args.n_total_expert) return;
    const uint count = htpe[expert];
    const uint map_base = tgpig.y * 32u;
    for (uint i = 0; i < 32u; i++) {
        const uint map_row = map_base + i;
        if (map_row >= count) break;
        const int id = hids[(uint64_t)expert * args.n_tokens + map_row];
        if (id < 0) continue;
        const uint token = (uint)id / args.n_expert_used;
        const uint slot = (uint)id - token * args.n_expert_used;
        if (slot >= args.n_expert_used || token >= args.n_tokens) continue;
        const uint64_t selected_off = (uint64_t)token * args.n_expert_used + slot;
        glm_q4_K_pair_swiglu_simd_f32_impl<N_R0_Q4_K>(
            args, gate, up, x, weights, mid, scratch,
            tgpig, slot, token, selected_off, (int)expert, tiisg, sgitg);
    }
}

kernel void kernel_glm_q4_K_pair_swiglu2_mapped_row_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *gate,
        device const char *up,
        device const float *x,
        device const uint32_t *htpe,
        device const int32_t *hids,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint expert = tgpig.z;
    const uint map_row = tgpig.y;
    if (expert >= args.n_total_expert || map_row >= htpe[expert]) return;
    const int id = hids[(uint64_t)expert * args.n_tokens + map_row];
    if (id < 0) return;
    const uint token = (uint)id / args.n_expert_used;
    const uint slot = (uint)id - token * args.n_expert_used;
    if (slot >= args.n_expert_used || token >= args.n_tokens) return;
    const uint64_t selected_off = (uint64_t)token * args.n_expert_used + slot;
    glm_q4_K_pair_swiglu_simd_f32_impl<N_R0_Q4_K>(
        args, gate, up, x, weights, mid, scratch,
        tgpig, slot, token, selected_off, (int)expert, tiisg, sgitg);
}

static inline void glm_q5_K_pair_swiglu_f32_impl(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *gate,
        device const char *up,
        device const float *x,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch,
        uint3 tgpig,
        uint slot,
        uint token,
        uint64_t selected_off,
        int expert,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = 2;
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const uint row0 = ((uint)tgpig.x * (uint)NSG + (uint)sgitg) * N_R0_Q5_PAIR_K;
    if (row0 >= args.mid_dim || slot >= args.n_expert_used || token >= args.n_tokens) return;

    const uint64_t mid_base = (uint64_t)token * args.mid_token_stride +
                              (uint64_t)slot * args.mid_dim;
    if (expert < 0 || (uint)expert >= args.n_total_expert) {
        if (tiisg == 0u) {
            for (short row = 0; row < N_R0_Q5_PAIR_K && row0 + (uint)row < args.mid_dim; row++) {
                mid[mid_base + row0 + (uint)row] = 0.0f;
            }
        }
        return;
    }

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const uint bit0 = 2u * (uint)iq;
    const uint bit1 = bit0 + 1u;
    const uint bit2 = bit0 + 4u;
    const uint bit3 = bit0 + 5u;
    const int nb = args.in_dim / QK_K;

    const uint64_t expert_gate = (uint64_t)(uint)expert * args.gate_expert_bytes;
    const uint64_t expert_up = (uint64_t)(uint)expert * args.up_expert_bytes;
    device const block_q5_K *gate_rows =
        (device const block_q5_K *)(gate +
            expert_gate + (uint64_t)row0 * args.gate_row_bytes);
    device const block_q5_K *up_rows =
        (device const block_q5_K *)(up +
            expert_up + (uint64_t)row0 * args.up_row_bytes);
    device const float *y = x + (uint64_t)token * args.in_dim;
    device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    float sumg[N_R0_Q5_PAIR_K] = {0.f};
    float sumu[N_R0_Q5_PAIR_K] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        float yl[16];
        float yh[16];
        float4 sumy = {0.f, 0.f, 0.f, 0.f};

        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
            yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
            yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
        }

        device const uint16_t *scg = (device const uint16_t *)gate_rows[ib].scales + iq;
        device const uint16_t *qg1 = (device const uint16_t *)gate_rows[ib].qs + 16 * iq + 4 * ir;
        device const uint16_t *qhg = (device const uint16_t *)gate_rows[ib].qh + 4 * ir;
        device const half *dhg = &gate_rows[ib].d;
        device const uint16_t *scu = (device const uint16_t *)up_rows[ib].scales + iq;
        device const uint16_t *qu1 = (device const uint16_t *)up_rows[ib].qs + 16 * iq + 4 * ir;
        device const uint16_t *qhu = (device const uint16_t *)up_rows[ib].qh + 4 * ir;
        device const half *dhu = &up_rows[ib].d;

        for (short row = 0; row < N_R0_Q5_PAIR_K && row0 + (uint)row < args.mid_dim; row++) {
            sc16[0] = scg[0] & kmask1;
            sc16[1] = scg[2] & kmask1;
            sc16[2] = ((scg[4] >> 0) & kmask2) | ((scg[0] & kmask3) >> 2);
            sc16[3] = ((scg[4] >> 4) & kmask2) | ((scg[2] & kmask3) >> 2);

            device const uint16_t *qg2 = qg1 + 32;
            float4 accg = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                const uint ql1 = (uint)qg1[i];
                const uint ql2 = (uint)qg2[i];
                const uint hb = (uint)qhg[i];
                accg[0] += yl[2 * i + 0] *
                               (float)((ql1 & 0x000Fu) + (((hb >> bit0) & 1u) << 4u)) +
                           yl[2 * i + 1] *
                               (float)(((ql1 >> 8u) & 0x000Fu) + (((hb >> (bit0 + 8u)) & 1u) << 4u));
                accg[1] += yl[2 * i + 8] *
                               (float)(((ql1 >> 4u) & 0x000Fu) + (((hb >> bit1) & 1u) << 4u)) +
                           yl[2 * i + 9] *
                               (float)(((ql1 >> 12u) & 0x000Fu) + (((hb >> (bit1 + 8u)) & 1u) << 4u));
                accg[2] += yh[2 * i + 0] *
                               (float)((ql2 & 0x000Fu) + (((hb >> bit2) & 1u) << 4u)) +
                           yh[2 * i + 1] *
                               (float)(((ql2 >> 8u) & 0x000Fu) + (((hb >> (bit2 + 8u)) & 1u) << 4u));
                accg[3] += yh[2 * i + 8] *
                               (float)(((ql2 >> 4u) & 0x000Fu) + (((hb >> bit3) & 1u) << 4u)) +
                           yh[2 * i + 9] *
                               (float)(((ql2 >> 12u) & 0x000Fu) + (((hb >> (bit3 + 8u)) & 1u) << 4u));
            }

            sumg[row] += dhg[0] * (accg[0] * sc8[0] + accg[1] * sc8[1] +
                                   accg[2] * sc8[4] + accg[3] * sc8[5]) -
                         dhg[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                   sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            sc16[0] = scu[0] & kmask1;
            sc16[1] = scu[2] & kmask1;
            sc16[2] = ((scu[4] >> 0) & kmask2) | ((scu[0] & kmask3) >> 2);
            sc16[3] = ((scu[4] >> 4) & kmask2) | ((scu[2] & kmask3) >> 2);

            device const uint16_t *qu2 = qu1 + 32;
            float4 accu = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                const uint ql1 = (uint)qu1[i];
                const uint ql2 = (uint)qu2[i];
                const uint hb = (uint)qhu[i];
                accu[0] += yl[2 * i + 0] *
                               (float)((ql1 & 0x000Fu) + (((hb >> bit0) & 1u) << 4u)) +
                           yl[2 * i + 1] *
                               (float)(((ql1 >> 8u) & 0x000Fu) + (((hb >> (bit0 + 8u)) & 1u) << 4u));
                accu[1] += yl[2 * i + 8] *
                               (float)(((ql1 >> 4u) & 0x000Fu) + (((hb >> bit1) & 1u) << 4u)) +
                           yl[2 * i + 9] *
                               (float)(((ql1 >> 12u) & 0x000Fu) + (((hb >> (bit1 + 8u)) & 1u) << 4u));
                accu[2] += yh[2 * i + 0] *
                               (float)((ql2 & 0x000Fu) + (((hb >> bit2) & 1u) << 4u)) +
                           yh[2 * i + 1] *
                               (float)(((ql2 >> 8u) & 0x000Fu) + (((hb >> (bit2 + 8u)) & 1u) << 4u));
                accu[3] += yh[2 * i + 8] *
                               (float)(((ql2 >> 4u) & 0x000Fu) + (((hb >> bit3) & 1u) << 4u)) +
                           yh[2 * i + 9] *
                               (float)(((ql2 >> 12u) & 0x000Fu) + (((hb >> (bit3 + 8u)) & 1u) << 4u));
            }

            sumu[row] += dhu[0] * (accu[0] * sc8[0] + accu[1] * sc8[1] +
                                   accu[2] * sc8[4] + accu[3] * sc8[5]) -
                         dhu[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                   sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            qg1 += args.gate_row_bytes / 2;
            qhg += args.gate_row_bytes / 2;
            scg += args.gate_row_bytes / 2;
            dhg += args.gate_row_bytes / 2;
            qu1 += args.up_row_bytes / 2;
            qhu += args.up_row_bytes / 2;
            scu += args.up_row_bytes / 2;
            dhu += args.up_row_bytes / 2;
        }

        y4 += 4 * QK_K;
    }

    for (short row = 0; row < N_R0_Q5_PAIR_K && row0 + (uint)row < args.mid_dim; row++) {
        const float g = simd_sum(sumg[row]);
        const float u = simd_sum(sumu[row]);
        if (tiisg == 0u) {
            const float sw = g / (1.0f + exp(-g));
            mid[mid_base + row0 + (uint)row] = sw * u * weights[selected_off];
        }
    }

    (void)scratch;
}

kernel void kernel_glm_q5_K_pair_swiglu_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *gate,
        device const char *up,
        device const float *x,
        device const int32_t *selected,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint slot = tgpig.y;
    const uint token = tgpig.z;
    if (slot >= args.n_expert_used || token >= args.n_tokens) return;
    const uint64_t selected_off = (uint64_t)token * args.n_expert_used + slot;
    const int expert = selected[selected_off];
    glm_q5_K_pair_swiglu_f32_impl(
        args, gate, up, x, weights, mid, scratch,
        tgpig, slot, token, selected_off, expert, tiisg, sgitg);
}

kernel void kernel_glm_q5_K_slots6_pair_swiglu_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *gate0,
        device const char *gate1,
        device const char *gate2,
        device const char *gate3,
        device const char *gate4,
        device const char *gate5,
        device const char *up0,
        device const char *up1,
        device const char *up2,
        device const char *up3,
        device const char *up4,
        device const char *up5,
        device const float *x,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint slot = tgpig.y;
    const uint token = tgpig.z;
    if (slot >= args.n_expert_used || token >= args.n_tokens) return;

    device const char *gate_cur = gate0;
    device const char *up_cur = up0;
    switch (slot) {
    case 1: gate_cur = gate1; up_cur = up1; break;
    case 2: gate_cur = gate2; up_cur = up2; break;
    case 3: gate_cur = gate3; up_cur = up3; break;
    case 4: gate_cur = gate4; up_cur = up4; break;
    case 5: gate_cur = gate5; up_cur = up5; break;
    default: break;
    }

    const uint64_t selected_off = (uint64_t)token * args.n_expert_used + slot;
    glm_q5_K_pair_swiglu_f32_impl(
        args, gate_cur, up_cur, x, weights, mid, scratch,
        tgpig, slot, token, selected_off, 0, tiisg, sgitg);
}

kernel void kernel_glm_q5_K_slots8_pair_swiglu_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *gate0,
        device const char *gate1,
        device const char *gate2,
        device const char *gate3,
        device const char *gate4,
        device const char *gate5,
        device const char *gate6,
        device const char *gate7,
        device const char *up0,
        device const char *up1,
        device const char *up2,
        device const char *up3,
        device const char *up4,
        device const char *up5,
        device const char *up6,
        device const char *up7,
        device const float *x,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint slot = tgpig.y;
    const uint token = tgpig.z;
    if (slot >= args.n_expert_used || token >= args.n_tokens) return;

    device const char *gate_cur = gate0;
    device const char *up_cur = up0;
    switch (slot) {
    case 1: gate_cur = gate1; up_cur = up1; break;
    case 2: gate_cur = gate2; up_cur = up2; break;
    case 3: gate_cur = gate3; up_cur = up3; break;
    case 4: gate_cur = gate4; up_cur = up4; break;
    case 5: gate_cur = gate5; up_cur = up5; break;
    case 6: gate_cur = gate6; up_cur = up6; break;
    case 7: gate_cur = gate7; up_cur = up7; break;
    default: break;
    }

    const uint64_t selected_off = (uint64_t)token * args.n_expert_used + slot;
    glm_q5_K_pair_swiglu_f32_impl(
        args, gate_cur, up_cur, x, weights, mid, scratch,
        tgpig, slot, token, selected_off, 0, tiisg, sgitg);
}

kernel void kernel_glm_q5_K_pair_swiglu_mapped_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *gate,
        device const char *up,
        device const float *x,
        device const uint32_t *htpe,
        device const int32_t *hids,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint expert = tgpig.z;
    if (expert >= args.n_total_expert) return;
    const uint count = htpe[expert];
    const uint map_base = tgpig.y * 32u;
    for (uint i = 0; i < 32u; i++) {
        const uint map_row = map_base + i;
        if (map_row >= count) break;
        const int id = hids[(uint64_t)expert * args.n_tokens + map_row];
        if (id < 0) continue;
        const uint token = (uint)id / args.n_expert_used;
        const uint slot = (uint)id - token * args.n_expert_used;
        if (slot >= args.n_expert_used || token >= args.n_tokens) continue;
        const uint64_t selected_off = (uint64_t)token * args.n_expert_used + slot;
        glm_q5_K_pair_swiglu_f32_impl(
            args, gate, up, x, weights, mid, scratch,
            tgpig, slot, token, selected_off, (int)expert, tiisg, sgitg);
    }
}

kernel void kernel_glm_q5_K_pair_swiglu_mapped_row_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *gate,
        device const char *up,
        device const float *x,
        device const uint32_t *htpe,
        device const int32_t *hids,
        device const float *weights,
        device float *mid,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint expert = tgpig.z;
    const uint map_row = tgpig.y;
    if (expert >= args.n_total_expert || map_row >= htpe[expert]) return;
    const int id = hids[(uint64_t)expert * args.n_tokens + map_row];
    if (id < 0) return;
    const uint token = (uint)id / args.n_expert_used;
    const uint slot = (uint)id - token * args.n_expert_used;
    if (slot >= args.n_expert_used || token >= args.n_tokens) return;
    const uint64_t selected_off = (uint64_t)token * args.n_expert_used + slot;
    glm_q5_K_pair_swiglu_f32_impl(
        args, gate, up, x, weights, mid, scratch,
        tgpig, slot, token, selected_off, (int)expert, tiisg, sgitg);
}

kernel void kernel_glm_q5_K_down_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *down,
        device const int32_t *selected,
        device const float *mid,
        device float *out,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const short NSG = 2;
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const uint row0 = ((uint)tgpig.x * (uint)NSG + (uint)sgitg) * N_R0_Q5_K;
    const uint token = tgpig.y;
    if (row0 >= args.out_dim || token >= args.n_tokens) return;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const uint bit0 = 2u * (uint)iq;
    const uint bit1 = bit0 + 1u;
    const uint bit2 = bit0 + 4u;
    const uint bit3 = bit0 + 5u;
    const int nb = args.mid_dim / QK_K;

    float sumf[N_R0_Q5_K] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;
    const uint64_t selected_base = (uint64_t)token * args.n_expert_used;
    const uint64_t mid_base = (uint64_t)token * args.mid_token_stride;
    for (uint slot = 0; slot < args.n_expert_used; slot++) {
        const int expert = selected[selected_base + slot];
        if (expert < 0 || (uint)expert >= args.n_total_expert) continue;
        device const block_q5_K *x =
            (device const block_q5_K *)(down +
                (uint64_t)(uint)expert * args.down_expert_bytes +
                (uint64_t)row0 * args.down_row_bytes);
        device const float *y = mid + mid_base + (uint64_t)slot * args.mid_dim;
        device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[16];
            float yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};

            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
                yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
                yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
            }

            device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
            device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const uint16_t *qh = (device const uint16_t *)x[ib].qh + 4 * ir;
            device const half *dh = &x[ib].d;

            for (short row = 0; row < N_R0_Q5_K && row0 + (uint)row < args.out_dim; row++) {
                sc16[0] = sc[0] & kmask1;
                sc16[1] = sc[2] & kmask1;
                sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                device const uint16_t *q2 = q1 + 32;
                float4 acc = {0.f, 0.f, 0.f, 0.f};

                FOR_UNROLL (short i = 0; i < 4; ++i) {
                    const uint ql1 = (uint)q1[i];
                    const uint ql2 = (uint)q2[i];
                    const uint hb = (uint)qh[i];
                    acc[0] += yl[2 * i + 0] *
                                  (float)((ql1 & 0x000Fu) + (((hb >> bit0) & 1u) << 4u)) +
                              yl[2 * i + 1] *
                                  (float)(((ql1 >> 8u) & 0x000Fu) + (((hb >> (bit0 + 8u)) & 1u) << 4u));
                    acc[1] += yl[2 * i + 8] *
                                  (float)(((ql1 >> 4u) & 0x000Fu) + (((hb >> bit1) & 1u) << 4u)) +
                              yl[2 * i + 9] *
                                  (float)(((ql1 >> 12u) & 0x000Fu) + (((hb >> (bit1 + 8u)) & 1u) << 4u));
                    acc[2] += yh[2 * i + 0] *
                                  (float)((ql2 & 0x000Fu) + (((hb >> bit2) & 1u) << 4u)) +
                              yh[2 * i + 1] *
                                  (float)(((ql2 >> 8u) & 0x000Fu) + (((hb >> (bit2 + 8u)) & 1u) << 4u));
                    acc[3] += yh[2 * i + 8] *
                                  (float)(((ql2 >> 4u) & 0x000Fu) + (((hb >> bit3) & 1u) << 4u)) +
                              yh[2 * i + 9] *
                                  (float)(((ql2 >> 12u) & 0x000Fu) + (((hb >> (bit3 + 8u)) & 1u) << 4u));
                }

                sumf[row] += dh[0] * (acc[0] * sc8[0] + acc[1] * sc8[1] +
                                      acc[2] * sc8[4] + acc[3] * sc8[5]) -
                             dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                      sumy[2] * sc8[6] + sumy[3] * sc8[7]);

                q1 += args.down_row_bytes / 2;
                qh += args.down_row_bytes / 2;
                sc += args.down_row_bytes / 2;
                dh += args.down_row_bytes / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    for (short row = 0; row < N_R0_Q5_K && row0 + (uint)row < args.out_dim; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0u) {
            out[(uint64_t)token * args.out_dim + row0 + (uint)row] = sum_all;
        }
    }
}

kernel void kernel_glm_q2_K_down_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *down,
        device const int32_t *selected,
        device const float *mid,
        device float *out,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const short NSG = 2;
    const uint row0 = ((uint)tgpig.x * (uint)NSG + (uint)sgitg) * N_R0_Q2_K;
    const uint token = tgpig.y;
    if (row0 >= args.out_dim || token >= args.n_tokens) return;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const short is = (8 * ir) / 16;
    const int nb = args.mid_dim / QK_K;
    float sumf[N_R0_Q2_K] = {0.f};
    const uint64_t selected_base = (uint64_t)token * args.n_expert_used;
    const uint64_t mid_base = (uint64_t)token * args.mid_token_stride;
    for (uint slot = 0; slot < args.n_expert_used; slot++) {
        const int expert = selected[selected_base + slot];
        if (expert < 0 || (uint)expert >= args.n_total_expert) continue;
        device const block_q2_K *x =
            (device const block_q2_K *)(down +
                (uint64_t)(uint)expert * args.down_expert_bytes +
                (uint64_t)row0 * args.down_row_bytes);
        device const float *y = mid + mid_base + (uint64_t)slot * args.mid_dim;
        device const float *y4 = y + ix * QK_K + 128 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[32];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            for (short i = 0; i < 8; ++i) {
                yl[i +  0] = y4[i +  0]; sumy[0] += yl[i +  0];
                yl[i +  8] = y4[i + 32]; sumy[1] += yl[i +  8];
                yl[i + 16] = y4[i + 64]; sumy[2] += yl[i + 16];
                yl[i + 24] = y4[i + 96]; sumy[3] += yl[i + 24];
            }

            device const uint8_t *sc = (device const uint8_t *)x[ib].scales + 8 * iq + is;
            device const uint16_t *qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half *dh = &x[ib].d;
            for (short row = 0; row < N_R0_Q2_K && row0 + (uint)row < args.out_dim; row++) {
                float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                for (int i = 0; i < 8; i += 2) {
                    acc1[0] += yl[i +  0] * (qs[i / 2] & 0x0003);
                    acc2[0] += yl[i +  1] * (qs[i / 2] & 0x0300);
                    acc1[1] += yl[i +  8] * (qs[i / 2] & 0x000c);
                    acc2[1] += yl[i +  9] * (qs[i / 2] & 0x0c00);
                    acc1[2] += yl[i + 16] * (qs[i / 2] & 0x0030);
                    acc2[2] += yl[i + 17] * (qs[i / 2] & 0x3000);
                    acc1[3] += yl[i + 24] * (qs[i / 2] & 0x00c0);
                    acc2[3] += yl[i + 25] * (qs[i / 2] & 0xc000);
                }

                const float d = dh[0];
                const float m = dh[1] * 1.f / 16.f;
                sumf[row] += d * ((acc1[0] + 1.f / 256.f * acc2[0]) * (sc[0] & 0xF) * 1.f /  1.f +
                                  (acc1[1] + 1.f / 256.f * acc2[1]) * (sc[2] & 0xF) * 1.f /  4.f +
                                  (acc1[2] + 1.f / 256.f * acc2[2]) * (sc[4] & 0xF) * 1.f / 16.f +
                                  (acc1[3] + 1.f / 256.f * acc2[3]) * (sc[6] & 0xF) * 1.f / 64.f) -
                             m * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                                  sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));

                qs += args.down_row_bytes / 2;
                sc += args.down_row_bytes;
                dh += args.down_row_bytes / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    for (short row = 0; row < N_R0_Q2_K && row0 + (uint)row < args.out_dim; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0u) {
            out[(uint64_t)token * args.out_dim + row0 + (uint)row] = sum_all;
        }
    }

    (void)scratch;
}

kernel void kernel_glm_q2_K_addr_down_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const uint64_t *down_addrs,
        device const int32_t *selected,
        device const float *mid,
        device float *out,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const short NSG = 2;
    const uint row0 = ((uint)tgpig.x * (uint)NSG + (uint)sgitg) * N_R0_Q2_K;
    const uint token = tgpig.y;
    if (row0 >= args.out_dim || token >= args.n_tokens) return;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const short is = (8 * ir) / 16;
    const int nb = args.mid_dim / QK_K;
    float sumf[N_R0_Q2_K] = {0.f};
    const uint64_t selected_base = (uint64_t)token * args.n_expert_used;
    const uint64_t mid_base = (uint64_t)token * args.mid_token_stride;
    for (uint slot = 0; slot < args.n_expert_used; slot++) {
        const int expert = selected[selected_base + slot];
        if (expert < 0 || (uint)expert >= args.n_total_expert) continue;
        const uint64_t down_addr = down_addrs[(uint)expert];
        if (down_addr == 0) continue;
        device const block_q2_K *x =
            (device const block_q2_K *)(reinterpret_cast<device const char *>(down_addr) +
                (uint64_t)row0 * args.down_row_bytes);
        device const float *y = mid + mid_base + (uint64_t)slot * args.mid_dim;
        device const float *y4 = y + ix * QK_K + 128 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[32];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            for (short i = 0; i < 8; ++i) {
                yl[i +  0] = y4[i +  0]; sumy[0] += yl[i +  0];
                yl[i +  8] = y4[i + 32]; sumy[1] += yl[i +  8];
                yl[i + 16] = y4[i + 64]; sumy[2] += yl[i + 16];
                yl[i + 24] = y4[i + 96]; sumy[3] += yl[i + 24];
            }

            device const uint8_t *sc = (device const uint8_t *)x[ib].scales + 8 * iq + is;
            device const uint16_t *qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half *dh = &x[ib].d;
            for (short row = 0; row < N_R0_Q2_K && row0 + (uint)row < args.out_dim; row++) {
                float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                for (int i = 0; i < 8; i += 2) {
                    acc1[0] += yl[i +  0] * (qs[i / 2] & 0x0003);
                    acc2[0] += yl[i +  1] * (qs[i / 2] & 0x0300);
                    acc1[1] += yl[i +  8] * (qs[i / 2] & 0x000c);
                    acc2[1] += yl[i +  9] * (qs[i / 2] & 0x0c00);
                    acc1[2] += yl[i + 16] * (qs[i / 2] & 0x0030);
                    acc2[2] += yl[i + 17] * (qs[i / 2] & 0x3000);
                    acc1[3] += yl[i + 24] * (qs[i / 2] & 0x00c0);
                    acc2[3] += yl[i + 25] * (qs[i / 2] & 0xc000);
                }

                const float d = dh[0];
                const float m = dh[1] * 1.f / 16.f;
                sumf[row] += d * ((acc1[0] + 1.f / 256.f * acc2[0]) * (sc[0] & 0xF) * 1.f /  1.f +
                                  (acc1[1] + 1.f / 256.f * acc2[1]) * (sc[2] & 0xF) * 1.f /  4.f +
                                  (acc1[2] + 1.f / 256.f * acc2[2]) * (sc[4] & 0xF) * 1.f / 16.f +
                                  (acc1[3] + 1.f / 256.f * acc2[3]) * (sc[6] & 0xF) * 1.f / 64.f) -
                             m * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                                  sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));

                qs += args.down_row_bytes / 2;
                sc += args.down_row_bytes;
                dh += args.down_row_bytes / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    for (short row = 0; row < N_R0_Q2_K && row0 + (uint)row < args.out_dim; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0u) {
            out[(uint64_t)token * args.out_dim + row0 + (uint)row] = sum_all;
        }
    }

    (void)scratch;
}

kernel void kernel_glm_q4_K_down_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *down,
        device const int32_t *selected,
        device const float *mid,
        device float *out,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    const uint ntg = 256u;
    const uint row = tgpig.x;
    const uint token = tgpig.y;
    if (row >= args.out_dim || token >= args.n_tokens) return;

    float acc = 0.0f;
    const uint64_t selected_base = (uint64_t)token * args.n_expert_used;
    const uint64_t mid_base = (uint64_t)token * args.mid_token_stride;
    for (uint slot = 0; slot < args.n_expert_used; slot++) {
        const int expert = selected[selected_base + slot];
        if (expert < 0 || (uint)expert >= args.n_total_expert) continue;
        device const block_q4_K *down_row =
            (device const block_q4_K *)(down +
                (uint64_t)(uint)expert * args.down_expert_bytes +
                (uint64_t)row * args.down_row_bytes);
        device const float *slot_mid = mid + mid_base + (uint64_t)slot * args.mid_dim;
        for (uint k = tid; k < args.mid_dim; k += ntg) {
            acc += ds4_glm_q4_K_value(down_row, k) * slot_mid[k];
        }
    }

    scratch[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = ntg >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        out[(uint64_t)token * args.out_dim + row] = scratch[0];
    }
}

kernel void kernel_glm_q4_K_addr_down_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const uint64_t *down_addrs,
        device const int32_t *selected,
        device const float *mid,
        device float *out,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    const uint ntg = 256u;
    const uint row = tgpig.x;
    const uint token = tgpig.y;
    if (row >= args.out_dim || token >= args.n_tokens) return;

    float acc = 0.0f;
    const uint64_t selected_base = (uint64_t)token * args.n_expert_used;
    const uint64_t mid_base = (uint64_t)token * args.mid_token_stride;
    for (uint slot = 0; slot < args.n_expert_used; slot++) {
        const int expert = selected[selected_base + slot];
        if (expert < 0 || (uint)expert >= args.n_total_expert) continue;
        const uint64_t down_addr = down_addrs[(uint)expert];
        if (down_addr == 0) continue;
        device const block_q4_K *down_row =
            (device const block_q4_K *)(reinterpret_cast<device const char *>(down_addr) +
                (uint64_t)row * args.down_row_bytes);
        device const float *slot_mid = mid + mid_base + (uint64_t)slot * args.mid_dim;
        for (uint k = tid; k < args.mid_dim; k += ntg) {
            acc += ds4_glm_q4_K_value(down_row, k) * slot_mid[k];
        }
    }

    scratch[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = ntg >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        out[(uint64_t)token * args.out_dim + row] = scratch[0];
    }
}

kernel void kernel_glm_q4_K_down_simd_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *down,
        device const int32_t *selected,
        device const float *mid,
        device float *out,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const short NSG = 2;
    const short nr0 = N_R0_Q4_K;
    const int nb = args.mid_dim / QK_K;
    const uint row0 = ((uint)tgpig.x * (uint)NSG + (uint)sgitg) * (uint)nr0;
    const uint token = tgpig.y;
    if (row0 >= args.out_dim || token >= args.n_tokens) return;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    float sumf[N_R0_Q4_K] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    const uint64_t selected_base = (uint64_t)token * args.n_expert_used;
    const uint64_t mid_base = (uint64_t)token * args.mid_token_stride;
    for (uint slot = 0; slot < args.n_expert_used; slot++) {
        const int expert = selected[selected_base + slot];
        if (expert < 0 || (uint)expert >= args.n_total_expert) continue;

        device const block_q4_K *x =
            (device const block_q4_K *)(down +
                (uint64_t)(uint)expert * args.down_expert_bytes +
                (uint64_t)row0 * args.down_row_bytes);
        device const float *y = mid + mid_base + (uint64_t)slot * args.mid_dim;
        device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[16];
            float yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};

            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
                yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
                yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
            }

            device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
            device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half *dh = &x[ib].d;

            for (short row = 0; row < nr0 && row0 + (uint)row < args.out_dim; row++) {
                sc16[0] = sc[0] & kmask1;
                sc16[1] = sc[2] & kmask1;
                sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                device const uint16_t *q2 = q1 + 32;
                float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                FOR_UNROLL (short i = 0; i < 4; ++i) {
                    acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                    acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                    acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                    acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                    acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                    acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                    acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                    acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
                }

                sumf[row] += dh[0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                      (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                      (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                      (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                             dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                      sumy[2] * sc8[6] + sumy[3] * sc8[7]);

                q1 += args.down_row_bytes / 2;
                sc += args.down_row_bytes / 2;
                dh += args.down_row_bytes / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    for (short row = 0; row < nr0 && row0 + (uint)row < args.out_dim; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0u) {
            out[(uint64_t)token * args.out_dim + row0 + (uint)row] = sum_all;
        }
    }
}

kernel void kernel_glm_q4_K_addr_down_simd_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const uint64_t *down_addrs,
        device const int32_t *selected,
        device const float *mid,
        device float *out,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const short NSG = 2;
    const short nr0 = N_R0_Q4_K;
    const int nb = args.mid_dim / QK_K;
    const uint row0 = ((uint)tgpig.x * (uint)NSG + (uint)sgitg) * (uint)nr0;
    const uint token = tgpig.y;
    if (row0 >= args.out_dim || token >= args.n_tokens) return;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    float sumf[N_R0_Q4_K] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    const uint64_t selected_base = (uint64_t)token * args.n_expert_used;
    const uint64_t mid_base = (uint64_t)token * args.mid_token_stride;
    for (uint slot = 0; slot < args.n_expert_used; slot++) {
        const int expert = selected[selected_base + slot];
        if (expert < 0 || (uint)expert >= args.n_total_expert) continue;
        const uint64_t down_addr = down_addrs[(uint)expert];
        if (down_addr == 0) continue;

        device const block_q4_K *x =
            (device const block_q4_K *)(reinterpret_cast<device const char *>(down_addr) +
                (uint64_t)row0 * args.down_row_bytes);
        device const float *y = mid + mid_base + (uint64_t)slot * args.mid_dim;
        device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[16];
            float yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};

            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
                yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
                yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
            }

            device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
            device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half *dh = &x[ib].d;

            for (short row = 0; row < nr0 && row0 + (uint)row < args.out_dim; row++) {
                sc16[0] = sc[0] & kmask1;
                sc16[1] = sc[2] & kmask1;
                sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                device const uint16_t *q2 = q1 + 32;
                float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                FOR_UNROLL (short i = 0; i < 4; ++i) {
                    acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                    acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                    acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                    acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                    acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                    acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                    acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                    acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
                }

                sumf[row] += dh[0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                      (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                      (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                      (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                             dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                      sumy[2] * sc8[6] + sumy[3] * sc8[7]);

                q1 += args.down_row_bytes / 2;
                sc += args.down_row_bytes / 2;
                dh += args.down_row_bytes / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    for (short row = 0; row < nr0 && row0 + (uint)row < args.out_dim; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0u) {
            out[(uint64_t)token * args.out_dim + row0 + (uint)row] = sum_all;
        }
    }
}

kernel void kernel_glm_q6_K_down_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *down,
        device const int32_t *selected,
        device const float *mid,
        device float *out,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const short NSG = 2;
    constexpr uint kmask1 = 0x03u;
    constexpr uint kmask2 = 0x0Cu;
    constexpr uint kmask3 = 0x30u;
    constexpr uint kmask4 = 0xC0u;

    const uint row0 = ((uint)tgpig.x * (uint)NSG + (uint)sgitg) * N_R0_Q6_K;
    const uint token = tgpig.y;
    if (row0 >= args.out_dim || token >= args.n_tokens) return;

    const int nb = args.mid_dim / QK_K;
    float sumf[N_R0_Q6_K] = {0.f};
    float yl[16];
    const short tid = tiisg / 2;
    const short ix = tiisg % 2;
    const short ip = tid / 8;
    const short il = tid % 8;
    const short l0 = 4 * il;
    const short is = 8 * ip + l0 / 16;
    const short y_offset = 128 * ip + l0;
    const short q_offset_l = 64 * ip + l0;
    const short q_offset_h = 32 * ip + l0;

    const uint64_t selected_base = (uint64_t)token * args.n_expert_used;
    const uint64_t mid_base = (uint64_t)token * args.mid_token_stride;
    for (uint slot = 0; slot < args.n_expert_used; slot++) {
        const int expert = selected[selected_base + slot];
        if (expert < 0 || (uint)expert >= args.n_total_expert) continue;
        device const block_q6_K *x =
            (device const block_q6_K *)(down +
                (uint64_t)(uint)expert * args.down_expert_bytes +
                (uint64_t)row0 * args.down_row_bytes);
        device const float *yy = mid + mid_base + (uint64_t)slot * args.mid_dim;

        for (int ib = ix; ib < nb; ib += 2) {
            device const uchar *q1 = x[ib].ql + q_offset_l;
            device const uchar *q2 = q1 + 32;
            device const uchar *qh = x[ib].qh + q_offset_h;
            device const char *sc = x[ib].scales + is;
            device const half *dh = &x[ib].d;
            device const float *y = yy + (uint64_t)ib * QK_K + y_offset;

            for (short l = 0; l < 4; ++l) {
                yl[4 * l + 0] = y[l + 0];
                yl[4 * l + 1] = y[l + 32];
                yl[4 * l + 2] = y[l + 64];
                yl[4 * l + 3] = y[l + 96];
            }

            for (short row = 0; row < N_R0_Q6_K && row0 + (uint)row < args.out_dim; row++) {
                float4 sums = {0.f, 0.f, 0.f, 0.f};

                FOR_UNROLL (short l = 0; l < 4; ++l) {
                    const uint h = (uint)qh[l];
                    sums[0] += yl[4 * l + 0] *
                        (float)((int)((q1[l] & 0x0Fu) | ((h & kmask1) << 4u)) - 32);
                    sums[1] += yl[4 * l + 1] *
                        (float)((int)((q2[l] & 0x0Fu) | ((h & kmask2) << 2u)) - 32);
                    sums[2] += yl[4 * l + 2] *
                        (float)((int)((q1[l] >> 4u) | (h & kmask3)) - 32);
                    sums[3] += yl[4 * l + 3] *
                        (float)((int)((q2[l] >> 4u) | ((h & kmask4) >> 2u)) - 32);
                }

                sumf[row] += dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] +
                                      sums[2] * sc[4] + sums[3] * sc[6]);

                q1 += args.down_row_bytes;
                q2 += args.down_row_bytes;
                qh += args.down_row_bytes;
                sc += args.down_row_bytes;
                dh += args.down_row_bytes / 2;
            }
        }
    }

    for (short row = 0; row < N_R0_Q6_K && row0 + (uint)row < args.out_dim; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0u) {
            out[(uint64_t)token * args.out_dim + row0 + (uint)row] = sum_all;
        }
    }
}

kernel void kernel_glm_q5_K_slots6_down_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *down0,
        device const char *down1,
        device const char *down2,
        device const char *down3,
        device const char *down4,
        device const char *down5,
        device const float *mid,
        device float *out,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const short NSG = 2;
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const uint row0 = ((uint)tgpig.x * (uint)NSG + (uint)sgitg) * N_R0_Q5_K;
    const uint token = tgpig.y;
    if (row0 >= args.out_dim || token >= args.n_tokens) return;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const uint bit0 = 2u * (uint)iq;
    const uint bit1 = bit0 + 1u;
    const uint bit2 = bit0 + 4u;
    const uint bit3 = bit0 + 5u;
    const int nb = args.mid_dim / QK_K;

    float sumf[N_R0_Q5_K] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;
    const uint64_t mid_base = (uint64_t)token * args.mid_token_stride;
    for (uint slot = 0; slot < args.n_expert_used; slot++) {
        device const char *down_cur = down0;
        switch (slot) {
        case 1: down_cur = down1; break;
        case 2: down_cur = down2; break;
        case 3: down_cur = down3; break;
        case 4: down_cur = down4; break;
        case 5: down_cur = down5; break;
        default: break;
        }

        device const block_q5_K *x =
            (device const block_q5_K *)(down_cur +
                (uint64_t)row0 * args.down_row_bytes);
        device const float *y = mid + mid_base + (uint64_t)slot * args.mid_dim;
        device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[16];
            float yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};

            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
                yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
                yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
            }

            device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
            device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const uint16_t *qh = (device const uint16_t *)x[ib].qh + 4 * ir;
            device const half *dh = &x[ib].d;

            for (short row = 0; row < N_R0_Q5_K && row0 + (uint)row < args.out_dim; row++) {
                sc16[0] = sc[0] & kmask1;
                sc16[1] = sc[2] & kmask1;
                sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                device const uint16_t *q2 = q1 + 32;
                float4 acc = {0.f, 0.f, 0.f, 0.f};

                FOR_UNROLL (short i = 0; i < 4; ++i) {
                    const uint ql1 = (uint)q1[i];
                    const uint ql2 = (uint)q2[i];
                    const uint hb = (uint)qh[i];
                    acc[0] += yl[2 * i + 0] *
                                  (float)((ql1 & 0x000Fu) + (((hb >> bit0) & 1u) << 4u)) +
                              yl[2 * i + 1] *
                                  (float)(((ql1 >> 8u) & 0x000Fu) + (((hb >> (bit0 + 8u)) & 1u) << 4u));
                    acc[1] += yl[2 * i + 8] *
                                  (float)(((ql1 >> 4u) & 0x000Fu) + (((hb >> bit1) & 1u) << 4u)) +
                              yl[2 * i + 9] *
                                  (float)(((ql1 >> 12u) & 0x000Fu) + (((hb >> (bit1 + 8u)) & 1u) << 4u));
                    acc[2] += yh[2 * i + 0] *
                                  (float)((ql2 & 0x000Fu) + (((hb >> bit2) & 1u) << 4u)) +
                              yh[2 * i + 1] *
                                  (float)(((ql2 >> 8u) & 0x000Fu) + (((hb >> (bit2 + 8u)) & 1u) << 4u));
                    acc[3] += yh[2 * i + 8] *
                                  (float)(((ql2 >> 4u) & 0x000Fu) + (((hb >> bit3) & 1u) << 4u)) +
                              yh[2 * i + 9] *
                                  (float)(((ql2 >> 12u) & 0x000Fu) + (((hb >> (bit3 + 8u)) & 1u) << 4u));
                }

                sumf[row] += dh[0] * (acc[0] * sc8[0] + acc[1] * sc8[1] +
                                      acc[2] * sc8[4] + acc[3] * sc8[5]) -
                             dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                      sumy[2] * sc8[6] + sumy[3] * sc8[7]);

                q1 += args.down_row_bytes / 2;
                qh += args.down_row_bytes / 2;
                sc += args.down_row_bytes / 2;
                dh += args.down_row_bytes / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    for (short row = 0; row < N_R0_Q5_K && row0 + (uint)row < args.out_dim; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0u) {
            out[(uint64_t)token * args.out_dim + row0 + (uint)row] = sum_all;
        }
    }
}

kernel void kernel_glm_q5_K_slots8_down_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *down0,
        device const char *down1,
        device const char *down2,
        device const char *down3,
        device const char *down4,
        device const char *down5,
        device const char *down6,
        device const char *down7,
        device const float *mid,
        device float *out,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const short NSG = 2;
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const uint row0 = ((uint)tgpig.x * (uint)NSG + (uint)sgitg) * N_R0_Q5_K;
    const uint token = tgpig.y;
    if (row0 >= args.out_dim || token >= args.n_tokens) return;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const uint bit0 = 2u * (uint)iq;
    const uint bit1 = bit0 + 1u;
    const uint bit2 = bit0 + 4u;
    const uint bit3 = bit0 + 5u;
    const int nb = args.mid_dim / QK_K;

    float sumf[N_R0_Q5_K] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;
    const uint64_t mid_base = (uint64_t)token * args.mid_token_stride;
    for (uint slot = 0; slot < args.n_expert_used; slot++) {
        device const char *down_cur = down0;
        switch (slot) {
        case 1: down_cur = down1; break;
        case 2: down_cur = down2; break;
        case 3: down_cur = down3; break;
        case 4: down_cur = down4; break;
        case 5: down_cur = down5; break;
        case 6: down_cur = down6; break;
        case 7: down_cur = down7; break;
        default: break;
        }

        device const block_q5_K *x =
            (device const block_q5_K *)(down_cur +
                (uint64_t)row0 * args.down_row_bytes);
        device const float *y = mid + mid_base + (uint64_t)slot * args.mid_dim;
        device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[16];
            float yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};

            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
                yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
                yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
            }

            device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
            device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const uint16_t *qh = (device const uint16_t *)x[ib].qh + 4 * ir;
            device const half *dh = &x[ib].d;

            for (short row = 0; row < N_R0_Q5_K && row0 + (uint)row < args.out_dim; row++) {
                sc16[0] = sc[0] & kmask1;
                sc16[1] = sc[2] & kmask1;
                sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                device const uint16_t *q2 = q1 + 32;
                float4 acc = {0.f, 0.f, 0.f, 0.f};

                FOR_UNROLL (short i = 0; i < 4; ++i) {
                    const uint ql1 = (uint)q1[i];
                    const uint ql2 = (uint)q2[i];
                    const uint hb = (uint)qh[i];
                    acc[0] += yl[2 * i + 0] *
                                  (float)((ql1 & 0x000Fu) + (((hb >> bit0) & 1u) << 4u)) +
                              yl[2 * i + 1] *
                                  (float)(((ql1 >> 8u) & 0x000Fu) + (((hb >> (bit0 + 8u)) & 1u) << 4u));
                    acc[1] += yl[2 * i + 8] *
                                  (float)(((ql1 >> 4u) & 0x000Fu) + (((hb >> bit1) & 1u) << 4u)) +
                              yl[2 * i + 9] *
                                  (float)(((ql1 >> 12u) & 0x000Fu) + (((hb >> (bit1 + 8u)) & 1u) << 4u));
                    acc[2] += yh[2 * i + 0] *
                                  (float)((ql2 & 0x000Fu) + (((hb >> bit2) & 1u) << 4u)) +
                              yh[2 * i + 1] *
                                  (float)(((ql2 >> 8u) & 0x000Fu) + (((hb >> (bit2 + 8u)) & 1u) << 4u));
                    acc[3] += yh[2 * i + 8] *
                                  (float)(((ql2 >> 4u) & 0x000Fu) + (((hb >> bit3) & 1u) << 4u)) +
                              yh[2 * i + 9] *
                                  (float)(((ql2 >> 12u) & 0x000Fu) + (((hb >> (bit3 + 8u)) & 1u) << 4u));
                }

                sumf[row] += dh[0] * (acc[0] * sc8[0] + acc[1] * sc8[1] +
                                      acc[2] * sc8[4] + acc[3] * sc8[5]) -
                             dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                      sumy[2] * sc8[6] + sumy[3] * sc8[7]);

                q1 += args.down_row_bytes / 2;
                qh += args.down_row_bytes / 2;
                sc += args.down_row_bytes / 2;
                dh += args.down_row_bytes / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    for (short row = 0; row < N_R0_Q5_K && row0 + (uint)row < args.out_dim; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0u) {
            out[(uint64_t)token * args.out_dim + row0 + (uint)row] = sum_all;
        }
    }
}

kernel void kernel_glm_q6_K_slots6_down_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *down0,
        device const char *down1,
        device const char *down2,
        device const char *down3,
        device const char *down4,
        device const char *down5,
        device const float *mid,
        device float *out,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const short NSG = 2;
    constexpr uint kmask1 = 0x03u;
    constexpr uint kmask2 = 0x0Cu;
    constexpr uint kmask3 = 0x30u;
    constexpr uint kmask4 = 0xC0u;

    const uint row0 = ((uint)tgpig.x * (uint)NSG + (uint)sgitg) * N_R0_Q6_K;
    const uint token = tgpig.y;
    if (row0 >= args.out_dim || token >= args.n_tokens) return;

    const int nb = args.mid_dim / QK_K;
    float sumf[N_R0_Q6_K] = {0.f};
    float yl[16];
    const short tid = tiisg / 2;
    const short ix = tiisg % 2;
    const short ip = tid / 8;
    const short il = tid % 8;
    const short l0 = 4 * il;
    const short is = 8 * ip + l0 / 16;
    const short y_offset = 128 * ip + l0;
    const short q_offset_l = 64 * ip + l0;
    const short q_offset_h = 32 * ip + l0;

    const uint64_t mid_base = (uint64_t)token * args.mid_token_stride;
    for (uint slot = 0; slot < args.n_expert_used; slot++) {
        device const char *down_cur = down0;
        switch (slot) {
        case 1: down_cur = down1; break;
        case 2: down_cur = down2; break;
        case 3: down_cur = down3; break;
        case 4: down_cur = down4; break;
        case 5: down_cur = down5; break;
        default: break;
        }

        device const block_q6_K *x =
            (device const block_q6_K *)(down_cur +
                (uint64_t)row0 * args.down_row_bytes);
        device const float *yy = mid + mid_base + (uint64_t)slot * args.mid_dim;

        for (int ib = ix; ib < nb; ib += 2) {
            device const uchar *q1 = x[ib].ql + q_offset_l;
            device const uchar *q2 = q1 + 32;
            device const uchar *qh = x[ib].qh + q_offset_h;
            device const char *sc = x[ib].scales + is;
            device const half *dh = &x[ib].d;
            device const float *y = yy + (uint64_t)ib * QK_K + y_offset;

            for (short l = 0; l < 4; ++l) {
                yl[4 * l + 0] = y[l + 0];
                yl[4 * l + 1] = y[l + 32];
                yl[4 * l + 2] = y[l + 64];
                yl[4 * l + 3] = y[l + 96];
            }

            for (short row = 0; row < N_R0_Q6_K && row0 + (uint)row < args.out_dim; row++) {
                float4 sums = {0.f, 0.f, 0.f, 0.f};

                FOR_UNROLL (short l = 0; l < 4; ++l) {
                    const uint h = (uint)qh[l];
                    sums[0] += yl[4 * l + 0] *
                        (float)((int)((q1[l] & 0x0Fu) | ((h & kmask1) << 4u)) - 32);
                    sums[1] += yl[4 * l + 1] *
                        (float)((int)((q2[l] & 0x0Fu) | ((h & kmask2) << 2u)) - 32);
                    sums[2] += yl[4 * l + 2] *
                        (float)((int)((q1[l] >> 4u) | (h & kmask3)) - 32);
                    sums[3] += yl[4 * l + 3] *
                        (float)((int)((q2[l] >> 4u) | ((h & kmask4) >> 2u)) - 32);
                }

                sumf[row] += dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] +
                                      sums[2] * sc[4] + sums[3] * sc[6]);

                q1 += args.down_row_bytes;
                q2 += args.down_row_bytes;
                qh += args.down_row_bytes;
                sc += args.down_row_bytes;
                dh += args.down_row_bytes / 2;
            }
        }
    }

    for (short row = 0; row < N_R0_Q6_K && row0 + (uint)row < args.out_dim; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0u) {
            out[(uint64_t)token * args.out_dim + row0 + (uint)row] = sum_all;
        }
    }
}

kernel void kernel_glm_q6_K_slots8_down_f32(
        constant ds4_metal_glm_routed_moe_args &args,
        device const char *down0,
        device const char *down1,
        device const char *down2,
        device const char *down3,
        device const char *down4,
        device const char *down5,
        device const char *down6,
        device const char *down7,
        device const float *mid,
        device float *out,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const short NSG = 2;
    constexpr uint kmask1 = 0x03u;
    constexpr uint kmask2 = 0x0Cu;
    constexpr uint kmask3 = 0x30u;
    constexpr uint kmask4 = 0xC0u;

    const uint row0 = ((uint)tgpig.x * (uint)NSG + (uint)sgitg) * N_R0_Q6_K;
    const uint token = tgpig.y;
    if (row0 >= args.out_dim || token >= args.n_tokens) return;

    const int nb = args.mid_dim / QK_K;
    float sumf[N_R0_Q6_K] = {0.f};
    float yl[16];
    const short tid = tiisg / 2;
    const short ix = tiisg % 2;
    const short ip = tid / 8;
    const short il = tid % 8;
    const short l0 = 4 * il;
    const short is = 8 * ip + l0 / 16;
    const short y_offset = 128 * ip + l0;
    const short q_offset_l = 64 * ip + l0;
    const short q_offset_h = 32 * ip + l0;

    const uint64_t mid_base = (uint64_t)token * args.mid_token_stride;
    for (uint slot = 0; slot < args.n_expert_used; slot++) {
        device const char *down_cur = down0;
        switch (slot) {
        case 1: down_cur = down1; break;
        case 2: down_cur = down2; break;
        case 3: down_cur = down3; break;
        case 4: down_cur = down4; break;
        case 5: down_cur = down5; break;
        case 6: down_cur = down6; break;
        case 7: down_cur = down7; break;
        default: break;
        }

        device const block_q6_K *x =
            (device const block_q6_K *)(down_cur +
                (uint64_t)row0 * args.down_row_bytes);
        device const float *yy = mid + mid_base + (uint64_t)slot * args.mid_dim;

        for (int ib = ix; ib < nb; ib += 2) {
            device const uchar *q1 = x[ib].ql + q_offset_l;
            device const uchar *q2 = q1 + 32;
            device const uchar *qh = x[ib].qh + q_offset_h;
            device const char *sc = x[ib].scales + is;
            device const half *dh = &x[ib].d;
            device const float *y = yy + (uint64_t)ib * QK_K + y_offset;

            for (short l = 0; l < 4; ++l) {
                yl[4 * l + 0] = y[l + 0];
                yl[4 * l + 1] = y[l + 32];
                yl[4 * l + 2] = y[l + 64];
                yl[4 * l + 3] = y[l + 96];
            }

            for (short row = 0; row < N_R0_Q6_K && row0 + (uint)row < args.out_dim; row++) {
                float4 sums = {0.f, 0.f, 0.f, 0.f};

                FOR_UNROLL (short l = 0; l < 4; ++l) {
                    const uint h = (uint)qh[l];
                    sums[0] += yl[4 * l + 0] *
                        (float)((int)((q1[l] & 0x0Fu) | ((h & kmask1) << 4u)) - 32);
                    sums[1] += yl[4 * l + 1] *
                        (float)((int)((q2[l] & 0x0Fu) | ((h & kmask2) << 2u)) - 32);
                    sums[2] += yl[4 * l + 2] *
                        (float)((int)((q1[l] >> 4u) | (h & kmask3)) - 32);
                    sums[3] += yl[4 * l + 3] *
                        (float)((int)((q2[l] >> 4u) | ((h & kmask4) >> 2u)) - 32);
                }

                sumf[row] += dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] +
                                      sums[2] * sc[4] + sums[3] * sc[6]);

                q1 += args.down_row_bytes;
                q2 += args.down_row_bytes;
                qh += args.down_row_bytes;
                sc += args.down_row_bytes;
                dh += args.down_row_bytes / 2;
            }
        }
    }

    for (short row = 0; row < N_R0_Q6_K && row0 + (uint)row < args.out_dim; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0u) {
            out[(uint64_t)token * args.out_dim + row0 + (uint)row] = sum_all;
        }
    }
}

template <typename type4x4>
void dequantize_q4_K(device const block_q4_K *xb, short il, thread type4x4 &reg) {
    device const uchar *q = xb->qs;

    short is = (il / 4) * 2;
    q = q + (il / 4) * 32 + 16 * (il & 1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il / 2, xb->scales);
    const float d = il < 2 ?
        (float)xb->d :
        (float)xb->d * (1.0f / 16.0f);
    const float min = (float)xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask = il < 2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = dl * (q[i] & mask) - ml;
    }
}

template <typename type4x4>
void dequantize_mlx_affine4_64(
        device const block_mlx_affine4_64 *xb,
        short il,
        thread type4x4 &reg) {
    const float scale = as_type<float>((uint)xb->scale_bf16 << 16u);
    const float bias = as_type<float>((uint)xb->bias_bf16 << 16u);
    const uint value_base = (uint)il * 16u;
    for (uint i = 0u; i < 16u; i++) {
        const uint value = value_base + i;
        const uchar packed = xb->qs[value >> 1u];
        const uint q = (value & 1u) != 0u
            ? (uint)(packed >> 4u)
            : (uint)(packed & 0x0fu);
        reg[i / 4u][i % 4u] = scale * (float)q + bias;
    }
}

template <typename type4x4>
void dequantize_q5_K(device const block_q5_K *xb, short il, thread type4x4 &reg) {
    const short group = il / 2;
    const short l0 = (il & 1) * 16;
    const uchar2 sm = get_scale_min_k4_just2(group, 0, xb->scales);
    const float dl = (float)xb->d * (float)sm.x;
    const float ml = (float)xb->dmin * (float)sm.y;
    device const uchar *q = xb->qs + (group / 2) * 32 + l0;
    device const uchar *qh = xb->qh + l0;
    const uint shift = (uint)(group & 1) * 4u;
    const uchar hmask = (uchar)(1u << (uint)group);

    for (int i = 0; i < 16; ++i) {
        uint v = ((uint)q[i] >> shift) & 0x0Fu;
        v += (qh[i] & hmask) ? 16u : 0u;
        reg[i / 4][i % 4] = dl * (float)v - ml;
    }
}

template <typename type4x4>
void dequantize_q6_K(device const block_q6_K *xb, short il, thread type4x4 &reg) {
    const short n128 = il / 8;
    const short il128 = il & 7;
    const short quarter = il128 / 2;
    const short l0 = (il128 & 1) * 16;
    const uint ql_base = (uint)n128 * 64u;
    const uint qh_base = (uint)n128 * 32u;
    const uint sc_base = (uint)n128 * 8u + (uint)quarter * 2u + (uint)(il128 & 1);
    const float d = (float)xb->d * (float)((int)xb->scales[sc_base]);

    for (int i = 0; i < 16; ++i) {
        const uint l = (uint)l0 + (uint)i;
        uint v;
        if (quarter == 0) {
            v = ((uint)xb->ql[ql_base + l] & 0x0Fu) |
                ((((uint)xb->qh[qh_base + l] >> 0u) & 3u) << 4u);
        } else if (quarter == 1) {
            v = ((uint)xb->ql[ql_base + 32u + l] & 0x0Fu) |
                ((((uint)xb->qh[qh_base + l] >> 2u) & 3u) << 4u);
        } else if (quarter == 2) {
            v = ((uint)xb->ql[ql_base + l] >> 4u) |
                ((((uint)xb->qh[qh_base + l] >> 4u) & 3u) << 4u);
        } else {
            v = ((uint)xb->ql[ql_base + 32u + l] >> 4u) |
                ((((uint)xb->qh[qh_base + l] >> 6u) & 3u) << 4u);
        }
        reg[i / 4][i % 4] = d * (float)((int)v - 32);
    }
}

template <typename type4x4>
void dequantize_iq2_xxs(device const block_iq2_xxs * xb, short il, thread type4x4 & reg) {
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    device const uint16_t * q2 = xb->qs + 4*ib32;
    const uint32_t aux32_g = q2[0] | (q2[1] << 16);
    const uint32_t aux32_s = q2[2] | (q2[3] << 16);
    thread const uint8_t * aux8 = (thread const uint8_t *)&aux32_g;
    const float dl = d * (0.5f + (aux32_s >> 28)) * 0.25f;
    constant uint8_t * grid = (constant uint8_t *)(iq2xxs_grid + aux8[2*il+0]);
    uint8_t signs = ksigns_iq2xs[(aux32_s >> 14*il) & 127];
    for (int i = 0; i < 8; ++i) {
        reg[i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
    grid = (constant uint8_t *)(iq2xxs_grid + aux8[2*il+1]);
    signs = ksigns_iq2xs[(aux32_s >> (14*il+7)) & 127];
    for (int i = 0; i < 8; ++i) {
        reg[2+i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
}

template <typename type4x4>
void dequantize_iq2_xs(device const block_iq2_xs * xb, short il, thread type4x4 & reg) {
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    device const uint16_t * q2 = xb->qs + 4*ib32;
    const float dl = d * (0.5f + ((xb->scales[ib32] >> 4*il) & 0xf)) * 0.25f;
    constant uint8_t * grid = (constant uint8_t *)(iq2xs_grid + (q2[2*il+0] & 511));
    uint8_t signs = ksigns_iq2xs[q2[2*il+0] >> 9];
    for (int i = 0; i < 8; ++i) {
        reg[i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
    grid = (constant uint8_t *)(iq2xs_grid + (q2[2*il+1] & 511));
    signs = ksigns_iq2xs[q2[2*il+1] >> 9];
    for (int i = 0; i < 8; ++i) {
        reg[2+i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
}

template <typename type4x4>
void dequantize_iq3_xxs(device const block_iq3_xxs * xb, short il, thread type4x4 & reg) {
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    device const uint8_t * q3 = xb->qs + 8*ib32;
    device const uint16_t * gas = (device const uint16_t *)(xb->qs + QK_K/4) + 2*ib32;
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float dl = d * (0.5f + (aux32 >> 28)) * 0.5f;
    constant uint8_t * grid1 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+0]);
    constant uint8_t * grid2 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+1]);
    uint8_t signs = ksigns_iq2xs[(aux32 >> 14*il) & 127];
    for (int i = 0; i < 4; ++i) {
        reg[0][i] = dl * grid1[i] * (signs & kmask_iq2xs[i+0] ? -1.f : 1.f);
        reg[1][i] = dl * grid2[i] * (signs & kmask_iq2xs[i+4] ? -1.f : 1.f);
    }
    grid1 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+2]);
    grid2 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+3]);
    signs = ksigns_iq2xs[(aux32 >> (14*il+7)) & 127];
    for (int i = 0; i < 4; ++i) {
        reg[2][i] = dl * grid1[i] * (signs & kmask_iq2xs[i+0] ? -1.f : 1.f);
        reg[3][i] = dl * grid2[i] * (signs & kmask_iq2xs[i+4] ? -1.f : 1.f);
    }
}

template <typename type4x4>
void dequantize_iq4_xs(device const block_iq4_xs * xb, short il, thread type4x4 & reg) {
    const int ib32 = il/2;
    il = il%2;
    device const uint32_t * q4 = (device const uint32_t *)xb->qs + 4*ib32;
    const int ls = ((xb->scales_l[ib32/2] >> 4*(ib32%2)) & 0xf) |
                   (((xb->scales_h >> 2*ib32) & 3) << 4);
    const float d = (float)xb->d * (ls - 32);
    uint32_t aux32;
    thread const uint8_t * q8 = (thread const uint8_t *)&aux32;
    for (int i = 0; i < 4; ++i) {
        aux32 = (q4[i] >> 4*il) & 0x0f0f0f0f;
        reg[i][0] = d * kvalues_iq4nl_f[q8[0]];
        reg[i][1] = d * kvalues_iq4nl_f[q8[1]];
        reg[i][2] = d * kvalues_iq4nl_f[q8[2]];
        reg[i][3] = d * kvalues_iq4nl_f[q8[3]];
    }
}

struct ds4_metal_args_mul_mv_id {
    int32_t  nei0;
    int32_t  nei1;
    uint64_t nbi1;
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    int32_t  ne0;
    int32_t  ne1;
    uint64_t nb1;
    int32_t  nr0;
};

struct ds4_metal_moe_expert_group_args {
    uint32_t expert_base;
    uint32_t expert_count;
    uint32_t accumulate;
    uint32_t pad0;
};

struct ds4_metal_q4_gather_slots6_args {
    uint64_t expert_bytes;
    uint32_t group_size;
    uint32_t n_slots;
};

struct ds4_metal_q4_expert_table {
    array<device const char *, 384> experts [[id(0)]];
};

struct ds4_metal_expert_address_table {
    device const uint64_t *addrs;
};

struct ds4_metal_stream_expert_validate_args {
    uint32_t n_total_expert;
    uint32_t n_expert;
};

struct ds4_metal_stream_expert_split_args {
    uint32_t active_mask;
    uint32_t accumulate;
};

struct ds4_metal_args_mul_mm_id_map0 {
    int32_t  ne02;
    int32_t  ne10;
    int32_t  ne11;
    uint64_t nb11;
    uint64_t nb12;
    int32_t  ne21;
    int32_t  ne20;
    uint64_t nb21;
};

struct ds4_metal_args_mul_mm_id {
    int32_t  ne00;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne20;
    int32_t  ne21;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
};

template<int nr0, typename args_t>
void kernel_mul_mv_q2_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q2_K * x = (device const block_q2_K *) (src0 + offset0);
    device const float      * y = (device const float      *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const short ix = tiisg/8;  // 0...3
    const short it = tiisg%8;  // 0...7
    const short iq = it/4;     // 0 or 1
    const short ir = it%4;     // 0...3
    const short is = (8*ir)/16;// 0 or 1

    device const float * y4 = y + ix * QK_K + 128 * iq + 8 * ir;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
            yl[i+ 8] = y4[i+32]; sumy[1] += yl[i+ 8];
            yl[i+16] = y4[i+64]; sumy[2] += yl[i+16];
            yl[i+24] = y4[i+96]; sumy[3] += yl[i+24];
        }

        device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales + 8*iq + is;
        device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half     * dh = &x[ib].d;

        for (short row = 0; row < nr0; row++) {
            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            for (int i = 0; i < 8; i += 2) {
                acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);
                acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);
                acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);
                acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);
                acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);
                acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);
                acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);
                acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);
            }
            float dall = dh[0];
            float dmin = dh[1] * 1.f/16.f;
            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
                                 (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * 1.f/ 4.f +
                                 (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * 1.f/16.f +
                                 (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * 1.f/64.f) -
                         dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) + sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));

            qs += args.nb01/2;
            sc += args.nb01;
            dh += args.nb01/2;
        }

        y4 += 4 * QK_K;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

template<int nr0, typename args_t>
void kernel_mul_mv_q4_K_f32_impl(
        args_t args,
        device const char *src0,
        device const char *src1,
        device       char *dst,
        threadgroup  char *shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    const int nb = args.ne00 / QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;

    const uint64_t offset0 = first_row * args.nb01 + (i12 / args.r2) * args.nb02 + (i13 / args.r3) * args.nb03;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;

    device const block_q4_K *x = (device const block_q4_K *)(src0 + offset0);
    device const float *y = (device const float *)(src1 + offset1);

    float yl[16];
    float yh[16];
    float sumf[nr0] = {0.f};

    device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};

        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
            yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
            yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
        }

        device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
        device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half *dh = &x[ib].d;

        for (short row = 0; row < nr0; row++) {
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const uint16_t *q2 = q1 + 32;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
            }

            sumf[row] += dh[0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                  (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                  (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                  (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1 += args.nb01 / 2;
            sc += args.nb01 / 2;
            dh += args.nb01 / 2;
        }

        y4 += 4 * QK_K;
    }

    device float *dst_f32 = (device float *)dst + (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }

    (void)shmem;
}

/* One SIMD lane owns two adjacent weights in each 64-value group. */
template<int nr0, typename args_t>
void kernel_mul_mv_mlx_affine4_64_f32_impl(
        args_t args,
        device const char *src0,
        device const char *src1,
        device       char *dst,
        threadgroup  char *shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;
    const uint64_t offset0 =
        (uint64_t)first_row * args.nb01 +
        (uint64_t)(i12 / args.r2) * args.nb02 +
        (uint64_t)(i13 / args.r3) * args.nb03;
    const uint64_t offset1 =
        (uint64_t)r1 * args.nb11 +
        (uint64_t)i12 * args.nb12 +
        (uint64_t)i13 * args.nb13;
    device const char *matrix = src0 + offset0;
    device const float *y = (device const float *)(src1 + offset1);
    device float *dst_f32 =
        (device float *)dst +
        (uint64_t)im * args.ne0 * args.ne1 +
        (uint64_t)r1 * args.ne0;
    const int groups = args.ne00 / 64;
    float sumf[nr0] = {0.f};

    for (short row = 0; row < nr0; row++) {
        if (first_row + row >= args.ne0) break;
        device const block_mlx_affine4_64 *blocks =
            (device const block_mlx_affine4_64 *)(
                matrix + (uint64_t)row * args.nb01);
        float acc = 0.f;
        for (int group = 0; group < groups; group++) {
            device const block_mlx_affine4_64 *block = blocks + group;
            const uchar packed = block->qs[tiisg];
            const float scale =
                as_type<float>((uint)block->scale_bf16 << 16u);
            const float bias =
                as_type<float>((uint)block->bias_bf16 << 16u);
            const uint value = (uint)group * 64u + (uint)tiisg * 2u;
            acc +=
                (scale * (float)(packed & 0x0fu) + bias) * y[value] +
                (scale * (float)(packed >> 4u) + bias) * y[value + 1u];
        }
        sumf[row] = acc;
    }
    for (short row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum;
    }
    (void)shmem;
}

template<int nr0, typename args_t>
void kernel_mul_mv_q5_K_f32_impl(
        args_t args,
        device const char *src0,
        device const char *src1,
        device       char *dst,
        threadgroup  char *shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    const int nb = args.ne00 / QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const int first_row = (r0 * NSG + sgitg) * nr0;
    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;
    const uint64_t offset0 =
        (uint64_t)first_row * args.nb01 +
        (uint64_t)(i12 / args.r2) * args.nb02 +
        (uint64_t)(i13 / args.r3) * args.nb03;
    const uint64_t offset1 =
        (uint64_t)r1 * args.nb11 +
        (uint64_t)i12 * args.nb12 +
        (uint64_t)i13 * args.nb13;

    device const block_q5_K *x =
        (device const block_q5_K *)(src0 + offset0);
    device const float *yy = (device const float *)(src1 + offset1);

    float sumf[nr0] = {0.f};
    float yl[16];
    float yh[16];

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short tid = tiisg / 4;
    const short ix  = tiisg % 4;
    const short iq  = tid / 4;
    const short ir  = tid % 4;
    const short l0 = 8 * ir;
    const short q_offset = 32 * iq + l0;
    const short y_offset = 64 * iq + l0;

    const uint8_t hm1 = 1u << (2 * iq);
    const uint8_t hm2 = hm1 << 1;
    const uint8_t hm3 = hm1 << 4;
    const uint8_t hm4 = hm2 << 4;

    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;
    device const float *y1 = yy + ix * QK_K + y_offset;

    for (int i = ix; i < nb; i += 4) {
        device const uint8_t *q1 = x[i].qs + q_offset;
        device const uint8_t *qh = x[i].qh + l0;
        device const half *dh = &x[i].d;
        device const uint16_t *a =
            (device const uint16_t *)x[i].scales + iq;
        device const float *y2 = y1 + 128;

        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short l = 0; l < 8; ++l) {
            yl[l + 0] = y1[l +  0]; sumy[0] += yl[l + 0];
            yl[l + 8] = y1[l + 32]; sumy[1] += yl[l + 8];
            yh[l + 0] = y2[l +  0]; sumy[2] += yh[l + 0];
            yh[l + 8] = y2[l + 32]; sumy[3] += yh[l + 8];
        }

        for (short row = 0; row < nr0; ++row) {
            device const uint8_t *q2 = q1 + 64;

            sc16[0] = a[0] & kmask1;
            sc16[1] = a[2] & kmask1;
            sc16[2] = ((a[4] >> 0) & kmask2) |
                      ((a[0] & kmask3) >> 2);
            sc16[3] = ((a[4] >> 4) & kmask2) |
                      ((a[2] & kmask3) >> 2);

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            FOR_UNROLL (short l = 0; l < 8; ++l) {
                const uint8_t h = qh[l];
                acc1[0] += yl[l + 0] * (q1[l] & 0x0F);
                acc1[1] += yl[l + 8] * (q1[l] & 0xF0);
                acc1[2] += yh[l + 0] * (q2[l] & 0x0F);
                acc1[3] += yh[l + 8] * (q2[l] & 0xF0);
                acc2[0] += h & hm1 ? yl[l + 0] : 0.f;
                acc2[1] += h & hm2 ? yl[l + 8] : 0.f;
                acc2[2] += h & hm3 ? yh[l + 0] : 0.f;
                acc2[3] += h & hm4 ? yh[l + 8] : 0.f;
            }

            sumf[row] +=
                dh[0] * (sc8[0] * (acc1[0]      + 16.f * acc2[0]) +
                         sc8[1] * (acc1[1]/16.f + 16.f * acc2[1]) +
                         sc8[4] * (acc1[2]      + 16.f * acc2[2]) +
                         sc8[5] * (acc1[3]/16.f + 16.f * acc2[3])) -
                dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                         sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1 += args.nb01;
            qh += args.nb01;
            dh += args.nb01 / 2;
            a  += args.nb01 / 2;
        }

        y1 += 4 * QK_K;
    }

    device float *dst_f32 =
        (device float *)dst +
        (uint64_t)im * args.ne0 * args.ne1 +
        (uint64_t)r1 * args.ne0;
    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        const float total = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = total;
        }
    }

    (void)shmem;
}

template<int nr0, typename args_t>
void kernel_mul_mv_q6_K_f32_impl(
        args_t args,
        device const char *src0,
        device const char *src1,
        device       char *dst,
        threadgroup  char *shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    constexpr uint8_t kmask1 = 0x03;
    constexpr uint8_t kmask2 = 0x0C;
    constexpr uint8_t kmask3 = 0x30;
    constexpr uint8_t kmask4 = 0xC0;

    const int nb = args.ne00 / QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const int first_row = (r0 * NSG + sgitg) * nr0;
    const uint i12 = im % args.ne12;
    const uint i13 = im / args.ne12;
    const uint64_t offset0 =
        (uint64_t)first_row * args.nb01 +
        (uint64_t)(i12 / args.r2) * args.nb02 +
        (uint64_t)(i13 / args.r3) * args.nb03;
    const uint64_t offset1 =
        (uint64_t)r1 * args.nb11 +
        (uint64_t)i12 * args.nb12 +
        (uint64_t)i13 * args.nb13;

    device const block_q6_K *x =
        (device const block_q6_K *)(src0 + offset0);
    device const float *yy = (device const float *)(src1 + offset1);

    float sumf[nr0] = {0.f};
    float yl[16];

    const short tid = tiisg / 2;
    const short ix  = tiisg % 2;
    const short ip  = tid / 8;
    const short il  = tid % 8;
    const short l0  = 4 * il;
    const short is  = 8 * ip + l0 / 16;
    const short y_offset   = 128 * ip + l0;
    const short q_offset_l =  64 * ip + l0;
    const short q_offset_h =  32 * ip + l0;

    for (int i = ix; i < nb; i += 2) {
        device const uint8_t *q1 = x[i].ql + q_offset_l;
        device const uint8_t *q2 = q1 + 32;
        device const uint8_t *qh = x[i].qh + q_offset_h;
        device const char    *sc = x[i].scales + is;
        device const half    *dh = &x[i].d;
        device const float *y = yy + i * QK_K + y_offset;

        for (short l = 0; l < 4; ++l) {
            yl[4 * l + 0] = y[l +  0];
            yl[4 * l + 1] = y[l + 32];
            yl[4 * l + 2] = y[l + 64];
            yl[4 * l + 3] = y[l + 96];
        }

        for (short row = 0; row < nr0; ++row) {
            float4 sums = {0.f, 0.f, 0.f, 0.f};
            FOR_UNROLL (short l = 0; l < 4; ++l) {
                sums[0] += yl[4*l + 0] *
                    ((int8_t)((q1[l] & 0x0F) |
                              ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] *
                    ((int8_t)((q2[l] & 0x0F) |
                              ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] *
                    ((int8_t)((q1[l] >> 4) |
                              ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] *
                    ((int8_t)((q2[l] >> 4) |
                              ((qh[l] & kmask4) >> 2)) - 32);
            }

            sumf[row] += dh[0] *
                (sums[0] * sc[0] + sums[1] * sc[2] +
                 sums[2] * sc[4] + sums[3] * sc[6]);

            q1 += args.nb01;
            q2 += args.nb01;
            qh += args.nb01;
            sc += args.nb01;
            dh += args.nb01 / 2;
        }
    }

    device float *dst_f32 =
        (device float *)dst +
        (uint64_t)im * args.ne0 * args.ne1 +
        (uint64_t)r1 * args.ne0;
    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        const float total = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = total;
        }
    }

    (void)shmem;
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq2_xxs_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq2_xxs * x = (device const block_iq2_xxs *) (src0 + offset0);
    device const float         * y = (device const float         *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint64_t * svalues = (threadgroup uint64_t *)(shmem);
    threadgroup uint8_t  * ssigns  = (threadgroup uint8_t  *)(svalues + 256);
    {
        int nval = 4;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = ds4_metal_iq2xxs_grid[pos + i];
        nval = 2;
        pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) ssigns[pos+i] = ds4_metal_ksigns_iq2xs[pos+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_xxs * xr = x + ibl;
        device const uint16_t * q2 = xr->qs + 4 * ib;
        device const half * dh = &xr->d;

        for (short row = 0; row < nr0; row++) {
            const float db = dh[0];
            device const uint8_t * aux8 = (device const uint8_t *)q2;
            const uint32_t aux32 = q2[2] | (q2[3] << 16);
            const float d = db * (0.5f + (aux32 >> 28));

            float sum = 0;
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(svalues + aux8[l]);
                const uint8_t signs = ssigns[(aux32 >> 7*l) & 127];
                for (short j = 0; j < 8; ++j) {
                    sum += yl[8*l + j] * grid[j] * (signs & ds4_metal_kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumf[row] += d * sum;

            dh += args.nb01/2;
            q2 += args.nb01/2;
        }

        y4 += 32 * 32;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all * 0.25f;
        }
    }
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq2_xs_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    const int nb = args.ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const int first_row = (r0 * NSG + sgitg) * nr0;
    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;
    const uint64_t offset0 = first_row*args.nb01 +
        (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 = r1*args.nb11 +
        i12*args.nb12 + i13*args.nb13;
    device const block_iq2_xs * x =
        (device const block_iq2_xs *)(src0 + offset0);
    device const float * y = (device const float *)(src1 + offset1);

    float yl[32];
    float sumf[nr0] = {0.f};
    const int nb32 = nb * (QK_K / 32);
    threadgroup uint64_t * svalues = (threadgroup uint64_t *)shmem;
    threadgroup uint8_t * ssigns =
        (threadgroup uint8_t *)(svalues + 512);
    {
        int nval = 8;
        int pos = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) {
            svalues[pos + i] = iq2xs_grid[pos + i];
        }
        nval = 2;
        pos = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) {
            ssigns[pos + i] = ksigns_iq2xs[pos + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;
    device const float * y4 = y + 32 * ix;
    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 32; ++i) yl[i] = y4[i];
        const int ibl = ib32 / (QK_K / 32);
        const int ib = ib32 % (QK_K / 32);
        device const block_iq2_xs * xr = x + ibl;
        device const uint16_t * q2 = xr->qs + 4 * ib;
        device const uint8_t * sc = xr->scales + ib;
        device const half * dh = &xr->d;

        for (short row = 0; row < nr0; row++) {
            const float db = dh[0];
            const float d1 = db * (0.5f + (sc[0] & 0xf));
            const float d2 = db * (0.5f + (sc[0] >> 4));
            float sum1 = 0.f;
            float sum2 = 0.f;
            for (short l = 0; l < 2; ++l) {
                const threadgroup uint8_t * grid =
                    (const threadgroup uint8_t *)(svalues + (q2[l] & 511));
                const uint8_t signs = ssigns[q2[l] >> 9];
                for (short j = 0; j < 8; ++j) {
                    sum1 += yl[8*l + j] * grid[j] *
                        (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            for (short l = 2; l < 4; ++l) {
                const threadgroup uint8_t * grid =
                    (const threadgroup uint8_t *)(svalues + (q2[l] & 511));
                const uint8_t signs = ssigns[q2[l] >> 9];
                for (short j = 0; j < 8; ++j) {
                    sum2 += yl[8*l + j] * grid[j] *
                        (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumf[row] += d1 * sum1 + d2 * sum2;
            dh += args.nb01/2;
            q2 += args.nb01/2;
            sc += args.nb01;
        }
        y4 += 32 * 32;
    }

    device float * dst_f32 = (device float *)dst +
        (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all * 0.25f;
    }
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq3_xxs_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    const int nb = args.ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const int first_row = (r0 * NSG + sgitg) * nr0;
    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;
    const uint64_t offset0 = first_row*args.nb01 +
        (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 = r1*args.nb11 +
        i12*args.nb12 + i13*args.nb13;
    device const block_iq3_xxs * x =
        (device const block_iq3_xxs *)(src0 + offset0);
    device const float * y = (device const float *)(src1 + offset1);

    float yl[32];
    float sumf[nr0] = {0.f};
    const int nb32 = nb * (QK_K / 32);
    threadgroup uint32_t * svalues = (threadgroup uint32_t *)shmem;
    threadgroup uint8_t * ssigns =
        (threadgroup uint8_t *)(svalues + 256);
    {
        int nval = 4;
        int pos = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) {
            svalues[pos + i] = iq3xxs_grid[pos + i];
        }
        nval = 2;
        pos = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) {
            ssigns[pos + i] = ksigns_iq2xs[pos + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;
    device const float * y4 = y + 32 * ix;
    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 32; ++i) yl[i] = y4[i];
        const int ibl = ib32 / (QK_K / 32);
        const int ib = ib32 % (QK_K / 32);
        device const block_iq3_xxs * xr = x + ibl;
        device const uint8_t * q3 = xr->qs + 8 * ib;
        device const uint16_t * gas =
            (device const uint16_t *)(xr->qs + QK_K/4) + 2 * ib;
        device const half * dh = &xr->d;

        for (short row = 0; row < nr0; row++) {
            const uint32_t aux32 = gas[0] | (gas[1] << 16);
            const float d = (float)dh[0] * (0.5f + (aux32 >> 28));
            float2 sum = {0.f};
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint8_t * grid1 =
                    (const threadgroup uint8_t *)(svalues + q3[2*l+0]);
                const threadgroup uint8_t * grid2 =
                    (const threadgroup uint8_t *)(svalues + q3[2*l+1]);
                const uint8_t signs = ssigns[(aux32 >> 7*l) & 127];
                for (short j = 0; j < 4; ++j) {
                    sum[0] += yl[8*l + j] * grid1[j] *
                        (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                    sum[1] += yl[8*l + j + 4] * grid2[j] *
                        (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
                }
            }
            sumf[row] += d * (sum[0] + sum[1]);
            dh += args.nb01/2;
            q3 += args.nb01;
            gas += args.nb01/2;
        }
        y4 += 32 * 32;
    }

    device float * dst_f32 = (device float *)dst +
        (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all * 0.5f;
    }
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq4_xs_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    threadgroup float * shmem_f32 = (threadgroup float *)shmem;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const int first_row = (r0 * NSG + sgitg) * nr0;
    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;
    const uint64_t offset0 = first_row*args.nb01 +
        (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 = r1*args.nb11 +
        i12*args.nb12 + i13*args.nb13;
    device const block_iq4_xs * x =
        (device const block_iq4_xs *)(src0 + offset0);
    device const float * y = (device const float *)(src1 + offset1);
    const int nb = args.ne00/QK_K;
    const int ns01 = args.nb01/args.nb00;
    const short ix = tiisg/16;
    const short it = tiisg%16;
    const short ib = it/2;
    const short il = it%2;

    shmem_f32[tiisg] = kvalues_iq4nl_f[tiisg%16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf[nr0] = {0.f};
    device const float * yb = y + ix * QK_K + ib * 32 + il * 8;
    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *)aux32;

    for (int ibl = ix; ibl < nb && ibl < ns01; ibl += 2) {
        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];
        for (short row = 0; row < nr0; ++row) {
            device const block_iq4_xs & xb = x[row*ns01 + ibl];
            device const uint32_t * q4 =
                (device const uint32_t *)(xb.qs + 16*ib + 8*il);
            float4 acc1 = {0.f};
            float4 acc2 = {0.f};
            aux32[0] = q4[0] & 0x0f0f0f0f;
            aux32[1] = (q4[0] >> 4) & 0x0f0f0f0f;
            float4 qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]],
                          shmem_f32[q8[2]], shmem_f32[q8[3]]};
            float4 qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]],
                          shmem_f32[q8[6]], shmem_f32[q8[7]]};
            acc1 += yl[0] * qf1;
            acc2 += yl[1] * qf2;
            aux32[0] = q4[1] & 0x0f0f0f0f;
            aux32[1] = (q4[1] >> 4) & 0x0f0f0f0f;
            qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]],
                   shmem_f32[q8[2]], shmem_f32[q8[3]]};
            qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]],
                   shmem_f32[q8[6]], shmem_f32[q8[7]]};
            acc1 += yl[2] * qf1;
            acc2 += yl[3] * qf2;
            acc1 += acc2;
            const int ls =
                (((xb.scales_l[ib/2] >> 4*(ib%2)) & 0xf) |
                 (((xb.scales_h >> 2*ib) & 3) << 4)) - 32;
            sumf[row] += (float)xb.d * ls *
                (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
        }
        yb += 2 * QK_K;
    }

    device float * dst_f32 = (device float *)dst +
        (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }
}

template<int nr0>
void kernel_mul_mv_iq2_xxs_pair_f32_impl(
        ds4_metal_args_mul_mv args,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq2_xxs * xg = (device const block_iq2_xxs *) (src0_gate + offset0);
    device const block_iq2_xxs * xu = (device const block_iq2_xxs *) (src0_up   + offset0);
    device const float         * y  = (device const float         *) (src1      + offset1);

    float yl[32];
    float sumg[nr0]={0.f};
    float sumu[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint64_t * svalues = (threadgroup uint64_t *)(shmem);
    threadgroup uint8_t  * ssigns  = (threadgroup uint8_t  *)(svalues + 256);
    {
        int nval = 4;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = ds4_metal_iq2xxs_grid[pos + i];
        nval = 2;
        pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) ssigns[pos+i] = ds4_metal_ksigns_iq2xs[pos+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;
    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_xxs * xgr = xg + ibl;
        device const block_iq2_xxs * xur = xu + ibl;
        device const uint16_t * qg = xgr->qs + 4 * ib;
        device const uint16_t * qu = xur->qs + 4 * ib;
        device const half * dhg = &xgr->d;
        device const half * dhu = &xur->d;

        for (short row = 0; row < nr0; row++) {
            device const uint8_t * aux8g = (device const uint8_t *)qg;
            device const uint8_t * aux8u = (device const uint8_t *)qu;
            const uint32_t aux32g = qg[2] | (qg[3] << 16);
            const uint32_t aux32u = qu[2] | (qu[3] << 16);
            const float dg = (float)dhg[0] * (0.5f + (aux32g >> 28));
            const float du = (float)dhu[0] * (0.5f + (aux32u >> 28));

            float sg = 0;
            float su = 0;
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint8_t * gridg = (const threadgroup uint8_t *)(svalues + aux8g[l]);
                const threadgroup uint8_t * gridu = (const threadgroup uint8_t *)(svalues + aux8u[l]);
                const uint8_t signg = ssigns[(aux32g >> 7*l) & 127];
                const uint8_t signu = ssigns[(aux32u >> 7*l) & 127];
                for (short j = 0; j < 8; ++j) {
                    const float v = yl[8*l + j];
                    sg += v * gridg[j] * (signg & ds4_metal_kmask_iq2xs[j] ? -1.f : 1.f);
                    su += v * gridu[j] * (signu & ds4_metal_kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumg[row] += dg * sg;
            sumu[row] += du * su;

            dhg += args.nb01/2;
            dhu += args.nb01/2;
            qg  += args.nb01/2;
            qu  += args.nb01/2;
        }

        y4 += 32 * 32;
    }

    device float * dst_gate_f32 = (device float *) dst_gate + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
    device float * dst_up_f32   = (device float *) dst_up   + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        const float sum_gate = simd_sum(sumg[row]);
        const float sum_up   = simd_sum(sumu[row]);
        if (tiisg == 0) {
            dst_gate_f32[first_row + row] = sum_gate * 0.25f;
            dst_up_f32[first_row + row]   = sum_up   * 0.25f;
        }
    }
}

typedef void (kernel_mul_mv2_disp_t)(
        ds4_metal_args_mul_mv args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg);

template<kernel_mul_mv2_disp_t disp_fn>
void mmv_fn(
        ds4_metal_args_mul_mv args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiitg,
        ushort tiisg,
        ushort sgitg) {
    disp_fn(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

typedef decltype(mmv_fn<kernel_mul_mv_q2_K_f32_impl<N_R0_Q2_K>>) mul_mv_id_disp_fn_t;

// Decode-time expert matvec. The ids tensor selects the routed expert for each
// slot, then this wrapper invokes the quantized row kernel for Q8_0, Q2_K, or
// IQ2_XXS weights without materializing per-expert dispatches on the CPU.
template<mul_mv_id_disp_fn_t disp_fn>
kernel void kernel_mul_mv_id(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    (void)tiitg;

    const int iid1 = tgpig.z/args.nei0;
    const int idx  = tgpig.z%args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *) (ids + iid1*args.nbi1))[idx];

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    const int64_t i1 = idx;
    const int64_t i2 = i12;

    device const char * src0_cur = src0s + i02*args.nb02;
    device const char * src1_cur = src1  + i11*args.nb11 + i12*args.nb12;

    device char * dst_cur = dst + (i1*args.ne0 + i2*args.ne1*args.ne0)*sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        /*.ne00 =*/ args.ne00,
        /*.ne01 =*/ args.ne01,
        /*.ne02 =*/ 1,
        /*.nb00 =*/ args.nb00,
        /*.nb01 =*/ args.nb01,
        /*.nb02 =*/ args.nb02,
        /*.nb03 =*/ args.nb02,
        /*.ne10 =*/ args.ne10,
        /*.ne11 =*/ 1,
        /*.ne12 =*/ 1,
        /*.nb10 =*/ args.nb10,
        /*.nb11 =*/ args.nb11,
        /*.nb12 =*/ args.nb12,
        /*.nb13 =*/ args.nb12,
        /*.ne0  =*/ args.ne0,
        /*.ne1  =*/ 1,
        /*.nr0  =*/ args.nr0,
        /*.r2   =*/ 1,
        /*.r3   =*/ 1,
    };

    disp_fn(
        args0,
        /* src0 */ src0_cur,
        /* src1 */ src1_cur,
        /* dst  */ dst_cur,
        shmem,
        tgpig,
        tiitg,
        tiisg,
        sgitg);
}

typedef decltype(kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q2_K_f32_impl<N_R0_Q2_K>>>) kernel_mul_mv_id_q_t;
typedef decltype(kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0>>>) kernel_mul_mv_id_q8_0_t;

// Host-visible decode MoE matvec variants for the DS4 quant formats.
template [[host_name("kernel_mul_mv_id_q8_0_f32")]]    kernel kernel_mul_mv_id_q8_0_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0>>>;
template [[host_name("kernel_mul_mv_id_q2_K_f32")]]    kernel kernel_mul_mv_id_q_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q2_K_f32_impl<N_R0_Q2_K>>>;
// Decode-time Q4_K matrix-vector multiply over a plain (non-indexed) matrix.
// The routed path already instantiates this dot product through the id
// variant above; this host entry exposes the same reduction for the DENSE
// projections, where the bytes actually dominate decode: attention output/q_b
// and the shared expert are read in full on every token.
[[host_name("kernel_mul_mv_q4_K_f32")]]
kernel void kernel_mul_mv_q4_K_f32(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K, constant ds4_metal_args_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

template [[host_name("kernel_mul_mv_id_q4_K_f32")]]    kernel kernel_mul_mv_id_q_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>>>;
template [[host_name("kernel_mul_mv_id_mlx_affine4_64_f32")]] kernel kernel_mul_mv_id_q_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_mlx_affine4_64_f32_impl<N_R0_MLX_AFFINE4>>>;
template [[host_name("kernel_mul_mv_id_q5_K_f32")]]    kernel kernel_mul_mv_id_q_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q5_K_f32_impl<N_R0_Q5_K>>>;
template [[host_name("kernel_mul_mv_id_q6_K_f32")]]    kernel kernel_mul_mv_id_q_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q6_K_f32_impl<N_R0_Q6_K>>>;
template [[host_name("kernel_mul_mv_id_iq2_xxs_f32")]] kernel kernel_mul_mv_id_q_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq2_xxs_f32_impl<N_R0_IQ2_XXS>>>;
template [[host_name("kernel_mul_mv_id_iq2_xs_f32")]]  kernel kernel_mul_mv_id_q_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq2_xs_f32_impl<N_R0_IQ2_XS>>>;
template [[host_name("kernel_mul_mv_id_iq3_xxs_f32")]] kernel kernel_mul_mv_id_q_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq3_xxs_f32_impl<N_R0_IQ3_XXS>>>;
template [[host_name("kernel_mul_mv_id_iq4_xs_f32")]]  kernel kernel_mul_mv_id_q_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq4_xs_f32_impl<N_R0_IQ4_XS>>>;

template<mul_mv_id_disp_fn_t disp_fn>
kernel void kernel_mul_mv_slots6(
        constant ds4_metal_args_mul_mv_id &args,
        device const char *src0_0,
        device const char *src0_1,
        device const char *src0_2,
        device const char *src0_3,
        device const char *src0_4,
        device const char *src0_5,
        device const char *src1,
        device       char *dst,
        threadgroup  char *shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx = tgpig.z % args.nei0;
    tgpig.z = 0;

    device const char *src0_cur = src0_0;
    switch (idx) {
    case 1: src0_cur = src0_1; break;
    case 2: src0_cur = src0_2; break;
    case 3: src0_cur = src0_3; break;
    case 4: src0_cur = src0_4; break;
    case 5: src0_cur = src0_5; break;
    default: break;
    }

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;
    device const char *src1_cur =
        src1 + i11 * args.nb11 + i12 * args.nb12;
    device char *dst_cur =
        dst + (idx * args.ne0 +
               i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };
    disp_fn(args0,
            src0_cur,
            src1_cur,
            dst_cur,
            shmem,
            tgpig,
            tiitg,
            tiisg,
            sgitg);
}

typedef decltype(kernel_mul_mv_slots6<
    mmv_fn<kernel_mul_mv_iq2_xs_f32_impl<N_R0_IQ2_XS>>>)
    kernel_mul_mv_slots6_q_t;

template [[host_name("kernel_mul_mv_slots6_iq2_xs_f32")]]
kernel kernel_mul_mv_slots6_q_t kernel_mul_mv_slots6<
    mmv_fn<kernel_mul_mv_iq2_xs_f32_impl<N_R0_IQ2_XS>>>;
template [[host_name("kernel_mul_mv_slots6_iq3_xxs_f32")]]
kernel kernel_mul_mv_slots6_q_t kernel_mul_mv_slots6<
    mmv_fn<kernel_mul_mv_iq3_xxs_f32_impl<N_R0_IQ3_XXS>>>;
template [[host_name("kernel_mul_mv_slots6_iq4_xs_f32")]]
kernel kernel_mul_mv_slots6_q_t kernel_mul_mv_slots6<
    mmv_fn<kernel_mul_mv_iq4_xs_f32_impl<N_R0_IQ4_XS>>>;

// DS4 attention output low projection, specialized for the fixed block
// diagonal mapping used by the model:
//
//     low[token, group, rank] = heads[token, group, :] * Woa[group, rank, :]
//
// The generic GGML-style id matvec supports arbitrary routed expert ids.  Here
// the id is always equal to the group number, so this wrapper keeps the exact
// Q8_0 dot kernel but removes the id-buffer load and the CPU-side id table.
kernel void kernel_dsv4_attn_out_low_q8_0_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z/args.nei0;
    const int idx  = tgpig.z%args.nei0;

    tgpig.z = 0;

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char * src0_cur = src0s + idx*args.nb02;
    device const char * src1_cur = src1  + i11*args.nb11 + i12*args.nb12;
    device       char * dst_cur  = dst   + (idx*args.ne0 + i12*args.ne1*args.ne0)*sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        /*.ne00 =*/ args.ne00,
        /*.ne01 =*/ args.ne01,
        /*.ne02 =*/ 1,
        /*.nb00 =*/ args.nb00,
        /*.nb01 =*/ args.nb01,
        /*.nb02 =*/ args.nb02,
        /*.nb03 =*/ args.nb02,
        /*.ne10 =*/ args.ne10,
        /*.ne11 =*/ 1,
        /*.ne12 =*/ 1,
        /*.nb10 =*/ args.nb10,
        /*.nb11 =*/ args.nb11,
        /*.nb12 =*/ args.nb12,
        /*.nb13 =*/ args.nb12,
        /*.ne0  =*/ args.ne0,
        /*.ne1  =*/ 1,
        /*.nr0  =*/ args.nr0,
        /*.r2   =*/ 1,
        /*.r3   =*/ 1,
    };

    kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0, thread ds4_metal_args_mul_mv &>(
        args0,
        src0_cur,
        src1_cur,
        dst_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
}

kernel void kernel_mul_mv_id_iq2_xxs_pair_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z/args.nei0;
    const int idx  = tgpig.z%args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *) (ids + iid1*args.nbi1))[idx];

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char * src0_gate_cur = src0_gate + i02*args.nb02;
    device const char * src0_up_cur   = src0_up   + i02*args.nb02;
    device const char * src1_cur      = src1      + i11*args.nb11 + i12*args.nb12;

    device char * dst_gate_cur = dst_gate + (idx*args.ne0 + i12*args.ne1*args.ne0)*sizeof(float);
    device char * dst_up_cur   = dst_up   + (idx*args.ne0 + i12*args.ne1*args.ne0)*sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    (void)tiitg;
    kernel_mul_mv_iq2_xxs_pair_f32_impl<N_R0_IQ2_XXS>(
        args0,
        src0_gate_cur,
        src0_up_cur,
        src1_cur,
        dst_gate_cur,
        dst_up_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
}

// Decode-only routed expert gate/up projection fused with the DS4 activation:
//
//     mid = silu(clamp(gate)) * clamp(up) * route_weight
//
// The quantized dot products are intentionally the same IQ2_XXS paired path as
// `kernel_mul_mv_id_iq2_xxs_pair_f32`.  The only extra work is done by lane 0
// after each exact reduced row has been produced.  This removes the separate
// routed activation dispatch and avoids rereading the gate/up rows before the
// down projection.  The host uses this only for the normal release path where
// diagnostics do not request clamped gate/up intermediates.
kernel void kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *) (ids + iid1 * args.nbi1))[idx];
    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_IQ2_XXS;
    const int nb32 = nb * (QK_K / 32);

    device const block_iq2_xxs *xg =
        (device const block_iq2_xxs *)(src0_gate + i02 * args.nb02 + (uint64_t)first_row * args.nb01);
    device const block_iq2_xxs *xu =
        (device const block_iq2_xxs *)(src0_up + i02 * args.nb02 + (uint64_t)first_row * args.nb01);
    device const float *y =
        (device const float *)(src1 + i11 * args.nb11 + i12 * args.nb12);

    float yl[32];
    float sumg[N_R0_IQ2_XXS] = {0.f};
    float sumu[N_R0_IQ2_XXS] = {0.f};

    threadgroup uint64_t *svalues = (threadgroup uint64_t *)(shmem);
    threadgroup uint8_t  *ssigns  = (threadgroup uint8_t *)(svalues + 256);
    {
        int nval = 4;
        int pos = (32 * sgitg + tiisg) * nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = ds4_metal_iq2xxs_grid[pos + i];
        nval = 2;
        pos = (32 * sgitg + tiisg) * nval;
        for (int i = 0; i < nval; ++i) ssigns[pos + i] = ds4_metal_ksigns_iq2xs[pos + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;
    device const float *y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_xxs *xgr = xg + ibl;
        device const block_iq2_xxs *xur = xu + ibl;
        device const uint16_t *qg = xgr->qs + 4 * ib;
        device const uint16_t *qu = xur->qs + 4 * ib;
        device const half *dhg = &xgr->d;
        device const half *dhu = &xur->d;

        for (short row = 0; row < N_R0_IQ2_XXS; row++) {
            device const uint8_t *aux8g = (device const uint8_t *)qg;
            device const uint8_t *aux8u = (device const uint8_t *)qu;
            const uint32_t aux32g = qg[2] | (qg[3] << 16);
            const uint32_t aux32u = qu[2] | (qu[3] << 16);
            const float dg = (float)dhg[0] * (0.5f + (aux32g >> 28));
            const float du = (float)dhu[0] * (0.5f + (aux32u >> 28));

            float sg = 0;
            float su = 0;
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint8_t *gridg = (const threadgroup uint8_t *)(svalues + aux8g[l]);
                const threadgroup uint8_t *gridu = (const threadgroup uint8_t *)(svalues + aux8u[l]);
                const uint8_t signg = ssigns[(aux32g >> 7 * l) & 127];
                const uint8_t signu = ssigns[(aux32u >> 7 * l) & 127];
                for (short j = 0; j < 8; ++j) {
                    const float v = yl[8 * l + j];
                    sg += v * gridg[j] * (signg & ds4_metal_kmask_iq2xs[j] ? -1.f : 1.f);
                    su += v * gridu[j] * (signu & ds4_metal_kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumg[row] += dg * sg;
            sumu[row] += du * su;

            dhg += args.nb01 / 2;
            dhu += args.nb01 / 2;
            qg  += args.nb01 / 2;
            qu  += args.nb01 / 2;
        }

        y4 += 32 * 32;
    }

    device float *dst_gate_f32 =
        (device float *)dst_gate + (uint64_t)i12 * args.ne0 * args.ne1 + (uint64_t)i11 * args.ne0;
    device float *dst_up_f32 =
        (device float *)dst_up + (uint64_t)i12 * args.ne0 * args.ne1 + (uint64_t)i11 * args.ne0;
    const uint64_t pair_row = (uint64_t)i12 * (uint64_t)args.nei0 + (uint64_t)idx;
    device float *dst_mid_f32 =
        (device float *)(dst_mid + pair_row * act.mid_row_stride);
    device const float *route_w =
        (device const float *)(weights + pair_row * act.weight_stride);

    const float c = act.clamp_value;
    const float route_weight = route_w[0];
    for (int row = 0; row < N_R0_IQ2_XXS && first_row + row < args.ne0; ++row) {
        const float sum_gate = simd_sum(sumg[row]);
        const float sum_up   = simd_sum(sumu[row]);
        if (tiisg == 0) {
            const uint out_row = first_row + row;
            const float gate = sum_gate * 0.25f;
            const float up = sum_up * 0.25f;
            float g = gate;
            float u = up;
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            dst_gate_f32[out_row] = gate;
            dst_up_f32[out_row] = up;
            const float silu = g / (1.0f + exp(-g));
            dst_mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

kernel void kernel_mul_mv_slots6_iq2_xxs_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const char * src0_gate0,
        device const char * src0_gate1,
        device const char * src0_gate2,
        device const char * src0_gate3,
        device const char * src0_gate4,
        device const char * src0_gate5,
        device const char * src0_up0,
        device const char * src0_up1,
        device const char * src0_up2,
        device const char * src0_up3,
        device const char * src0_up4,
        device const char * src0_up5,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char *src0_gate_cur = src0_gate0;
    device const char *src0_up_cur = src0_up0;
    switch (idx) {
    case 1: src0_gate_cur = src0_gate1; src0_up_cur = src0_up1; break;
    case 2: src0_gate_cur = src0_gate2; src0_up_cur = src0_up2; break;
    case 3: src0_gate_cur = src0_gate3; src0_up_cur = src0_up3; break;
    case 4: src0_gate_cur = src0_gate4; src0_up_cur = src0_up4; break;
    case 5: src0_gate_cur = src0_gate5; src0_up_cur = src0_up5; break;
    default: break;
    }

    device const char *src1_cur = src1 + i11 * args.nb11 + i12 * args.nb12;

    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    kernel_mul_mv_iq2_xxs_pair_f32_impl<N_R0_IQ2_XXS>(
        args0,
        src0_gate_cur,
        src0_up_cur,
        src1_cur,
        dst_gate_cur,
        dst_up_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);

    const short NSG = FC_mul_mv_nsg;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_IQ2_XXS;
    device float *gate_f32 = (device float *)dst_gate_cur;
    device float *up_f32 = (device float *)dst_up_cur;
    const uint64_t pair_row = (uint64_t)i12 * (uint64_t)args.nei0 + (uint64_t)idx;
    device float *mid_f32 = (device float *)(dst_mid + pair_row * act.mid_row_stride);
    device const float *route_w = (device const float *)(weights + pair_row * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    if (tiisg == 0) {
        for (int row = 0; row < N_R0_IQ2_XXS && first_row + row < args.ne0; ++row) {
            const uint out_row = first_row + row;
            float g = gate_f32[out_row];
            float u = up_f32[out_row];
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

kernel void kernel_mul_mv_addr_iq2_xxs_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const uint64_t * gate_addrs,
        device const uint64_t * up_addrs,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *)(ids + iid1 * args.nbi1))[idx];
    if (i02 < 0 || i02 >= args.ne02 || i02 >= 384) {
        return;
    }
    const uint64_t gate_addr = gate_addrs[(uint)i02];
    const uint64_t up_addr = up_addrs[(uint)i02];
    if (gate_addr == 0 || up_addr == 0) {
        return;
    }

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char *src0_gate_cur =
        reinterpret_cast<device const char *>(gate_addr);
    device const char *src0_up_cur =
        reinterpret_cast<device const char *>(up_addr);
    device const char *src1_cur = src1 + i11 * args.nb11 + i12 * args.nb12;

    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    kernel_mul_mv_iq2_xxs_pair_f32_impl<N_R0_IQ2_XXS>(
        args0,
        src0_gate_cur,
        src0_up_cur,
        src1_cur,
        dst_gate_cur,
        dst_up_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);

    const short NSG = FC_mul_mv_nsg;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_IQ2_XXS;
    device float *gate_f32 = (device float *)dst_gate_cur;
    device float *up_f32 = (device float *)dst_up_cur;
    const uint64_t pair_row = (uint64_t)i12 * (uint64_t)args.nei0 + (uint64_t)idx;
    device float *mid_f32 = (device float *)(dst_mid + pair_row * act.mid_row_stride);
    device const float *route_w = (device const float *)(weights + pair_row * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    if (tiisg == 0) {
        for (int row = 0; row < N_R0_IQ2_XXS && first_row + row < args.ne0; ++row) {
            const uint out_row = first_row + row;
            float g = gate_f32[out_row];
            float u = up_f32[out_row];
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

kernel void kernel_mul_mv_addr_iq2_xxs_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const uint64_t * addrs,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    (void)tiitg;

    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *)(ids + iid1 * args.nbi1))[idx];
    if (i02 < 0 || i02 >= args.ne02 || i02 >= 384) {
        return;
    }
    const uint64_t addr = addrs[(uint)i02];
    if (addr == 0) {
        return;
    }

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char *src0_cur = reinterpret_cast<device const char *>(addr);
    device const char *src1_cur = src1 + i11 * args.nb11 + i12 * args.nb12;
    device char *dst_cur = dst + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    kernel_mul_mv_iq2_xxs_f32_impl<N_R0_IQ2_XXS>(
        args0,
        src0_cur,
        src1_cur,
        dst_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
}

kernel void kernel_mul_mv_addr_iq2_xxs_pair_swiglu_masked_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        constant ds4_metal_stream_expert_split_args & split,
        device const uint64_t * gate_addrs,
        device const uint64_t * up_addrs,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;
    if ((split.active_mask & (1u << (uint)idx)) == 0) {
        return;
    }

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *)(ids + iid1 * args.nbi1))[idx];
    if (i02 < 0 || i02 >= args.ne02 || i02 >= 384) {
        return;
    }
    const uint64_t gate_addr = gate_addrs[(uint)i02];
    const uint64_t up_addr = up_addrs[(uint)i02];
    if (gate_addr == 0 || up_addr == 0) {
        return;
    }

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char *src0_gate_cur =
        reinterpret_cast<device const char *>(gate_addr);
    device const char *src0_up_cur =
        reinterpret_cast<device const char *>(up_addr);
    device const char *src1_cur = src1 + i11 * args.nb11 + i12 * args.nb12;

    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    kernel_mul_mv_iq2_xxs_pair_f32_impl<N_R0_IQ2_XXS>(
        args0,
        src0_gate_cur,
        src0_up_cur,
        src1_cur,
        dst_gate_cur,
        dst_up_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);

    const short NSG = FC_mul_mv_nsg;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_IQ2_XXS;
    device float *gate_f32 = (device float *)dst_gate_cur;
    device float *up_f32 = (device float *)dst_up_cur;
    const uint64_t pair_row = (uint64_t)i12 * (uint64_t)args.nei0 + (uint64_t)idx;
    device float *mid_f32 = (device float *)(dst_mid + pair_row * act.mid_row_stride);
    device const float *route_w = (device const float *)(weights + pair_row * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    if (tiisg == 0) {
        for (int row = 0; row < N_R0_IQ2_XXS && first_row + row < args.ne0; ++row) {
            const uint out_row = first_row + row;
            float g = gate_f32[out_row];
            float u = up_f32[out_row];
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

kernel void kernel_stream_expert_cache_validate(
        constant ds4_metal_stream_expert_validate_args & args,
        device const char * ids,
        device const uint64_t * gate_addrs,
        device const uint64_t * up_addrs,
        device const uint64_t * down_addrs,
        device uint32_t * status,
        uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;

    uint32_t miss_mask = 0;
    uint32_t invalid_mask = 0;
    const uint32_t n_expert = min(args.n_expert, (uint32_t)6);
    device const int32_t *selected = (device const int32_t *)ids;

    status[3] = n_expert;
    for (uint32_t i = 0; i < 6; i++) {
        const int32_t expert = i < n_expert ? selected[i] : -1;
        status[4 + i] = as_type<uint32_t>(expert);
        if (i >= n_expert) continue;
        if (expert < 0 ||
            (uint32_t)expert >= args.n_total_expert ||
            (uint32_t)expert >= 384) {
            invalid_mask |= (1u << i);
            continue;
        }
        const uint32_t e = (uint32_t)expert;
        if (gate_addrs[e] == 0 || up_addrs[e] == 0 || down_addrs[e] == 0) {
            miss_mask |= (1u << i);
        }
    }

    status[0] = (miss_mask == 0 && invalid_mask == 0) ? 1u : 0u;
    status[1] = miss_mask;
    status[2] = invalid_mask;
}

kernel void kernel_mul_mv_id_q4_K_pair_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *)(ids + iid1 * args.nbi1))[idx];
    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char *src0_gate_cur = src0_gate + i02 * args.nb02;
    device const char *src0_up_cur   = src0_up   + i02 * args.nb02;
    device const char *src1_cur      = src1      + i11 * args.nb11 + i12 * args.nb12;

    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    (void)tiitg;
    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_gate_cur,
        src1_cur,
        dst_gate_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_up_cur,
        src1_cur,
        dst_up_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
}

// Same release-path fusion as the IQ2_XXS kernel above for the Q4_K expert
// variant.  The Q4 pair path reuses the existing exact matvec implementation
// for gate and up, then the same lane that wrote each row derives the routed
// SwiGLU input.  This keeps Q4 behavior aligned with the Q2 optimization while
// preserving the old pair projection arithmetic.
kernel void kernel_mul_mv_id_q4_K_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *)(ids + iid1 * args.nbi1))[idx];
    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char *src0_gate_cur = src0_gate + i02 * args.nb02;
    device const char *src0_up_cur   = src0_up   + i02 * args.nb02;
    device const char *src1_cur      = src1      + i11 * args.nb11 + i12 * args.nb12;

    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    const short NSG = FC_mul_mv_nsg;
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_Q4_K;
    device float *gate_f32 = (device float *)dst_gate_cur;
    device float *up_f32 = (device float *)dst_up_cur;
    const uint64_t pair_row = (uint64_t)i12 * (uint64_t)args.nei0 + (uint64_t)idx;
    device float *mid_f32 = (device float *)(dst_mid + pair_row * act.mid_row_stride);
    device const float *route_w = (device const float *)(weights + pair_row * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    device const block_q4_K *xg =
        (device const block_q4_K *)(src0_gate_cur + (uint64_t)first_row * args.nb01);
    device const block_q4_K *xu =
        (device const block_q4_K *)(src0_up_cur + (uint64_t)first_row * args.nb01);
    device const float *y = (device const float *)src1_cur;
    device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    float sumg[N_R0_Q4_K] = {0.f};
    float sumu[N_R0_Q4_K] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        float yl[16];
        float yh[16];
        float4 sumy = {0.f, 0.f, 0.f, 0.f};

        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
            yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
            yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
        }

        device const uint16_t *scg = (device const uint16_t *)xg[ib].scales + iq;
        device const uint16_t *qg1 = (device const uint16_t *)xg[ib].qs + 16 * iq + 4 * ir;
        device const half *dhg = &xg[ib].d;
        device const uint16_t *scu = (device const uint16_t *)xu[ib].scales + iq;
        device const uint16_t *qu1 = (device const uint16_t *)xu[ib].qs + 16 * iq + 4 * ir;
        device const half *dhu = &xu[ib].d;

        for (short row = 0; row < N_R0_Q4_K; row++) {
            sc16[0] = scg[0] & kmask1;
            sc16[1] = scg[2] & kmask1;
            sc16[2] = ((scg[4] >> 0) & kmask2) | ((scg[0] & kmask3) >> 2);
            sc16[3] = ((scg[4] >> 4) & kmask2) | ((scg[2] & kmask3) >> 2);

            device const uint16_t *qg2 = qg1 + 32;
            float4 acc1g = {0.f, 0.f, 0.f, 0.f};
            float4 acc2g = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                acc1g[0] += yl[2 * i + 0] * (qg1[i] & 0x000F);
                acc1g[1] += yl[2 * i + 1] * (qg1[i] & 0x0F00);
                acc1g[2] += yl[2 * i + 8] * (qg1[i] & 0x00F0);
                acc1g[3] += yl[2 * i + 9] * (qg1[i] & 0xF000);
                acc2g[0] += yh[2 * i + 0] * (qg2[i] & 0x000F);
                acc2g[1] += yh[2 * i + 1] * (qg2[i] & 0x0F00);
                acc2g[2] += yh[2 * i + 8] * (qg2[i] & 0x00F0);
                acc2g[3] += yh[2 * i + 9] * (qg2[i] & 0xF000);
            }

            sumg[row] += dhg[0] * ((acc1g[0] + 1.f / 256.f * acc1g[1]) * sc8[0] +
                                   (acc1g[2] + 1.f / 256.f * acc1g[3]) * sc8[1] * 1.f / 16.f +
                                   (acc2g[0] + 1.f / 256.f * acc2g[1]) * sc8[4] +
                                   (acc2g[2] + 1.f / 256.f * acc2g[3]) * sc8[5] * 1.f / 16.f) -
                         dhg[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                   sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            sc16[0] = scu[0] & kmask1;
            sc16[1] = scu[2] & kmask1;
            sc16[2] = ((scu[4] >> 0) & kmask2) | ((scu[0] & kmask3) >> 2);
            sc16[3] = ((scu[4] >> 4) & kmask2) | ((scu[2] & kmask3) >> 2);

            device const uint16_t *qu2 = qu1 + 32;
            float4 acc1u = {0.f, 0.f, 0.f, 0.f};
            float4 acc2u = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                acc1u[0] += yl[2 * i + 0] * (qu1[i] & 0x000F);
                acc1u[1] += yl[2 * i + 1] * (qu1[i] & 0x0F00);
                acc1u[2] += yl[2 * i + 8] * (qu1[i] & 0x00F0);
                acc1u[3] += yl[2 * i + 9] * (qu1[i] & 0xF000);
                acc2u[0] += yh[2 * i + 0] * (qu2[i] & 0x000F);
                acc2u[1] += yh[2 * i + 1] * (qu2[i] & 0x0F00);
                acc2u[2] += yh[2 * i + 8] * (qu2[i] & 0x00F0);
                acc2u[3] += yh[2 * i + 9] * (qu2[i] & 0xF000);
            }

            sumu[row] += dhu[0] * ((acc1u[0] + 1.f / 256.f * acc1u[1]) * sc8[0] +
                                   (acc1u[2] + 1.f / 256.f * acc1u[3]) * sc8[1] * 1.f / 16.f +
                                   (acc2u[0] + 1.f / 256.f * acc2u[1]) * sc8[4] +
                                   (acc2u[2] + 1.f / 256.f * acc2u[3]) * sc8[5] * 1.f / 16.f) -
                         dhu[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                   sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            qg1 += args.nb01 / 2;
            scg += args.nb01 / 2;
            dhg += args.nb01 / 2;
            qu1 += args.nb01 / 2;
            scu += args.nb01 / 2;
            dhu += args.nb01 / 2;
        }

        y4 += 4 * QK_K;
    }

    for (int row = 0; row < N_R0_Q4_K && first_row + row < args.ne0; ++row) {
        const float gate = simd_sum(sumg[row]);
        const float up = simd_sum(sumu[row]);
        if (tiisg == 0) {
            const uint out_row = first_row + row;
            float g = gate;
            float u = up;
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            gate_f32[out_row] = gate;
            up_f32[out_row] = up;
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

static inline void kernel_mul_mv_mlx_affine4_64_pair_swiglu_f32_impl(
        constant ds4_metal_args_mul_mv_id &args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args &act,
        device const char *src0_gate,
        device const char *src0_up,
        device const char *src1,
        device       char *dst_gate,
        device       char *dst_up,
        device       char *dst_mid,
        device const float *route_weight,
        uint3 tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    const int first_row =
        (tgpig.x * NSG + sgitg) * N_R0_MLX_AFFINE4;
    device float *gate_f32 = (device float *)dst_gate;
    device float *up_f32 = (device float *)dst_up;
    device float *mid_f32 = (device float *)dst_mid;
    device const float *y = (device const float *)src1;
    const int groups = args.ne00 / 64;
    float sumg[N_R0_MLX_AFFINE4] = {0.f};
    float sumu[N_R0_MLX_AFFINE4] = {0.f};

    for (int group = 0; group < groups; group++) {
        const uint value = (uint)group * 64u + (uint)tiisg * 2u;
        const float y0 = y[value];
        const float y1 = y[value + 1u];
        for (short row = 0; row < N_R0_MLX_AFFINE4; row++) {
            if (first_row + row >= args.ne0) break;
            device const block_mlx_affine4_64 *gate_blocks =
                (device const block_mlx_affine4_64 *)(
                    src0_gate +
                    (uint64_t)(first_row + row) * args.nb01);
            device const block_mlx_affine4_64 *up_blocks =
                (device const block_mlx_affine4_64 *)(
                    src0_up +
                    (uint64_t)(first_row + row) * args.nb01);
            device const block_mlx_affine4_64 *gate_block =
                gate_blocks + group;
            device const block_mlx_affine4_64 *up_block = up_blocks + group;
            const uchar gq = gate_block->qs[tiisg];
            const uchar uq = up_block->qs[tiisg];
            const float gs =
                as_type<float>((uint)gate_block->scale_bf16 << 16u);
            const float gb =
                as_type<float>((uint)gate_block->bias_bf16 << 16u);
            const float us =
                as_type<float>((uint)up_block->scale_bf16 << 16u);
            const float ub =
                as_type<float>((uint)up_block->bias_bf16 << 16u);
            sumg[row] +=
                (gs * (float)(gq & 0x0fu) + gb) * y0 +
                (gs * (float)(gq >> 4u) + gb) * y1;
            sumu[row] +=
                (us * (float)(uq & 0x0fu) + ub) * y0 +
                (us * (float)(uq >> 4u) + ub) * y1;
        }
    }

    const float c = act.clamp_value;
    for (short row = 0;
         row < N_R0_MLX_AFFINE4 && first_row + row < args.ne0;
         row++) {
        const float gate = simd_sum(sumg[row]);
        const float up = simd_sum(sumu[row]);
        if (tiisg == 0) {
            const uint out_row = first_row + row;
            const float g = c > 1.0e-6f ? min(gate, c) : gate;
            const float u = c > 1.0e-6f ? clamp(up, -c, c) : up;
            gate_f32[out_row] = gate;
            up_f32[out_row] = up;
            mid_f32[out_row] =
                (g / (1.0f + exp(-g))) * u * route_weight[0];
        }
    }
}

#define DS4_DEFINE_AFFINE_PAIR_ID_KERNEL(NAME, IMPL)                         \
kernel void NAME(                                                            \
        constant ds4_metal_args_mul_mv_id &args,                             \
        constant ds4_metal_dsv4_moe_swiglu_weight_args &act,                 \
        device const char *src0_gate, device const char *src0_up,             \
        device const char *src1, device char *dst_gate,                       \
        device char *dst_up, device char *dst_mid, device const char *ids,    \
        device const char *weights,                                           \
        threadgroup char *shmem [[threadgroup(0)]],                           \
        uint3 tgpig [[threadgroup_position_in_grid]],                         \
        ushort tiitg [[thread_index_in_threadgroup]],                         \
        ushort tiisg [[thread_index_in_simdgroup]],                           \
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {                    \
    const int iid1 = tgpig.z / args.nei0;                                    \
    const int idx = tgpig.z % args.nei0;                                     \
    const int32_t expert =                                                   \
        ((device const int32_t *)(ids + (uint64_t)iid1 * args.nbi1))[idx];    \
    if (expert < 0 || expert >= args.ne02) return;                            \
    const int64_t route_slot = idx % args.ne11;                              \
    const uint64_t pair_row =                                                \
        (uint64_t)iid1 * (uint64_t)args.nei0 + (uint64_t)idx;                 \
    device const char *gate =                                                \
        src0_gate + (uint64_t)expert * args.nb02;                             \
    device const char *up = src0_up + (uint64_t)expert * args.nb02;           \
    device const char *x =                                                   \
        src1 + (uint64_t)route_slot * args.nb11 +                             \
        (uint64_t)iid1 * args.nb12;                                          \
    device const float *route_weight =                                       \
        (device const float *)(weights + pair_row * act.weight_stride);       \
    tgpig.z = 0;                                                             \
    IMPL(args, act, gate, up, x,                                             \
         dst_gate + pair_row * (uint64_t)args.ne0 * sizeof(float),           \
         dst_up + pair_row * (uint64_t)args.ne0 * sizeof(float),             \
         dst_mid + pair_row * act.mid_row_stride, route_weight,              \
         tgpig, tiisg, sgitg);                                               \
    (void)shmem; (void)tiitg;                                                \
}

#define DS4_DEFINE_AFFINE_PAIR_ADDR_KERNEL(NAME, IMPL)                       \
kernel void NAME(                                                            \
        constant ds4_metal_args_mul_mv_id &args,                             \
        constant ds4_metal_dsv4_moe_swiglu_weight_args &act,                 \
        device const ulong *gate_addrs, device const ulong *up_addrs,         \
        device const char *src1, device char *dst_gate,                       \
        device char *dst_up, device char *dst_mid, device const char *ids,    \
        device const char *weights,                                           \
        threadgroup char *shmem [[threadgroup(0)]],                           \
        uint3 tgpig [[threadgroup_position_in_grid]],                         \
        ushort tiitg [[thread_index_in_threadgroup]],                         \
        ushort tiisg [[thread_index_in_simdgroup]],                           \
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {                    \
    const int iid1 = tgpig.z / args.nei0;                                    \
    const int idx = tgpig.z % args.nei0;                                     \
    const int32_t expert =                                                   \
        ((device const int32_t *)(ids + (uint64_t)iid1 * args.nbi1))[idx];    \
    if (expert < 0 || expert >= args.ne02 || expert >= 384) return;           \
    device const char *gate = reinterpret_cast<device const char *>(          \
        gate_addrs[(uint)expert]);                                           \
    device const char *up = reinterpret_cast<device const char *>(            \
        up_addrs[(uint)expert]);                                             \
    if (!gate || !up) return;                                                \
    const int64_t route_slot = idx % args.ne11;                              \
    const uint64_t pair_row =                                                \
        (uint64_t)iid1 * (uint64_t)args.nei0 + (uint64_t)idx;                 \
    device const char *x =                                                   \
        src1 + (uint64_t)route_slot * args.nb11 +                             \
        (uint64_t)iid1 * args.nb12;                                          \
    device const float *route_weight =                                       \
        (device const float *)(weights + pair_row * act.weight_stride);       \
    tgpig.z = 0;                                                             \
    IMPL(args, act, gate, up, x,                                             \
         dst_gate + pair_row * (uint64_t)args.ne0 * sizeof(float),           \
         dst_up + pair_row * (uint64_t)args.ne0 * sizeof(float),             \
         dst_mid + pair_row * act.mid_row_stride, route_weight,              \
         tgpig, tiisg, sgitg);                                               \
    (void)shmem; (void)tiitg;                                                \
}

DS4_DEFINE_AFFINE_PAIR_ID_KERNEL(
    kernel_mul_mv_id_mlx_affine4_64_pair_swiglu_f32,
    kernel_mul_mv_mlx_affine4_64_pair_swiglu_f32_impl)
DS4_DEFINE_AFFINE_PAIR_ADDR_KERNEL(
    kernel_mul_mv_addr_mlx_affine4_64_pair_swiglu_f32,
    kernel_mul_mv_mlx_affine4_64_pair_swiglu_f32_impl)

#undef DS4_DEFINE_AFFINE_PAIR_ID_KERNEL
#undef DS4_DEFINE_AFFINE_PAIR_ADDR_KERNEL

kernel void kernel_mul_mv_table_q4_K_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const ds4_metal_q4_expert_table & gate_table,
        device const ds4_metal_q4_expert_table & up_table,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *)(ids + iid1 * args.nbi1))[idx];
    if (i02 < 0 || i02 >= args.ne02 || i02 >= 384) {
        return;
    }
    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char *src0_gate_cur = gate_table.experts[(uint)i02];
    device const char *src0_up_cur   = up_table.experts[(uint)i02];
    device const char *src1_cur      = src1 + i11 * args.nb11 + i12 * args.nb12;

    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    const short NSG = FC_mul_mv_nsg;
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_Q4_K;
    device float *gate_f32 = (device float *)dst_gate_cur;
    device float *up_f32 = (device float *)dst_up_cur;
    const uint64_t pair_row = (uint64_t)i12 * (uint64_t)args.nei0 + (uint64_t)idx;
    device float *mid_f32 = (device float *)(dst_mid + pair_row * act.mid_row_stride);
    device const float *route_w = (device const float *)(weights + pair_row * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    device const block_q4_K *xg =
        (device const block_q4_K *)(src0_gate_cur + (uint64_t)first_row * args.nb01);
    device const block_q4_K *xu =
        (device const block_q4_K *)(src0_up_cur + (uint64_t)first_row * args.nb01);
    device const float *y = (device const float *)src1_cur;
    device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    float sumg[N_R0_Q4_K] = {0.f};
    float sumu[N_R0_Q4_K] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        float yl[16];
        float yh[16];
        float4 sumy = {0.f, 0.f, 0.f, 0.f};

        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
            yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
            yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
        }

        device const uint16_t *scg = (device const uint16_t *)xg[ib].scales + iq;
        device const uint16_t *qg1 = (device const uint16_t *)xg[ib].qs + 16 * iq + 4 * ir;
        device const half *dhg = &xg[ib].d;
        device const uint16_t *scu = (device const uint16_t *)xu[ib].scales + iq;
        device const uint16_t *qu1 = (device const uint16_t *)xu[ib].qs + 16 * iq + 4 * ir;
        device const half *dhu = &xu[ib].d;

        for (short row = 0; row < N_R0_Q4_K; row++) {
            sc16[0] = scg[0] & kmask1;
            sc16[1] = scg[2] & kmask1;
            sc16[2] = ((scg[4] >> 0) & kmask2) | ((scg[0] & kmask3) >> 2);
            sc16[3] = ((scg[4] >> 4) & kmask2) | ((scg[2] & kmask3) >> 2);

            device const uint16_t *qg2 = qg1 + 32;
            float4 acc1g = {0.f, 0.f, 0.f, 0.f};
            float4 acc2g = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                acc1g[0] += yl[2 * i + 0] * (qg1[i] & 0x000F);
                acc1g[1] += yl[2 * i + 1] * (qg1[i] & 0x0F00);
                acc1g[2] += yl[2 * i + 8] * (qg1[i] & 0x00F0);
                acc1g[3] += yl[2 * i + 9] * (qg1[i] & 0xF000);
                acc2g[0] += yh[2 * i + 0] * (qg2[i] & 0x000F);
                acc2g[1] += yh[2 * i + 1] * (qg2[i] & 0x0F00);
                acc2g[2] += yh[2 * i + 8] * (qg2[i] & 0x00F0);
                acc2g[3] += yh[2 * i + 9] * (qg2[i] & 0xF000);
            }

            sumg[row] += dhg[0] * ((acc1g[0] + 1.f / 256.f * acc1g[1]) * sc8[0] +
                                   (acc1g[2] + 1.f / 256.f * acc1g[3]) * sc8[1] * 1.f / 16.f +
                                   (acc2g[0] + 1.f / 256.f * acc2g[1]) * sc8[4] +
                                   (acc2g[2] + 1.f / 256.f * acc2g[3]) * sc8[5] * 1.f / 16.f) -
                         dhg[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                   sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            sc16[0] = scu[0] & kmask1;
            sc16[1] = scu[2] & kmask1;
            sc16[2] = ((scu[4] >> 0) & kmask2) | ((scu[0] & kmask3) >> 2);
            sc16[3] = ((scu[4] >> 4) & kmask2) | ((scu[2] & kmask3) >> 2);

            device const uint16_t *qu2 = qu1 + 32;
            float4 acc1u = {0.f, 0.f, 0.f, 0.f};
            float4 acc2u = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                acc1u[0] += yl[2 * i + 0] * (qu1[i] & 0x000F);
                acc1u[1] += yl[2 * i + 1] * (qu1[i] & 0x0F00);
                acc1u[2] += yl[2 * i + 8] * (qu1[i] & 0x00F0);
                acc1u[3] += yl[2 * i + 9] * (qu1[i] & 0xF000);
                acc2u[0] += yh[2 * i + 0] * (qu2[i] & 0x000F);
                acc2u[1] += yh[2 * i + 1] * (qu2[i] & 0x0F00);
                acc2u[2] += yh[2 * i + 8] * (qu2[i] & 0x00F0);
                acc2u[3] += yh[2 * i + 9] * (qu2[i] & 0xF000);
            }

            sumu[row] += dhu[0] * ((acc1u[0] + 1.f / 256.f * acc1u[1]) * sc8[0] +
                                   (acc1u[2] + 1.f / 256.f * acc1u[3]) * sc8[1] * 1.f / 16.f +
                                   (acc2u[0] + 1.f / 256.f * acc2u[1]) * sc8[4] +
                                   (acc2u[2] + 1.f / 256.f * acc2u[3]) * sc8[5] * 1.f / 16.f) -
                         dhu[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                   sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            qg1 += args.nb01 / 2;
            scg += args.nb01 / 2;
            dhg += args.nb01 / 2;
            qu1 += args.nb01 / 2;
            scu += args.nb01 / 2;
            dhu += args.nb01 / 2;
        }

        y4 += 4 * QK_K;
    }

    for (int row = 0; row < N_R0_Q4_K && first_row + row < args.ne0; ++row) {
        const float gate = simd_sum(sumg[row]);
        const float up = simd_sum(sumu[row]);
        if (tiisg == 0) {
            const uint out_row = first_row + row;
            float g = gate;
            float u = up;
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            gate_f32[out_row] = gate;
            up_f32[out_row] = up;
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

kernel void kernel_mul_mv_addr_q4_K_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const ulong * gate_addrs,
        device const ulong * up_addrs,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *)(ids + iid1 * args.nbi1))[idx];
    if (i02 < 0 || i02 >= args.ne02 || i02 >= 384) {
        return;
    }
    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char *src0_gate_cur =
        reinterpret_cast<device const char *>(gate_addrs[(uint)i02]);
    device const char *src0_up_cur =
        reinterpret_cast<device const char *>(up_addrs[(uint)i02]);
    device const char *src1_cur = src1 + i11 * args.nb11 + i12 * args.nb12;

    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_gate_cur,
        src1_cur,
        dst_gate_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_up_cur,
        src1_cur,
        dst_up_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);

    const short NSG = FC_mul_mv_nsg;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_Q4_K;
    device float *gate_f32 = (device float *)dst_gate_cur;
    device float *up_f32 = (device float *)dst_up_cur;
    const uint64_t pair_row = (uint64_t)i12 * (uint64_t)args.nei0 + (uint64_t)idx;
    device float *mid_f32 = (device float *)(dst_mid + pair_row * act.mid_row_stride);
    device const float *route_w = (device const float *)(weights + pair_row * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    if (tiisg == 0) {
        for (int row = 0; row < N_R0_Q4_K && first_row + row < args.ne0; ++row) {
            const uint out_row = first_row + row;
            float g = gate_f32[out_row];
            float u = up_f32[out_row];
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

/* Host ABI is locked by ds4_qwen_expert_group_route's compile-time size
 * assertion.  Keeping the schedule record explicit makes the important split
 * visible here: work is dispatched expert-major, destinations remain
 * token-major/slot-major. */
struct ds4_metal_qwen35_expert_group_route {
    uint32_t expert;
    uint32_t token_row;
    uint32_t route_slot;
    uint32_t canonical_index;
};

/* Four routes selecting the same expert form one Metal work tile.  The host
 * constructs these records from the stable expert-major route list, so
 * route_begin always addresses consecutive records with the same expert.
 * Keeping this ABI separate from the route itself lets Qwen retain its
 * established one-route-per-threadgroup kernel. */
struct ds4_metal_expert_group_route_tile {
    uint32_t expert;
    uint32_t route_begin;
    uint32_t route_count;
    uint32_t reserved;
};

/* DeepSeek uses the same stable expert-major schedule as Qwen, but keeps its
 * native IQ2 pair primitive. Work is reordered only while reading gate/up
 * weights; every intermediate is scattered back to canonical token/slot
 * storage so the Q2 down reduction observes exactly the baseline order. */
kernel void kernel_mul_mv_addr_iq2_xxs_pair_swiglu_grouped_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const ulong * gate_addrs,
        device const ulong * up_addrs,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const ds4_metal_qwen35_expert_group_route * routes,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const uint grouped_index = tgpig.z;
    const ds4_metal_qwen35_expert_group_route route =
        routes[grouped_index];
    if (route.expert >= (uint)args.ne02 || route.expert >= 256u ||
        route.token_row >= (uint)args.nei1 ||
        route.route_slot >= (uint)args.nei0 ||
        route.canonical_index !=
            route.token_row * (uint)args.nei0 + route.route_slot) {
        return;
    }

    const ulong gate_addr = gate_addrs[route.expert];
    const ulong up_addr = up_addrs[route.expert];
    if (gate_addr == 0 || up_addr == 0) return;

    tgpig.z = 0;
    device const char *src0_gate_cur =
        reinterpret_cast<device const char *>(gate_addr);
    device const char *src0_up_cur =
        reinterpret_cast<device const char *>(up_addr);
    device const char *src1_cur =
        src1 + (uint64_t)route.token_row * args.nb12;
    const uint64_t dst_row = (uint64_t)route.canonical_index * args.ne0;
    device char *dst_gate_cur = dst_gate + dst_row * sizeof(float);
    device char *dst_up_cur = dst_up + dst_row * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    kernel_mul_mv_iq2_xxs_pair_f32_impl<N_R0_IQ2_XXS>(
        args0,
        src0_gate_cur,
        src0_up_cur,
        src1_cur,
        dst_gate_cur,
        dst_up_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);

    const short NSG = FC_mul_mv_nsg;
    const int first_row =
        (tgpig.x * NSG + sgitg) * N_R0_IQ2_XXS;
    device float *gate_f32 = (device float *)dst_gate_cur;
    device float *up_f32 = (device float *)dst_up_cur;
    device float *mid_f32 = (device float *)(
        dst_mid + (uint64_t)route.canonical_index * act.mid_row_stride);
    device const float *route_w = (device const float *)(
        weights + (uint64_t)route.canonical_index * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    if (tiisg == 0) {
        for (int row = 0;
             row < N_R0_IQ2_XXS && first_row + row < args.ne0;
             ++row) {
            const uint out_row = first_row + row;
            float g = gate_f32[out_row];
            float u = up_f32[out_row];
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

/* DeepSeek Flash's ordinary grouped kernel only changes dispatch order: every
 * route still fetches the same expert weights in an independent threadgroup.
 * This tiled implementation makes the reuse explicit.  Four simdgroups
 * consume up to four RHS rows for one expert while cooperatively staging all
 * compressed IQ2 blocks for row_tile output rows.  Staging compressed blocks
 * (rather than pre-dequantized floats) is deliberate: every route retains the
 * baseline lane-local accumulation and simd_sum order, which keeps the result
 * bit identical while cutting physical gate/up weight traffic.
 *
 * Flash geometry is 4096 input columns and 2048 output rows.  The row4
 * specialization uses 8448 bytes of compressed weights plus 2176 bytes of IQ2
 * lookup tables (10624 total).  Row8 uses 16896 + 2176 = 19072 bytes.  Both
 * specializations stage the complete tile and therefore need only one
 * threadgroup barrier. */
template<int route_tile, int row_tile>
void kernel_mul_mv_addr_iq2_xxs_pair_swiglu_tiled_f32_impl(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const ulong * gate_addrs,
        device const ulong * up_addrs,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const ds4_metal_qwen35_expert_group_route * routes,
        device const ds4_metal_expert_group_route_tile * tiles,
        device const char * weights,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiitg,
        ushort tiisg,
        ushort sgitg) {
    static_assert(route_tile == 2 || route_tile == 4,
                  "IQ2 expert tiles support only 2/4 routes");
    static_assert(row_tile == 4 || row_tile == 8,
                  "IQ2 expert tiles support only 4/8 output rows");
    constexpr uint threadgroup_width = (uint)route_tile * 32u;
    constexpr uint blocks_per_row = 16u;
    constexpr uint bytes_per_block = 66u;
    constexpr uint words_per_block = bytes_per_block / 2u;
    constexpr uint rows_and_projections = row_tile * 2u;
    constexpr uint compressed_words =
        rows_and_projections * blocks_per_row * words_per_block;

    const ds4_metal_expert_group_route_tile tile = tiles[tgpig.z];
    const uint total_routes = (uint)args.nei0 * (uint)args.nei1;
    if (tile.expert >= (uint)args.ne02 || tile.expert >= 256u ||
        tile.route_count == 0u || tile.route_count > (uint)route_tile ||
        tile.route_begin > total_routes ||
        tile.route_count > total_routes - tile.route_begin ||
        args.ne00 != 4096 || args.ne01 != 2048 ||
        args.ne0 != 2048 || args.nei0 != 6 ||
        args.nb01 != 16u * bytes_per_block) {
        return;
    }

    const ulong gate_addr = gate_addrs[tile.expert];
    const ulong up_addr = up_addrs[tile.expert];
    if (gate_addr == 0u || up_addr == 0u) return;

    const uint first_row = tgpig.x * row_tile;
    const bool active_route = sgitg < tile.route_count;
    ds4_metal_qwen35_expert_group_route route = {0u, 0u, 0u, 0u};
    if (active_route) {
        route = routes[tile.route_begin + (uint)sgitg];
    }
    const bool valid_route =
        active_route && route.expert == tile.expert &&
        route.token_row < (uint)args.nei1 &&
        route.route_slot < (uint)args.nei0 &&
        route.canonical_index ==
            route.token_row * (uint)args.nei0 + route.route_slot;

    device const uchar * gate_bytes =
        reinterpret_cast<device const uchar *>(gate_addr);
    device const uchar * up_bytes =
        reinterpret_cast<device const uchar *>(up_addr);
    device const float * y = valid_route ?
        (device const float *)(src1 + (uint64_t)route.token_row * args.nb12) :
        (device const float *)src1;

    float sumg[row_tile] = {0.f};
    float sumu[row_tile] = {0.f};
    float yl[32];

    threadgroup ulong * svalues = (threadgroup ulong *)shmem;
    threadgroup uchar * ssigns =
        (threadgroup uchar *)(svalues + 256u);
    threadgroup ushort * compressed =
        (threadgroup ushort *)(ssigns + 128u);

    /* Load the same lookup representation used by the baseline helper.  The
     * loader is indexed by the full threadgroup: reusing the old per-simdgroup
     * pattern with four simdgroups would overrun both lookup arrays. */
    for (uint i = (uint)tiitg; i < 256u; i += threadgroup_width) {
        svalues[i] = ds4_metal_iq2xxs_grid[i];
    }
    for (uint i = (uint)tiitg; i < 128u; i += threadgroup_width) {
        ssigns[i] = ds4_metal_ksigns_iq2xs[i];
    }

    /* Copy 16 complete IQ2 blocks for every gate/up output row.  ushort loads
     * preserve the half scale bits, are naturally aligned by the 66-byte block
     * ABI, and halve the scalar copy instruction count versus uchar staging. */
    for (uint word_index = (uint)tiitg;
         word_index < compressed_words;
         word_index += threadgroup_width) {
        const uint row_projection =
            word_index / (blocks_per_row * words_per_block);
        const uint within =
            word_index % (blocks_per_row * words_per_block);
        const uint row = row_projection % row_tile;
        const uint projection = row_projection / row_tile;
        const uint block = within / words_per_block;
        const uint word = within % words_per_block;
        const device ushort * source =
            (device const ushort *)((projection == 0u ? gate_bytes : up_bytes) +
                (uint64_t)(first_row + row) * args.nb01 +
                (uint64_t)block * bytes_per_block);
        compressed[word_index] = source[word];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Each iteration advances every lane by 32 IQ2 sub-blocks, exactly
     * matching the baseline sequence lane, lane+32, lane+64, lane+96. */
    for (uint stage = 0u; stage < 4u; ++stage) {

        if (valid_route) {
            const uint ib32 = stage * 32u + (uint)tiisg;
            device const float * y32 = y + ib32 * 32u;
            for (short i = 0; i < 32; ++i) yl[i] = y32[i];

            const uint block_in_stage = (uint)tiisg / 8u;
            const uint subblock = (uint)tiisg % 8u;
            const uint block = stage * 4u + block_in_stage;
            for (short row = 0; row < (short)row_tile; ++row) {
                const uint gate_block_index =
                    ((uint)row) * blocks_per_row + block;
                const uint up_block_index =
                    (row_tile + (uint)row) * blocks_per_row + block;
                threadgroup const block_iq2_xxs * xg =
                    (threadgroup const block_iq2_xxs *)compressed +
                    gate_block_index;
                threadgroup const block_iq2_xxs * xu =
                    (threadgroup const block_iq2_xxs *)compressed +
                    up_block_index;
                threadgroup const ushort * qg = xg->qs + 4u * subblock;
                threadgroup const ushort * qu = xu->qs + 4u * subblock;
                threadgroup const uchar * aux8g =
                    (threadgroup const uchar *)qg;
                threadgroup const uchar * aux8u =
                    (threadgroup const uchar *)qu;
                const uint aux32g = (uint)qg[2] | ((uint)qg[3] << 16);
                const uint aux32u = (uint)qu[2] | ((uint)qu[3] << 16);
                const float dg = (float)xg->d *
                    (0.5f + (float)(aux32g >> 28));
                const float du = (float)xu->d *
                    (0.5f + (float)(aux32u >> 28));

                float sg = 0.f;
                float su = 0.f;
                for (short l = 0; l < 4; ++l) {
                    threadgroup const uchar * gridg =
                        (threadgroup const uchar *)(svalues + aux8g[l]);
                    threadgroup const uchar * gridu =
                        (threadgroup const uchar *)(svalues + aux8u[l]);
                    const uchar signg =
                        ssigns[(aux32g >> (7*l)) & 127];
                    const uchar signu =
                        ssigns[(aux32u >> (7*l)) & 127];
                    for (short j = 0; j < 8; ++j) {
                        const float v = yl[8*l + j];
                        sg += v * (float)gridg[j] *
                            (signg & ds4_metal_kmask_iq2xs[j] ? -1.f : 1.f);
                        su += v * (float)gridu[j] *
                            (signu & ds4_metal_kmask_iq2xs[j] ? -1.f : 1.f);
                    }
                }
                sumg[row] += dg * sg;
                sumu[row] += du * su;
            }
        }
    }

    if (!valid_route) return;
    const uint64_t dst_row =
        (uint64_t)route.canonical_index * (uint64_t)args.ne0;
    device float * gate_out = (device float *)dst_gate + dst_row;
    device float * up_out = (device float *)dst_up + dst_row;
    device float * mid_out = (device float *)(
        dst_mid + (uint64_t)route.canonical_index * act.mid_row_stride);
    device const float * route_w = (device const float *)(
        weights + (uint64_t)route.canonical_index * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    for (short row = 0;
         row < (short)row_tile && first_row + (uint)row < (uint)args.ne0;
         ++row) {
        const float sum_gate = simd_sum(sumg[row]) * 0.25f;
        const float sum_up = simd_sum(sumu[row]) * 0.25f;
        if (tiisg == 0) {
            const uint out_row = first_row + (uint)row;
            gate_out[out_row] = sum_gate;
            up_out[out_row] = sum_up;
            float g = sum_gate;
            float u = sum_up;
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            const float silu = g / (1.0f + exp(-g));
            mid_out[out_row] = silu * u * route_weight;
        }
    }
}

/* Keep the original entry point as the conservative three-resident-tile
 * specialization.  The host dispatches ceil(2048 / 4) row groups. */
kernel void kernel_mul_mv_addr_iq2_xxs_pair_swiglu_tiled_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const ulong * gate_addrs,
        device const ulong * up_addrs,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const ds4_metal_qwen35_expert_group_route * routes,
        device const ds4_metal_expert_group_route_tile * tiles,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_addr_iq2_xxs_pair_swiglu_tiled_f32_impl<4, 4>(
        args, act, gate_addrs, up_addrs, src1, dst_gate, dst_up, dst_mid,
        routes, tiles, weights, shmem, tgpig, tiitg, tiisg, sgitg);
}

/* Two-route specialization for shallow expert buckets.  It stages the same
 * row4 compressed tile as the four-route kernel, but uses only two
 * simdgroups (64 threads).  The cooperative-copy stride follows the template
 * route count, so every lookup and compressed word is still initialized
 * before the single publishing barrier.  The host supplies route tiles with
 * at most two records and allocates the same 10624 bytes of threadgroup
 * memory as row4/route4. */
kernel void kernel_mul_mv_addr_iq2_xxs_pair_swiglu_tiled_pair_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const ulong * gate_addrs,
        device const ulong * up_addrs,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const ds4_metal_qwen35_expert_group_route * routes,
        device const ds4_metal_expert_group_route_tile * tiles,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_addr_iq2_xxs_pair_swiglu_tiled_f32_impl<2, 4>(
        args, act, gate_addrs, up_addrs, src1, dst_gate, dst_up, dst_mid,
        routes, tiles, weights, shmem, tgpig, tiitg, tiisg, sgitg);
}

/* Row8 halves the number of row-group dispatches.  Its full compressed tile
 * and lookup tables require exactly 19072 bytes of threadgroup memory; the
 * host dispatches ceil(2048 / 8) row groups. */
kernel void kernel_mul_mv_addr_iq2_xxs_pair_swiglu_tiled_r8_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const ulong * gate_addrs,
        device const ulong * up_addrs,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const ds4_metal_qwen35_expert_group_route * routes,
        device const ds4_metal_expert_group_route_tile * tiles,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_addr_iq2_xxs_pair_swiglu_tiled_f32_impl<4, 8>(
        args, act, gate_addrs, up_addrs, src1, dst_gate, dst_up, dst_mid,
        routes, tiles, weights, shmem, tgpig, tiitg, tiisg, sgitg);
}

/* Stage only 1024 input columns at a time.  The full-compressed variants
 * above minimize barriers by keeping all 16 IQ2 blocks for every row in
 * threadgroup memory.  This specialization trades three extra load/consume
 * rounds for substantially lower residency pressure:
 *
 *   256 IQ2 lookup ulongs + 128 sign bytes = 2176 bytes
 *   4 rows * gate/up * 4 IQ2 blocks * 66   = 2112 bytes
 *                                             ----------
 *                                             4288 bytes
 *
 * A simdgroup owns one route while all simdgroups cooperatively copy the
 * selected expert's compressed blocks.  Inactive simdgroups in a short tail
 * still execute both barriers.  For each lane the four stages consume slices
 * lane, lane+32, lane+64, lane+96, preserving the baseline accumulation and
 * simd_sum order exactly. */
template<int route_tile>
void kernel_mul_mv_addr_iq2_xxs_pair_swiglu_tiled_staged_f32_impl(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const ulong * gate_addrs,
        device const ulong * up_addrs,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const ds4_metal_qwen35_expert_group_route * routes,
        device const ds4_metal_expert_group_route_tile * tiles,
        device const char * weights,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiitg,
        ushort tiisg,
        ushort sgitg,
        ushort3 threads) {
    static_assert(route_tile == 2 || route_tile == 4,
                  "staged IQ2 expert tiles support only 2/4 routes");
    constexpr uint row_tile = 4u;
    constexpr uint threadgroup_width = (uint)route_tile * 32u;
    constexpr uint blocks_per_stage = 4u;
    constexpr uint bytes_per_block = 66u;
    constexpr uint words_per_block = bytes_per_block / 2u;
    constexpr uint rows_and_projections = row_tile * 2u;
    constexpr uint compressed_words =
        rows_and_projections * blocks_per_stage * words_per_block;

    const ds4_metal_expert_group_route_tile tile = tiles[tgpig.z];
    const uint total_routes = (uint)args.nei0 * (uint)args.nei1;
    const uint first_row = tgpig.x * row_tile;
    const uint actual_thread_count =
        (uint)threads.x * (uint)threads.y * (uint)threads.z;
    if (actual_thread_count != threadgroup_width ||
        tile.expert >= (uint)args.ne02 || tile.expert >= 256u ||
        tile.route_count == 0u || tile.route_count > (uint)route_tile ||
        tile.route_begin > total_routes ||
        tile.route_count > total_routes - tile.route_begin ||
        args.ne00 != 4096 || args.ne01 != 2048 ||
        args.ne0 != 2048 || args.nei0 != 6 ||
        args.nb01 != 16u * bytes_per_block ||
        first_row >= (uint)args.ne0) {
        return;
    }

    const ulong gate_addr = gate_addrs[tile.expert];
    const ulong up_addr = up_addrs[tile.expert];
    if (gate_addr == 0u || up_addr == 0u) return;

    const bool active_route = (uint)sgitg < tile.route_count;
    ds4_metal_qwen35_expert_group_route route = {0u, 0u, 0u, 0u};
    if (active_route) {
        route = routes[tile.route_begin + (uint)sgitg];
    }
    const bool valid_route =
        active_route && route.expert == tile.expert &&
        route.token_row < (uint)args.nei1 &&
        route.route_slot < (uint)args.nei0 &&
        route.canonical_index ==
            route.token_row * (uint)args.nei0 + route.route_slot;

    device const uchar * gate_bytes =
        reinterpret_cast<device const uchar *>(gate_addr);
    device const uchar * up_bytes =
        reinterpret_cast<device const uchar *>(up_addr);
    device const float * y = valid_route ?
        (device const float *)(src1 +
            (uint64_t)route.token_row * args.nb12) :
        (device const float *)src1;

    float sumg[row_tile] = {0.f};
    float sumu[row_tile] = {0.f};
    float yl[32];

    threadgroup ulong * svalues = (threadgroup ulong *)shmem;
    threadgroup uchar * ssigns =
        (threadgroup uchar *)(svalues + 256u);
    threadgroup ushort * compressed =
        (threadgroup ushort *)(ssigns + 128u);

    for (uint stage = 0u; stage < 4u; ++stage) {
        /* The tables are published by the same first-stage barrier as the
         * first compressed slab; no separate setup synchronization is
         * needed. */
        if (stage == 0u) {
            for (uint i = (uint)tiitg;
                 i < 256u;
                 i += threadgroup_width) {
                svalues[i] = ds4_metal_iq2xxs_grid[i];
            }
            for (uint i = (uint)tiitg;
                 i < 128u;
                 i += threadgroup_width) {
                ssigns[i] = ds4_metal_ksigns_iq2xs[i];
            }
        }

        for (uint word_index = (uint)tiitg;
             word_index < compressed_words;
             word_index += threadgroup_width) {
            const uint row_projection =
                word_index / (blocks_per_stage * words_per_block);
            const uint within =
                word_index % (blocks_per_stage * words_per_block);
            const uint row = row_projection % row_tile;
            const uint projection = row_projection / row_tile;
            const uint block_in_stage = within / words_per_block;
            const uint word = within % words_per_block;
            const uint block = stage * blocks_per_stage + block_in_stage;
            const device ushort * source =
                (device const ushort *)(
                    (projection == 0u ? gate_bytes : up_bytes) +
                    (uint64_t)(first_row + row) * args.nb01 +
                    (uint64_t)block * bytes_per_block);
            compressed[word_index] = source[word];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid_route) {
            const uint ib32 = stage * 32u + (uint)tiisg;
            device const float * y32 = y + ib32 * 32u;
            for (short i = 0; i < 32; ++i) yl[i] = y32[i];

            const uint block_in_stage = (uint)tiisg / 8u;
            const uint subblock = (uint)tiisg % 8u;
            for (short row = 0; row < (short)row_tile; ++row) {
                const uint gate_block_index =
                    (uint)row * blocks_per_stage + block_in_stage;
                const uint up_block_index =
                    (row_tile + (uint)row) * blocks_per_stage +
                    block_in_stage;
                threadgroup const block_iq2_xxs * xg =
                    (threadgroup const block_iq2_xxs *)compressed +
                    gate_block_index;
                threadgroup const block_iq2_xxs * xu =
                    (threadgroup const block_iq2_xxs *)compressed +
                    up_block_index;
                threadgroup const ushort * qg =
                    xg->qs + 4u * subblock;
                threadgroup const ushort * qu =
                    xu->qs + 4u * subblock;
                threadgroup const uchar * aux8g =
                    (threadgroup const uchar *)qg;
                threadgroup const uchar * aux8u =
                    (threadgroup const uchar *)qu;
                const uint aux32g =
                    (uint)qg[2] | ((uint)qg[3] << 16);
                const uint aux32u =
                    (uint)qu[2] | ((uint)qu[3] << 16);
                const float dg = (float)xg->d *
                    (0.5f + (float)(aux32g >> 28));
                const float du = (float)xu->d *
                    (0.5f + (float)(aux32u >> 28));

                float sg = 0.f;
                float su = 0.f;
                for (short l = 0; l < 4; ++l) {
                    threadgroup const uchar * gridg =
                        (threadgroup const uchar *)(svalues + aux8g[l]);
                    threadgroup const uchar * gridu =
                        (threadgroup const uchar *)(svalues + aux8u[l]);
                    const uchar signg =
                        ssigns[(aux32g >> (7*l)) & 127];
                    const uchar signu =
                        ssigns[(aux32u >> (7*l)) & 127];
                    for (short j = 0; j < 8; ++j) {
                        const float v = yl[8*l + j];
                        sg += v * (float)gridg[j] *
                            (signg & ds4_metal_kmask_iq2xs[j] ? -1.f : 1.f);
                        su += v * (float)gridu[j] *
                            (signu & ds4_metal_kmask_iq2xs[j] ? -1.f : 1.f);
                    }
                }
                sumg[row] += dg * sg;
                sumu[row] += du * su;
            }
        }

        /* This barrier is deliberately unconditional.  It protects the
         * single compressed slab from the next stage's cooperative overwrite
         * even when only one route in the tile is active. */
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (!valid_route) return;
    const uint64_t dst_row =
        (uint64_t)route.canonical_index * (uint64_t)args.ne0;
    device float * gate_out = (device float *)dst_gate + dst_row;
    device float * up_out = (device float *)dst_up + dst_row;
    device float * mid_out = (device float *)(
        dst_mid + (uint64_t)route.canonical_index * act.mid_row_stride);
    device const float * route_w = (device const float *)(
        weights + (uint64_t)route.canonical_index * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    for (short row = 0;
         row < (short)row_tile &&
             first_row + (uint)row < (uint)args.ne0;
         ++row) {
        const float sum_gate = simd_sum(sumg[row]) * 0.25f;
        const float sum_up = simd_sum(sumu[row]) * 0.25f;
        if (tiisg == 0) {
            const uint out_row = first_row + (uint)row;
            gate_out[out_row] = sum_gate;
            up_out[out_row] = sum_up;
            float g = sum_gate;
            float u = sum_up;
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            const float silu = g / (1.0f + exp(-g));
            mid_out[out_row] = silu * u * route_weight;
        }
    }
}

/* Four routes, four output rows, 128 threads, 4288 bytes threadgroup. */
kernel void kernel_mul_mv_addr_iq2_xxs_pair_swiglu_tiled_staged_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const ulong * gate_addrs,
        device const ulong * up_addrs,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const ds4_metal_qwen35_expert_group_route * routes,
        device const ds4_metal_expert_group_route_tile * tiles,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort3 threads[[threads_per_threadgroup]]) {
    kernel_mul_mv_addr_iq2_xxs_pair_swiglu_tiled_staged_f32_impl<4>(
        args, act, gate_addrs, up_addrs, src1, dst_gate, dst_up, dst_mid,
        routes, tiles, weights, shmem, tgpig, tiitg, tiisg, sgitg, threads);
}

/* Two routes, four output rows, 64 threads, 4288 bytes threadgroup. */
kernel void kernel_mul_mv_addr_iq2_xxs_pair_swiglu_tiled_pair_staged_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const ulong * gate_addrs,
        device const ulong * up_addrs,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const ds4_metal_qwen35_expert_group_route * routes,
        device const ds4_metal_expert_group_route_tile * tiles,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort3 threads[[threads_per_threadgroup]]) {
    kernel_mul_mv_addr_iq2_xxs_pair_swiglu_tiled_staged_f32_impl<2>(
        args, act, gate_addrs, up_addrs, src1, dst_gate, dst_up, dst_mid,
        routes, tiles, weights, shmem, tgpig, tiitg, tiisg, sgitg, threads);
}

/* Decode-shared DeepSeek IQ2 tile.  The compressed-tile variant above avoids
 * rereading an expert's bytes for every route, but still decodes those bytes
 * independently in all four simdgroups.  This variant moves that second kind
 * of reuse into threadgroup memory as well: for each 1024-column stage, the
 * whole threadgroup decodes gate/up for four output rows exactly once, then
 * four simdgroups consume the same magnitudes, sign masks and F32 scales for
 * their independent RHS rows.
 *
 * The 10240-byte layout is intentionally explicit and contains no F16 decoded
 * values:
 *
 *   8 planes * 32 slices * 32 magnitudes = 8192 uchar
 *   8 planes * 32 slices *  4 sign masks = 1024 uchar
 *   8 planes * 32 slices *  1 scale      = 1024 float bytes
 *
 * A barrier after decode publishes a complete stage; the second barrier keeps
 * every simdgroup out of the next overwrite until all active routes have
 * consumed it.  Inactive tail simdgroups participate in both barriers. */
kernel void kernel_mul_mv_addr_iq2_xxs_pair_swiglu_tiled_decode_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const ulong * gate_addrs,
        device const ulong * up_addrs,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const ds4_metal_qwen35_expert_group_route * routes,
        device const ds4_metal_expert_group_route_tile * tiles,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort3 threads[[threads_per_threadgroup]]) {
    constexpr uint route_tile = 4u;
    constexpr uint row_tile = 4u;
    constexpr uint slices_per_stage = 32u;
    constexpr uint values_per_slice = 32u;
    constexpr uint signs_per_slice = 4u;
    constexpr uint plane_count = 2u * row_tile;
    constexpr uint magnitude_count =
        plane_count * slices_per_stage * values_per_slice;
    constexpr uint sign_count =
        plane_count * slices_per_stage * signs_per_slice;

    const ds4_metal_expert_group_route_tile tile = tiles[tgpig.z];
    const uint total_routes = (uint)args.nei0 * (uint)args.nei1;
    const uint first_row = tgpig.x * row_tile;
    const uint thread_count =
        (uint)threads.x * (uint)threads.y * (uint)threads.z;
    if (thread_count != 128u ||
        tile.expert >= (uint)args.ne02 || tile.expert >= 256u ||
        tile.route_count == 0u || tile.route_count > route_tile ||
        tile.route_begin > total_routes ||
        tile.route_count > total_routes - tile.route_begin ||
        args.ne00 != 4096 || args.ne01 != 2048 ||
        args.ne0 != 2048 || args.nei0 != 6 ||
        args.nb01 != 16u * (uint)sizeof(block_iq2_xxs) ||
        first_row >= (uint)args.ne0) {
        return;
    }

    const ulong gate_addr = gate_addrs[tile.expert];
    const ulong up_addr = up_addrs[tile.expert];
    if (gate_addr == 0u || up_addr == 0u) return;

    const bool active_route = (uint)sgitg < tile.route_count;
    ds4_metal_qwen35_expert_group_route route = {0u, 0u, 0u, 0u};
    if (active_route) {
        route = routes[tile.route_begin + (uint)sgitg];
    }
    const bool valid_route =
        active_route && route.expert == tile.expert &&
        route.token_row < (uint)args.nei1 &&
        route.route_slot < (uint)args.nei0 &&
        route.canonical_index ==
            route.token_row * (uint)args.nei0 + route.route_slot;

    device const uchar * gate_bytes =
        reinterpret_cast<device const uchar *>(gate_addr);
    device const uchar * up_bytes =
        reinterpret_cast<device const uchar *>(up_addr);
    device const float * y = valid_route ?
        (device const float *)(src1 +
            (uint64_t)route.token_row * args.nb12) :
        (device const float *)src1;

    threadgroup uchar * decoded_magnitudes =
        (threadgroup uchar *)shmem;
    threadgroup uchar * decoded_signs =
        decoded_magnitudes + magnitude_count;
    threadgroup float * decoded_scales =
        (threadgroup float *)(decoded_signs + sign_count);

    float sumg[row_tile] = {0.f};
    float sumu[row_tile] = {0.f};
    float yl[values_per_slice];

    /* Preserve the baseline lane order: lane, lane+32, lane+64, lane+96.
     * Dequantization order is irrelevant to the sums; consumption below is
     * still stage, row, l, j with the original simd_sum and 0.25 factor. */
    for (uint stage = 0u; stage < 4u; ++stage) {
        const uint plane_slice_count = plane_count * slices_per_stage;
        for (uint item = (uint)tiitg;
             item < plane_slice_count;
             item += 128u) {
            const uint plane = item / slices_per_stage;
            const uint slice = item % slices_per_stage;
            const uint row = plane % row_tile;
            const uint projection = plane / row_tile;
            const uint block = stage * 4u + slice / 8u;
            const uint subblock = slice % 8u;
            device const uchar * source_bytes =
                projection == 0u ? gate_bytes : up_bytes;
            device const block_iq2_xxs * source =
                (device const block_iq2_xxs *)(source_bytes +
                    (uint64_t)(first_row + row) * args.nb01) + block;
            device const ushort * q = source->qs + 4u * subblock;
            device const uchar * aux8 = (device const uchar *)q;
            const uint aux32 = (uint)q[2] | ((uint)q[3] << 16);

            const uint magnitude_base = item * values_per_slice;
            const uint sign_base = item * signs_per_slice;
            for (uint l = 0u; l < 4u; ++l) {
                const ulong packed = ds4_metal_iq2xxs_grid[aux8[l]];
                decoded_signs[sign_base + l] =
                    ds4_metal_ksigns_iq2xs[(aux32 >> (7u*l)) & 127u];
                for (uint j = 0u; j < 8u; ++j) {
                    decoded_magnitudes[magnitude_base + 8u*l + j] =
                        (uchar)((packed >> (8u*j)) & 0xffu);
                }
            }
            decoded_scales[item] = (float)source->d *
                (0.5f + (float)(aux32 >> 28));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid_route) {
            const uint slice = (uint)tiisg;
            const uint ib32 = stage * slices_per_stage + slice;
            device const float * y32 = y + ib32 * values_per_slice;
            for (uint i = 0u; i < values_per_slice; ++i) {
                yl[i] = y32[i];
            }

            for (short row = 0; row < (short)row_tile; ++row) {
                const uint gate_item = (uint)row * slices_per_stage + slice;
                const uint up_item =
                    (row_tile + (uint)row) * slices_per_stage + slice;
                const uint gate_magnitude_base =
                    gate_item * values_per_slice;
                const uint up_magnitude_base =
                    up_item * values_per_slice;
                const uint gate_sign_base = gate_item * signs_per_slice;
                const uint up_sign_base = up_item * signs_per_slice;
                float sg = 0.f;
                float su = 0.f;
                for (short l = 0; l < 4; ++l) {
                    const uchar signg =
                        decoded_signs[gate_sign_base + (uint)l];
                    const uchar signu =
                        decoded_signs[up_sign_base + (uint)l];
                    for (short j = 0; j < 8; ++j) {
                        const uint value_index = 8u*(uint)l + (uint)j;
                        const float v = yl[value_index];
                        sg += v * (float)decoded_magnitudes[
                            gate_magnitude_base + value_index] *
                            (signg & ds4_metal_kmask_iq2xs[j] ? -1.f : 1.f);
                        su += v * (float)decoded_magnitudes[
                            up_magnitude_base + value_index] *
                            (signu & ds4_metal_kmask_iq2xs[j] ? -1.f : 1.f);
                    }
                }
                sumg[row] += decoded_scales[gate_item] * sg;
                sumu[row] += decoded_scales[up_item] * su;
            }
        }

        /* Tail simdgroups must reach this barrier before the next stage
         * overwrites the single decoded buffer. */
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (!valid_route) return;
    const uint64_t dst_row =
        (uint64_t)route.canonical_index * (uint64_t)args.ne0;
    device float * gate_out = (device float *)dst_gate + dst_row;
    device float * up_out = (device float *)dst_up + dst_row;
    device float * mid_out = (device float *)(
        dst_mid + (uint64_t)route.canonical_index * act.mid_row_stride);
    device const float * route_w = (device const float *)(
        weights + (uint64_t)route.canonical_index * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    for (short row = 0;
         row < (short)row_tile && first_row + (uint)row < (uint)args.ne0;
         ++row) {
        const float reduced_gate = simd_sum(sumg[row]);
        const float reduced_up = simd_sum(sumu[row]);
        if (tiisg == 0) {
            const uint out_row = first_row + (uint)row;
            const float gate = reduced_gate * 0.25f;
            const float up = reduced_up * 0.25f;
            gate_out[out_row] = gate;
            up_out[out_row] = up;
            float g = gate;
            float u = up;
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            const float silu = g / (1.0f + exp(-g));
            mid_out[out_row] = silu * u * route_weight;
        }
    }
}

kernel void kernel_mul_mv_addr_q4_K_pair_swiglu_grouped_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const ulong * gate_addrs,
        device const ulong * up_addrs,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const ds4_metal_qwen35_expert_group_route * routes,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const uint grouped_index = tgpig.z;
    const ds4_metal_qwen35_expert_group_route route =
        routes[grouped_index];
    if (route.expert >= (uint)args.ne02 || route.expert >= 256u ||
        route.token_row >= (uint)args.nei1 ||
        route.route_slot >= (uint)args.nei0 ||
        route.canonical_index !=
            route.token_row * (uint)args.nei0 + route.route_slot) {
        return;
    }

    tgpig.z = 0;
    device const char *src0_gate_cur =
        reinterpret_cast<device const char *>(gate_addrs[route.expert]);
    device const char *src0_up_cur =
        reinterpret_cast<device const char *>(up_addrs[route.expert]);
    device const char *src1_cur =
        src1 + (uint64_t)route.token_row * args.nb12;
    const uint64_t dst_row = (uint64_t)route.canonical_index * args.ne0;
    device char *dst_gate_cur = dst_gate + dst_row * sizeof(float);
    device char *dst_up_cur = dst_up + dst_row * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0, src0_gate_cur, src1_cur, dst_gate_cur,
        shmem, tgpig, tiisg, sgitg);
    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0, src0_up_cur, src1_cur, dst_up_cur,
        shmem, tgpig, tiisg, sgitg);

    const short NSG = FC_mul_mv_nsg;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_Q4_K;
    device float *gate_f32 = (device float *)dst_gate_cur;
    device float *up_f32 = (device float *)dst_up_cur;
    device float *mid_f32 = (device float *)(
        dst_mid + (uint64_t)route.canonical_index * act.mid_row_stride);
    device const float *route_w = (device const float *)(
        weights + (uint64_t)route.canonical_index * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    if (tiisg == 0) {
        for (int row = 0;
             row < N_R0_Q4_K && first_row + row < args.ne0;
             ++row) {
            const uint out_row = first_row + row;
            float g = gate_f32[out_row];
            float u = up_f32[out_row];
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

/* Small model-free contract kernels.  They intentionally do no arithmetic in
 * the scatter stage, so the host can demand bit identity (including signed
 * zero and arbitrary finite payloads), then check that reduction observes
 * canonical route slots 0..7 regardless of the expert-major dispatch order. */
struct ds4_metal_qwen35_expert_group_test_args {
    uint32_t route_count;
    uint32_t n_tokens;
    uint32_t routes_per_token;
    uint32_t width;
};

kernel void kernel_qwen35_expert_group_scatter_test_f32(
        constant ds4_metal_qwen35_expert_group_test_args & args,
        device const ds4_metal_qwen35_expert_group_route * routes,
        device const uint32_t * grouped_bits,
        device uint32_t * canonical_bits,
        uint gid [[thread_position_in_grid]]) {
    const uint grouped = gid / args.width;
    const uint column = gid - grouped * args.width;
    if (grouped >= args.route_count || column >= args.width) return;
    const ds4_metal_qwen35_expert_group_route route = routes[grouped];
    if (route.canonical_index >= args.route_count ||
        route.token_row >= args.n_tokens ||
        route.route_slot >= args.routes_per_token ||
        route.canonical_index !=
            route.token_row * args.routes_per_token + route.route_slot) {
        return;
    }
    canonical_bits[route.canonical_index * args.width + column] =
        grouped_bits[grouped * args.width + column];
}

kernel void kernel_qwen35_expert_group_reduce_test_f32(
        constant ds4_metal_qwen35_expert_group_test_args & args,
        device const float * canonical,
        device float * reduced,
        uint gid [[thread_position_in_grid]]) {
    const uint token = gid / args.width;
    const uint column = gid - token * args.width;
    if (token >= args.n_tokens || column >= args.width) return;
    float sum = 0.0f;
    /* This order is the numerical contract, not an optimization detail. */
    for (uint slot = 0; slot < args.routes_per_token; ++slot) {
        const uint canonical_index = token * args.routes_per_token + slot;
        sum += canonical[canonical_index * args.width + column];
    }
    reduced[token * args.width + column] = sum;
}

kernel void kernel_q4_gather_slots6(
        constant ds4_metal_q4_gather_slots6_args &args,
        device const char *src_group0,
        device const char *src_group1,
        device const char *src_group2,
        device const char *src_group3,
        device const char *src_group4,
        device const char *src_group5,
        device const int32_t *ids,
        device char *dst,
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint tiitg [[thread_index_in_threadgroup]]) {
    const uint slot = tgpig.y;
    if (slot >= args.n_slots || args.group_size == 0) return;

    const int32_t expert = ids[slot];
    if (expert < 0) return;

    const uint expert_u = (uint)expert;
    const uint group = expert_u / args.group_size;
    if (group >= 6) return;

    const uint local_expert = expert_u - group * args.group_size;
    device const char *src_group = src_group0;
    switch (group) {
    case 1: src_group = src_group1; break;
    case 2: src_group = src_group2; break;
    case 3: src_group = src_group3; break;
    case 4: src_group = src_group4; break;
    case 5: src_group = src_group5; break;
    default: break;
    }

    const uint64_t chunk = (uint64_t)tgpig.x * 256ul + (uint64_t)tiitg;
    const uint64_t n_chunks = args.expert_bytes >> 4;
    if (chunk >= n_chunks) return;

    device const uint4 *src =
        (device const uint4 *)(src_group + (uint64_t)local_expert * args.expert_bytes);
    device uint4 *out =
        (device uint4 *)(dst + (uint64_t)slot * args.expert_bytes);
    out[chunk] = src[chunk];
}

kernel void kernel_mul_mv_slots6_q4_K_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const char * src0_gate0,
        device const char * src0_gate1,
        device const char * src0_gate2,
        device const char * src0_gate3,
        device const char * src0_gate4,
        device const char * src0_gate5,
        device const char * src0_up0,
        device const char * src0_up1,
        device const char * src0_up2,
        device const char * src0_up3,
        device const char * src0_up4,
        device const char * src0_up5,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char *src0_gate_cur = src0_gate0;
    device const char *src0_up_cur = src0_up0;
    switch (idx) {
    case 1: src0_gate_cur = src0_gate1; src0_up_cur = src0_up1; break;
    case 2: src0_gate_cur = src0_gate2; src0_up_cur = src0_up2; break;
    case 3: src0_gate_cur = src0_gate3; src0_up_cur = src0_up3; break;
    case 4: src0_gate_cur = src0_gate4; src0_up_cur = src0_up4; break;
    case 5: src0_gate_cur = src0_gate5; src0_up_cur = src0_up5; break;
    default: break;
    }

    device const char *src1_cur = src1 + i11 * args.nb11 + i12 * args.nb12;

    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_gate_cur,
        src1_cur,
        dst_gate_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_up_cur,
        src1_cur,
        dst_up_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);

    const short NSG = FC_mul_mv_nsg;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_Q4_K;
    device float *gate_f32 = (device float *)dst_gate_cur;
    device float *up_f32 = (device float *)dst_up_cur;
    const uint64_t pair_row = (uint64_t)i12 * (uint64_t)args.nei0 + (uint64_t)idx;
    device float *mid_f32 = (device float *)(dst_mid + pair_row * act.mid_row_stride);
    device const float *route_w = (device const float *)(weights + pair_row * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    if (tiisg == 0) {
        for (int row = 0; row < N_R0_Q4_K && first_row + row < args.ne0; ++row) {
            const uint out_row = first_row + row;
            float g = gate_f32[out_row];
            float u = up_f32[out_row];
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

static inline device const char *ds4_q4_group24_select(
        uint32_t group_id,
        device const char *src00,
        device const char *src01,
        device const char *src02,
        device const char *src03,
        device const char *src04,
        device const char *src05,
        device const char *src06,
        device const char *src07,
        device const char *src08,
        device const char *src09,
        device const char *src10,
        device const char *src11,
        device const char *src12,
        device const char *src13,
        device const char *src14,
        device const char *src15,
        device const char *src16,
        device const char *src17,
        device const char *src18,
        device const char *src19,
        device const char *src20,
        device const char *src21,
        device const char *src22,
        device const char *src23) {
    switch (group_id) {
    case 1:  return src01;
    case 2:  return src02;
    case 3:  return src03;
    case 4:  return src04;
    case 5:  return src05;
    case 6:  return src06;
    case 7:  return src07;
    case 8:  return src08;
    case 9:  return src09;
    case 10: return src10;
    case 11: return src11;
    case 12: return src12;
    case 13: return src13;
    case 14: return src14;
    case 15: return src15;
    case 16: return src16;
    case 17: return src17;
    case 18: return src18;
    case 19: return src19;
    case 20: return src20;
    case 21: return src21;
    case 22: return src22;
    case 23: return src23;
    default: return src00;
    }
}

kernel void kernel_mul_mv_group6_q4_K_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const char * src0_gate0,
        device const char * src0_gate1,
        device const char * src0_gate2,
        device const char * src0_gate3,
        device const char * src0_gate4,
        device const char * src0_gate5,
        device const char * src0_up0,
        device const char * src0_up1,
        device const char * src0_up2,
        device const char * src0_up3,
        device const char * src0_up4,
        device const char * src0_up5,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr uint32_t expert_group_size = 64;
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t expert = ((device const int32_t *)(ids + iid1 * args.nbi1))[idx];
    if (expert < 0) {
        return;
    }
    const uint32_t expert_u = (uint32_t)expert;
    const uint32_t group_id = expert_u / expert_group_size;
    if (group_id >= 6) {
        return;
    }
    const uint32_t expert_local = expert_u - group_id * expert_group_size;

    device const char *src0_gate_cur = src0_gate0;
    device const char *src0_up_cur = src0_up0;
    switch (group_id) {
    case 1: src0_gate_cur = src0_gate1; src0_up_cur = src0_up1; break;
    case 2: src0_gate_cur = src0_gate2; src0_up_cur = src0_up2; break;
    case 3: src0_gate_cur = src0_gate3; src0_up_cur = src0_up3; break;
    case 4: src0_gate_cur = src0_gate4; src0_up_cur = src0_up4; break;
    case 5: src0_gate_cur = src0_gate5; src0_up_cur = src0_up5; break;
    default: break;
    }

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    src0_gate_cur += (uint64_t)expert_local * args.nb02;
    src0_up_cur   += (uint64_t)expert_local * args.nb02;
    device const char *src1_cur = src1 + i11 * args.nb11 + i12 * args.nb12;

    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_gate_cur,
        src1_cur,
        dst_gate_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_up_cur,
        src1_cur,
        dst_up_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);

    const short NSG = FC_mul_mv_nsg;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_Q4_K;
    device float *gate_f32 = (device float *)dst_gate_cur;
    device float *up_f32 = (device float *)dst_up_cur;
    const uint64_t pair_row = (uint64_t)i12 * (uint64_t)args.nei0 + (uint64_t)idx;
    device float *mid_f32 = (device float *)(dst_mid + pair_row * act.mid_row_stride);
    device const float *route_w = (device const float *)(weights + pair_row * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    if (tiisg == 0) {
        for (int row = 0; row < N_R0_Q4_K && first_row + row < args.ne0; ++row) {
            const uint out_row = first_row + row;
            float g = gate_f32[out_row];
            float u = up_f32[out_row];
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

kernel void kernel_mul_mv_group8_q4_K_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const char * src0_gate0,
        device const char * src0_gate1,
        device const char * src0_gate2,
        device const char * src0_gate3,
        device const char * src0_gate4,
        device const char * src0_gate5,
        device const char * src0_gate6,
        device const char * src0_gate7,
        device const char * src0_up0,
        device const char * src0_up1,
        device const char * src0_up2,
        device const char * src0_up3,
        device const char * src0_up4,
        device const char * src0_up5,
        device const char * src0_up6,
        device const char * src0_up7,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr uint32_t expert_group_size = 48;
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t expert = ((device const int32_t *)(ids + iid1 * args.nbi1))[idx];
    if (expert < 0) {
        return;
    }
    const uint32_t expert_u = (uint32_t)expert;
    const uint32_t group_id = expert_u / expert_group_size;
    if (group_id >= 8) {
        return;
    }
    const uint32_t expert_local = expert_u - group_id * expert_group_size;

    device const char *src0_gate_cur = src0_gate0;
    device const char *src0_up_cur = src0_up0;
    switch (group_id) {
    case 1: src0_gate_cur = src0_gate1; src0_up_cur = src0_up1; break;
    case 2: src0_gate_cur = src0_gate2; src0_up_cur = src0_up2; break;
    case 3: src0_gate_cur = src0_gate3; src0_up_cur = src0_up3; break;
    case 4: src0_gate_cur = src0_gate4; src0_up_cur = src0_up4; break;
    case 5: src0_gate_cur = src0_gate5; src0_up_cur = src0_up5; break;
    case 6: src0_gate_cur = src0_gate6; src0_up_cur = src0_up6; break;
    case 7: src0_gate_cur = src0_gate7; src0_up_cur = src0_up7; break;
    default: break;
    }

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    src0_gate_cur += (uint64_t)expert_local * args.nb02;
    src0_up_cur   += (uint64_t)expert_local * args.nb02;
    device const char *src1_cur = src1 + i11 * args.nb11 + i12 * args.nb12;

    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_gate_cur,
        src1_cur,
        dst_gate_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_up_cur,
        src1_cur,
        dst_up_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);

    const short NSG = FC_mul_mv_nsg;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_Q4_K;
    device float *gate_f32 = (device float *)dst_gate_cur;
    device float *up_f32 = (device float *)dst_up_cur;
    const uint64_t pair_row = (uint64_t)i12 * (uint64_t)args.nei0 + (uint64_t)idx;
    device float *mid_f32 = (device float *)(dst_mid + pair_row * act.mid_row_stride);
    device const float *route_w = (device const float *)(weights + pair_row * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    if (tiisg == 0) {
        for (int row = 0; row < N_R0_Q4_K && first_row + row < args.ne0; ++row) {
            const uint out_row = first_row + row;
            float g = gate_f32[out_row];
            float u = up_f32[out_row];
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

kernel void kernel_mul_mv_group24_q4_K_id_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src00,
        device const char * src01,
        device const char * src02,
        device const char * src03,
        device const char * src04,
        device const char * src05,
        device const char * src06,
        device const char * src07,
        device const char * src08,
        device const char * src09,
        device const char * src10,
        device const char * src11,
        device const char * src12,
        device const char * src13,
        device const char * src14,
        device const char * src15,
        device const char * src16,
        device const char * src17,
        device const char * src18,
        device const char * src19,
        device const char * src20,
        device const char * src21,
        device const char * src22,
        device const char * src23,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr uint32_t expert_group_size = 16;
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t expert = ((device const int32_t *)(ids + iid1 * args.nbi1))[idx];
    if (expert < 0) {
        return;
    }
    const uint32_t expert_u = (uint32_t)expert;
    const uint32_t group_id = expert_u / expert_group_size;
    if (group_id >= 24) {
        return;
    }
    const uint32_t expert_local = expert_u - group_id * expert_group_size;

    device const char *src0_cur = ds4_q4_group24_select(group_id,
                                                        src00, src01, src02, src03,
                                                        src04, src05, src06, src07,
                                                        src08, src09, src10, src11,
                                                        src12, src13, src14, src15,
                                                        src16, src17, src18, src19,
                                                        src20, src21, src22, src23);
    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    src0_cur += (uint64_t)expert_local * args.nb02;
    device const char *src1_cur = src1 + i11 * args.nb11 + i12 * args.nb12;
    device char *dst_cur = dst + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_cur,
        src1_cur,
        dst_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);

    (void)tiitg;
}

kernel void kernel_mul_mv_group_q4_K_pair_swiglu_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        constant ds4_metal_moe_expert_group_args & group,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device       char * dst_gate,
        device       char * dst_up,
        device       char * dst_mid,
        device const char * ids,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z / args.nei0;
    const int idx  = tgpig.z % args.nei0;

    tgpig.z = 0;

    const int32_t expert_global = ((device const int32_t *)(ids + iid1 * args.nbi1))[idx];
    if (expert_global < 0) {
        return;
    }
    const uint32_t expert_u = (uint32_t)expert_global;
    if (expert_u < group.expert_base ||
        expert_u >= group.expert_base + group.expert_count) {
        return;
    }
    const uint32_t expert_local = expert_u - group.expert_base;

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char *src0_gate_cur = src0_gate + (uint64_t)expert_local * args.nb02;
    device const char *src0_up_cur   = src0_up   + (uint64_t)expert_local * args.nb02;
    device const char *src1_cur      = src1      + i11 * args.nb11 + i12 * args.nb12;

    device char *dst_gate_cur = dst_gate + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);
    device char *dst_up_cur   = dst_up   + (idx * args.ne0 + i12 * args.ne1 * args.ne0) * sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        args.ne00, args.ne01, 1,
        args.nb00, args.nb01, args.nb02, args.nb02,
        args.ne10, 1, 1,
        args.nb10, args.nb11, args.nb12, args.nb12,
        args.ne0, 1, args.nr0, 1, 1,
    };

    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_gate_cur,
        src1_cur,
        dst_gate_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K>(
        args0,
        src0_up_cur,
        src1_cur,
        dst_up_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);

    const short NSG = FC_mul_mv_nsg;
    const int first_row = (tgpig.x * NSG + sgitg) * N_R0_Q4_K;
    device float *gate_f32 = (device float *)dst_gate_cur;
    device float *up_f32 = (device float *)dst_up_cur;
    const uint64_t pair_row = (uint64_t)i12 * (uint64_t)args.nei0 + (uint64_t)idx;
    device float *mid_f32 = (device float *)(dst_mid + pair_row * act.mid_row_stride);
    device const float *route_w = (device const float *)(weights + pair_row * act.weight_stride);
    const float c = act.clamp_value;
    const float route_weight = route_w[0];

    if (tiisg == 0) {
        for (int row = 0; row < N_R0_Q4_K && first_row + row < args.ne0; ++row) {
            const uint out_row = first_row + row;
            float g = gate_f32[out_row];
            float u = up_f32[out_row];
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            const float silu = g / (1.0f + exp(-g));
            mid_f32[out_row] = silu * u * route_weight;
        }
    }

    (void)tiitg;
}

kernel void kernel_mul_mv_id_q2_K_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q2_K;
    const int nb = args.ne00/QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * args.nbi1);
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    float sumf[nr0] = {0.f};

    const short ix = tiisg/8;
    const short it = tiisg%8;
    const short iq = it/4;
    const short ir = it%4;
    const short is = (8*ir)/16;

    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        const int32_t expert = token_ids[expert_slot];
        device const block_q2_K * x = (device const block_q2_K *)(src0s + expert*args.nb02 + first_row*args.nb01);
        device const float * y = (device const float *)(token_src1 + expert_slot*args.nb11);
        device const float * y4 = y + ix * QK_K + 128 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[32];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            for (short i = 0; i < 8; ++i) {
                yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
                yl[i+ 8] = y4[i+32]; sumy[1] += yl[i+ 8];
                yl[i+16] = y4[i+64]; sumy[2] += yl[i+16];
                yl[i+24] = y4[i+96]; sumy[3] += yl[i+24];
            }

            device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales + 8*iq + is;
            device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half     * dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                    for (int i = 0; i < 8; i += 2) {
                        acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);
                        acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);
                        acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);
                        acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);
                        acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);
                        acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);
                        acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);
                        acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);
                    }
                    float dall = dh[0];
                    float dmin = dh[1] * 1.f/16.f;
                    sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
                                         (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * 1.f/ 4.f +
                                         (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * 1.f/16.f +
                                         (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * 1.f/64.f) -
                                 dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                                         sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));
                }

                qs += args.nb01/2;
                sc += args.nb01;
                dh += args.nb01/2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float * dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

kernel void kernel_mul_mv_slots6_q2_K_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src00,
        device const char * src01,
        device const char * src02,
        device const char * src03,
        device const char * src04,
        device const char * src05,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q2_K;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    float sumf[nr0] = {0.f};

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const short is = (8 * ir) / 16;

    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        device const char *src0_cur = src00;
        switch (expert_slot) {
        case 1: src0_cur = src01; break;
        case 2: src0_cur = src02; break;
        case 3: src0_cur = src03; break;
        case 4: src0_cur = src04; break;
        case 5: src0_cur = src05; break;
        default: break;
        }
        device const block_q2_K *x =
            (device const block_q2_K *)(src0_cur + first_row * args.nb01);
        device const float *y = (device const float *)(token_src1 + expert_slot * args.nb11);
        device const float *y4 = y + ix * QK_K + 128 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[32];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            for (short i = 0; i < 8; ++i) {
                yl[i +  0] = y4[i +  0]; sumy[0] += yl[i +  0];
                yl[i +  8] = y4[i + 32]; sumy[1] += yl[i +  8];
                yl[i + 16] = y4[i + 64]; sumy[2] += yl[i + 16];
                yl[i + 24] = y4[i + 96]; sumy[3] += yl[i + 24];
            }

            device const uint8_t  *sc = (device const uint8_t *)x[ib].scales + 8 * iq + is;
            device const uint16_t *qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half     *dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                    for (int i = 0; i < 8; i += 2) {
                        acc1[0] += yl[i +  0] * (qs[i / 2] & 0x0003);
                        acc2[0] += yl[i +  1] * (qs[i / 2] & 0x0300);
                        acc1[1] += yl[i +  8] * (qs[i / 2] & 0x000c);
                        acc2[1] += yl[i +  9] * (qs[i / 2] & 0x0c00);
                        acc1[2] += yl[i + 16] * (qs[i / 2] & 0x0030);
                        acc2[2] += yl[i + 17] * (qs[i / 2] & 0x3000);
                        acc1[3] += yl[i + 24] * (qs[i / 2] & 0x00c0);
                        acc2[3] += yl[i + 25] * (qs[i / 2] & 0xc000);
                    }
                    float dall = dh[0];
                    float dmin = dh[1] * 1.f / 16.f;
                    sumf[row] += dall * ((acc1[0] + 1.f / 256.f * acc2[0]) * (sc[0] & 0xF) * 1.f /  1.f +
                                         (acc1[1] + 1.f / 256.f * acc2[1]) * (sc[2] & 0xF) * 1.f /  4.f +
                                         (acc1[2] + 1.f / 256.f * acc2[2]) * (sc[4] & 0xF) * 1.f / 16.f +
                                         (acc1[3] + 1.f / 256.f * acc2[3]) * (sc[6] & 0xF) * 1.f / 64.f) -
                                 dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                                         sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));
                }

                qs += args.nb01 / 2;
                sc += args.nb01;
                dh += args.nb01 / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float *dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

kernel void kernel_mul_mv_addr_q2_K_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const uint64_t * addrs,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q2_K;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;
    device const int32_t *token_ids =
        (device const int32_t *)(ids + (uint64_t)token * args.nbi1);

    float sumf[nr0] = {0.f};

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const short is = (8 * ir) / 16;

    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        const int32_t expert = token_ids[expert_slot];
        if (expert < 0 || expert >= args.ne02 || expert >= 384) {
            continue;
        }
        const uint64_t addr = addrs[(uint)expert];
        if (addr == 0) {
            continue;
        }
        device const char *src0_cur =
            reinterpret_cast<device const char *>(addr);
        device const block_q2_K *x =
            (device const block_q2_K *)(src0_cur + first_row * args.nb01);
        device const float *y = (device const float *)(token_src1 + expert_slot * args.nb11);
        device const float *y4 = y + ix * QK_K + 128 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[32];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            for (short i = 0; i < 8; ++i) {
                yl[i +  0] = y4[i +  0]; sumy[0] += yl[i +  0];
                yl[i +  8] = y4[i + 32]; sumy[1] += yl[i +  8];
                yl[i + 16] = y4[i + 64]; sumy[2] += yl[i + 16];
                yl[i + 24] = y4[i + 96]; sumy[3] += yl[i + 24];
            }

            device const uint8_t  *sc = (device const uint8_t *)x[ib].scales + 8 * iq + is;
            device const uint16_t *qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half     *dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                    for (int i = 0; i < 8; i += 2) {
                        acc1[0] += yl[i +  0] * (qs[i / 2] & 0x0003);
                        acc2[0] += yl[i +  1] * (qs[i / 2] & 0x0300);
                        acc1[1] += yl[i +  8] * (qs[i / 2] & 0x000c);
                        acc2[1] += yl[i +  9] * (qs[i / 2] & 0x0c00);
                        acc1[2] += yl[i + 16] * (qs[i / 2] & 0x0030);
                        acc2[2] += yl[i + 17] * (qs[i / 2] & 0x3000);
                        acc1[3] += yl[i + 24] * (qs[i / 2] & 0x00c0);
                        acc2[3] += yl[i + 25] * (qs[i / 2] & 0xc000);
                    }
                    float dall = dh[0];
                    float dmin = dh[1] * 1.f / 16.f;
                    sumf[row] += dall * ((acc1[0] + 1.f / 256.f * acc2[0]) * (sc[0] & 0xF) * 1.f /  1.f +
                                         (acc1[1] + 1.f / 256.f * acc2[1]) * (sc[2] & 0xF) * 1.f /  4.f +
                                         (acc1[2] + 1.f / 256.f * acc2[2]) * (sc[4] & 0xF) * 1.f / 16.f +
                                         (acc1[3] + 1.f / 256.f * acc2[3]) * (sc[6] & 0xF) * 1.f / 64.f) -
                                 dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                                         sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));
                }
                qs += args.nb01 / 2;
                sc += args.nb01;
                dh += args.nb01 / 2;
            }
            y4 += 4 * QK_K;
        }
    }

    device float * dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

kernel void kernel_mul_mv_addr_q2_K_sum6_masked_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_stream_expert_split_args & split,
        device const uint64_t * addrs,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q2_K;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;
    device const int32_t *token_ids =
        (device const int32_t *)(ids + (uint64_t)token * args.nbi1);

    float sumf[nr0] = {0.f};

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const short is = (8 * ir) / 16;

    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        if ((split.active_mask & (1u << (uint)expert_slot)) == 0) {
            continue;
        }
        const int32_t expert = token_ids[expert_slot];
        if (expert < 0 || expert >= args.ne02 || expert >= 384) {
            continue;
        }
        const uint64_t addr = addrs[(uint)expert];
        if (addr == 0) {
            continue;
        }
        device const char *src0_cur =
            reinterpret_cast<device const char *>(addr);
        device const block_q2_K *x =
            (device const block_q2_K *)(src0_cur + first_row * args.nb01);
        device const float *y = (device const float *)(token_src1 + expert_slot * args.nb11);
        device const float *y4 = y + ix * QK_K + 128 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[32];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            for (short i = 0; i < 8; ++i) {
                yl[i +  0] = y4[i +  0]; sumy[0] += yl[i +  0];
                yl[i +  8] = y4[i + 32]; sumy[1] += yl[i +  8];
                yl[i + 16] = y4[i + 64]; sumy[2] += yl[i + 16];
                yl[i + 24] = y4[i + 96]; sumy[3] += yl[i + 24];
            }

            device const uint8_t  *sc = (device const uint8_t *)x[ib].scales + 8 * iq + is;
            device const uint16_t *qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half     *dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                    for (int i = 0; i < 8; i += 2) {
                        acc1[0] += yl[i +  0] * (qs[i / 2] & 0x0003);
                        acc2[0] += yl[i +  1] * (qs[i / 2] & 0x0300);
                        acc1[1] += yl[i +  8] * (qs[i / 2] & 0x000c);
                        acc2[1] += yl[i +  9] * (qs[i / 2] & 0x0c00);
                        acc1[2] += yl[i + 16] * (qs[i / 2] & 0x0030);
                        acc2[2] += yl[i + 17] * (qs[i / 2] & 0x3000);
                        acc1[3] += yl[i + 24] * (qs[i / 2] & 0x00c0);
                        acc2[3] += yl[i + 25] * (qs[i / 2] & 0xc000);
                    }
                    float dall = dh[0];
                    float dmin = dh[1] * 1.f / 16.f;
                    sumf[row] += dall * ((acc1[0] + 1.f / 256.f * acc2[0]) * (sc[0] & 0xF) * 1.f /  1.f +
                                         (acc1[1] + 1.f / 256.f * acc2[1]) * (sc[2] & 0xF) * 1.f /  4.f +
                                         (acc1[2] + 1.f / 256.f * acc2[2]) * (sc[4] & 0xF) * 1.f / 16.f +
                                         (acc1[3] + 1.f / 256.f * acc2[3]) * (sc[6] & 0xF) * 1.f / 64.f) -
                                 dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                                         sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));
                }
                qs += args.nb01 / 2;
                sc += args.nb01;
                dh += args.nb01 / 2;
            }
            y4 += 4 * QK_K;
        }
    }

    device float * dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            if (split.accumulate) {
                dst_f32[first_row + row] += sum_all;
            } else {
                dst_f32[first_row + row] = sum_all;
            }
        }
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

kernel void kernel_mul_mv_id_q4_K_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q4_K;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * args.nbi1);
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    float sumf[nr0] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        const int32_t expert = token_ids[expert_slot];
        device const block_q4_K *x =
            (device const block_q4_K *)(src0s + expert * args.nb02 + first_row * args.nb01);
        device const float *y = (device const float *)(token_src1 + expert_slot * args.nb11);
        device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[16];
            float yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};

            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
                yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
                yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
            }

            device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
            device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half *dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    sc16[0] = sc[0] & kmask1;
                    sc16[1] = sc[2] & kmask1;
                    sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                    sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                    device const uint16_t *q2 = q1 + 32;

                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                    FOR_UNROLL (short i = 0; i < 4; ++i) {
                        acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                        acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                        acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                        acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                        acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                        acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                        acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                        acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
                    }

                    sumf[row] += dh[0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                          (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                          (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                          (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                                 dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                          sumy[2] * sc8[6] + sumy[3] * sc8[7]);
                }

                q1 += args.nb01 / 2;
                sc += args.nb01 / 2;
                dh += args.nb01 / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float *dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

kernel void kernel_mul_mv_group_q4_K_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        constant ds4_metal_moe_expert_group_args & group,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q4_K;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * args.nbi1);
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    float sumf[nr0] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        const int32_t expert = token_ids[expert_slot];
        if (expert < 0) {
            continue;
        }
        const uint32_t expert_u = (uint32_t)expert;
        if (expert_u < group.expert_base ||
            expert_u >= group.expert_base + group.expert_count) {
            continue;
        }
        const uint32_t expert_local = expert_u - group.expert_base;

        device const block_q4_K *x =
            (device const block_q4_K *)(src0s + (uint64_t)expert_local * args.nb02 + first_row * args.nb01);
        device const float *y = (device const float *)(token_src1 + expert_slot * args.nb11);
        device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[16];
            float yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};

            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
                yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
                yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
            }

            device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
            device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half *dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    sc16[0] = sc[0] & kmask1;
                    sc16[1] = sc[2] & kmask1;
                    sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                    sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                    device const uint16_t *q2 = q1 + 32;

                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                    FOR_UNROLL (short i = 0; i < 4; ++i) {
                        acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                        acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                        acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                        acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                        acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                        acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                        acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                        acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
                    }

                    sumf[row] += dh[0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                          (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                          (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                          (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                                 dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                          sumy[2] * sc8[6] + sumy[3] * sc8[7]);
                }

                q1 += args.nb01 / 2;
                sc += args.nb01 / 2;
                dh += args.nb01 / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float *dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            if (group.accumulate) {
                dst_f32[first_row + row] += sum_all;
            } else {
                dst_f32[first_row + row] = sum_all;
            }
        }
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

kernel void kernel_mul_mv_table_q4_K_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const ds4_metal_q4_expert_table & table,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q4_K;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * args.nbi1);
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    float sumf[nr0] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        const int32_t expert = token_ids[expert_slot];
        if (expert < 0 || expert >= args.ne02 || expert >= 384) {
            return;
        }
        device const block_q4_K *x =
            (device const block_q4_K *)(table.experts[(uint)expert] + first_row * args.nb01);
        device const float *y = (device const float *)(token_src1 + expert_slot * args.nb11);
        device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[16];
            float yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};

            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
                yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
                yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
            }

            device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
            device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half *dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    sc16[0] = sc[0] & kmask1;
                    sc16[1] = sc[2] & kmask1;
                    sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                    sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                    device const uint16_t *q2 = q1 + 32;

                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                    FOR_UNROLL (short i = 0; i < 4; ++i) {
                        acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                        acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                        acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                        acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                        acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                        acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                        acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                        acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
                    }

                    sumf[row] += dh[0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                          (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                          (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                          (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                                 dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                          sumy[2] * sc8[6] + sumy[3] * sc8[7]);
                }

                q1 += args.nb01 / 2;
                sc += args.nb01 / 2;
                dh += args.nb01 / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float *dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

kernel void kernel_mul_mv_addr_q4_K_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const ulong * addrs,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q4_K;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * args.nbi1);
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    float sumf[nr0] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    // The historical kernel name says sum6, but the address-table path also
    // serves Qwen top-8 prefill. The host validates nei0 as either 6 or 8.
    for (int expert_slot = 0; expert_slot < args.nei0; expert_slot++) {
        const int32_t expert = token_ids[expert_slot];
        if (expert < 0 || expert >= args.ne02 || expert >= 384) {
            return;
        }
        device const char *expert_base =
            reinterpret_cast<device const char *>(addrs[(uint)expert]);
        device const block_q4_K *x =
            (device const block_q4_K *)(expert_base + first_row * args.nb01);
        device const float *y = (device const float *)(token_src1 + expert_slot * args.nb11);
        device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[16];
            float yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};

            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
                yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
                yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
            }

            device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
            device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half *dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    sc16[0] = sc[0] & kmask1;
                    sc16[1] = sc[2] & kmask1;
                    sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                    sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                    device const uint16_t *q2 = q1 + 32;

                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                    FOR_UNROLL (short i = 0; i < 4; ++i) {
                        acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                        acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                        acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                        acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                        acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                        acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                        acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                        acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
                    }

                    sumf[row] += dh[0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                          (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                          (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                          (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                                 dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                          sumy[2] * sc8[6] + sumy[3] * sc8[7]);
                }

                q1 += args.nb01 / 2;
                sc += args.nb01 / 2;
                dh += args.nb01 / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float *dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

kernel void kernel_mul_mv_addr_mlx_affine4_64_sum8_f32(
        constant ds4_metal_args_mul_mv_id &args,
        device const ulong *addrs,
        device const char *src1,
        device       char *dst,
        device const char *ids,
        threadgroup  char *shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_MLX_AFFINE4;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    const int groups = args.ne00 / 64;
    device const int32_t *token_ids =
        (device const int32_t *)(ids + (uint64_t)token * args.nbi1);
    device const char *token_src1 =
        src1 + (uint64_t)token * args.nb12;
    float sumf[nr0] = {0.f};

    for (int expert_slot = 0; expert_slot < args.nei0; expert_slot++) {
        const int32_t expert = token_ids[expert_slot];
        if (expert < 0 || expert >= args.ne02 || expert >= 384) return;
        device const char *expert_base =
            reinterpret_cast<device const char *>(addrs[(uint)expert]);
        if (!expert_base) return;
        device const float *y =
            (device const float *)(token_src1 +
                                   (uint64_t)expert_slot * args.nb11);
        for (int group = 0; group < groups; group++) {
            const uint value =
                (uint)group * 64u + (uint)tiisg * 2u;
            const float y0 = y[value];
            const float y1 = y[value + 1u];
            for (short row = 0; row < nr0; row++) {
                if (first_row + row >= args.ne0) break;
                device const block_mlx_affine4_64 *blocks =
                    (device const block_mlx_affine4_64 *)(
                        expert_base +
                        (uint64_t)(first_row + row) * args.nb01);
                device const block_mlx_affine4_64 *block = blocks + group;
                const uchar packed = block->qs[tiisg];
                const float scale =
                    as_type<float>((uint)block->scale_bf16 << 16u);
                const float bias =
                    as_type<float>((uint)block->bias_bf16 << 16u);
                sumf[row] +=
                    (scale * (float)(packed & 0x0fu) + bias) * y0 +
                    (scale * (float)(packed >> 4u) + bias) * y1;
            }
        }
    }

    device float *dst_f32 =
        (device float *)(dst + (uint64_t)token * args.nb1);
    for (short row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum;
    }
    (void)shmem;
    (void)tiitg;
}

kernel void kernel_mul_mv_slots6_q4_K_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src00,
        device const char * src01,
        device const char * src02,
        device const char * src03,
        device const char * src04,
        device const char * src05,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int n_expert = args.nei0;
    if (n_expert <= 0 || n_expert > 6) return;

    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q4_K;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    float sumf[nr0] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int expert_slot = 0; expert_slot < n_expert; expert_slot++) {
        device const char *src0_cur = src00;
        switch (expert_slot) {
        case 1: src0_cur = src01; break;
        case 2: src0_cur = src02; break;
        case 3: src0_cur = src03; break;
        case 4: src0_cur = src04; break;
        case 5: src0_cur = src05; break;
        default: break;
        }
        device const block_q4_K *x =
            (device const block_q4_K *)(src0_cur + first_row * args.nb01);
        device const float *y = (device const float *)(token_src1 + expert_slot * args.nb11);
        device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[16];
            float yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};

            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
                yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
                yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
            }

            device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
            device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half *dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    sc16[0] = sc[0] & kmask1;
                    sc16[1] = sc[2] & kmask1;
                    sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                    sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                    device const uint16_t *q2 = q1 + 32;

                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                    FOR_UNROLL (short i = 0; i < 4; ++i) {
                        acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                        acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                        acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                        acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                        acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                        acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                        acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                        acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
                    }

                    sumf[row] += dh[0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                          (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                          (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                          (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                                 dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                          sumy[2] * sc8[6] + sumy[3] * sc8[7]);
                }

                q1 += args.nb01 / 2;
                sc += args.nb01 / 2;
                dh += args.nb01 / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float *dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

kernel void kernel_mul_mv_group6_q4_K_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src00,
        device const char * src01,
        device const char * src02,
        device const char * src03,
        device const char * src04,
        device const char * src05,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr uint32_t expert_group_size = 64;
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q4_K;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * args.nbi1);
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    float sumf[nr0] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        const int32_t expert = token_ids[expert_slot];
        if (expert < 0) {
            continue;
        }
        const uint32_t expert_u = (uint32_t)expert;
        const uint32_t group_id = expert_u / expert_group_size;
        if (group_id >= 6) {
            continue;
        }
        const uint32_t expert_local = expert_u - group_id * expert_group_size;

        device const char *src0_cur = src00;
        switch (group_id) {
        case 1: src0_cur = src01; break;
        case 2: src0_cur = src02; break;
        case 3: src0_cur = src03; break;
        case 4: src0_cur = src04; break;
        case 5: src0_cur = src05; break;
        default: break;
        }

        device const block_q4_K *x =
            (device const block_q4_K *)(src0_cur + (uint64_t)expert_local * args.nb02 + first_row * args.nb01);
        device const float *y = (device const float *)(token_src1 + expert_slot * args.nb11);
        device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[16];
            float yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};

            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
                yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
                yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
            }

            device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
            device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half *dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    sc16[0] = sc[0] & kmask1;
                    sc16[1] = sc[2] & kmask1;
                    sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                    sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                    device const uint16_t *q2 = q1 + 32;

                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                    FOR_UNROLL (short i = 0; i < 4; ++i) {
                        acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                        acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                        acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                        acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                        acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                        acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                        acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                        acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
                    }

                    sumf[row] += dh[0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                          (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                          (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                          (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                                 dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                          sumy[2] * sc8[6] + sumy[3] * sc8[7]);
                }

                q1 += args.nb01 / 2;
                sc += args.nb01 / 2;
                dh += args.nb01 / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float *dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

kernel void kernel_mul_mv_group8_q4_K_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src00,
        device const char * src01,
        device const char * src02,
        device const char * src03,
        device const char * src04,
        device const char * src05,
        device const char * src06,
        device const char * src07,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr uint32_t expert_group_size = 48;
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q4_K;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * args.nbi1);
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    float sumf[nr0] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        const int32_t expert = token_ids[expert_slot];
        if (expert < 0) {
            continue;
        }
        const uint32_t expert_u = (uint32_t)expert;
        const uint32_t group_id = expert_u / expert_group_size;
        if (group_id >= 8) {
            continue;
        }
        const uint32_t expert_local = expert_u - group_id * expert_group_size;

        device const char *src0_cur = src00;
        switch (group_id) {
        case 1: src0_cur = src01; break;
        case 2: src0_cur = src02; break;
        case 3: src0_cur = src03; break;
        case 4: src0_cur = src04; break;
        case 5: src0_cur = src05; break;
        case 6: src0_cur = src06; break;
        case 7: src0_cur = src07; break;
        default: break;
        }

        device const block_q4_K *x =
            (device const block_q4_K *)(src0_cur + (uint64_t)expert_local * args.nb02 + first_row * args.nb01);
        device const float *y = (device const float *)(token_src1 + expert_slot * args.nb11);
        device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[16];
            float yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};

            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
                yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
                yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
            }

            device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
            device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half *dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    sc16[0] = sc[0] & kmask1;
                    sc16[1] = sc[2] & kmask1;
                    sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                    sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                    device const uint16_t *q2 = q1 + 32;

                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                    FOR_UNROLL (short i = 0; i < 4; ++i) {
                        acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                        acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                        acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                        acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                        acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                        acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                        acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                        acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
                    }

                    sumf[row] += dh[0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                          (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                          (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                          (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                                 dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                          sumy[2] * sc8[6] + sumy[3] * sc8[7]);
                }

                q1 += args.nb01 / 2;
                sc += args.nb01 / 2;
                dh += args.nb01 / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float *dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

kernel void kernel_mul_mv_group24_q4_K_sum6_f32(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src00,
        device const char * src01,
        device const char * src02,
        device const char * src03,
        device const char * src04,
        device const char * src05,
        device const char * src06,
        device const char * src07,
        device const char * src08,
        device const char * src09,
        device const char * src10,
        device const char * src11,
        device const char * src12,
        device const char * src13,
        device const char * src14,
        device const char * src15,
        device const char * src16,
        device const char * src17,
        device const char * src18,
        device const char * src19,
        device const char * src20,
        device const char * src21,
        device const char * src22,
        device const char * src23,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr uint32_t expert_group_size = 16;
    const short NSG = FC_mul_mv_nsg;
    const short nr0 = N_R0_Q4_K;
    const int nb = args.ne00 / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;
    const uint token = tgpig.y;
    device const int32_t *token_ids = (device const int32_t *)(ids + (uint64_t)token * args.nbi1);
    device const char *token_src1 = src1 + (uint64_t)token * args.nb12;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    float sumf[nr0] = {0.f};
    uint16_t sc16[4];
    thread const uint8_t *sc8 = (thread const uint8_t *)sc16;

    for (int expert_slot = 0; expert_slot < 6; expert_slot++) {
        const int32_t expert = token_ids[expert_slot];
        if (expert < 0) {
            continue;
        }
        const uint32_t expert_u = (uint32_t)expert;
        const uint32_t group_id = expert_u / expert_group_size;
        if (group_id >= 24) {
            continue;
        }
        const uint32_t expert_local = expert_u - group_id * expert_group_size;

        device const char *src0_cur = ds4_q4_group24_select(group_id,
                                                            src00, src01, src02, src03,
                                                            src04, src05, src06, src07,
                                                            src08, src09, src10, src11,
                                                            src12, src13, src14, src15,
                                                            src16, src17, src18, src19,
                                                            src20, src21, src22, src23);
        device const block_q4_K *x =
            (device const block_q4_K *)(src0_cur + (uint64_t)expert_local * args.nb02 + first_row * args.nb01);
        device const float *y = (device const float *)(token_src1 + expert_slot * args.nb11);
        device const float *y4 = y + ix * QK_K + 64 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            float yl[16];
            float yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};

            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
                yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
                yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
            }

            device const uint16_t *sc = (device const uint16_t *)x[ib].scales + iq;
            device const uint16_t *q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half *dh = &x[ib].d;

            for (short row = 0; row < nr0; row++) {
                if (first_row + row < args.ne0) {
                    sc16[0] = sc[0] & kmask1;
                    sc16[1] = sc[2] & kmask1;
                    sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                    sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                    device const uint16_t *q2 = q1 + 32;

                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                    FOR_UNROLL (short i = 0; i < 4; ++i) {
                        acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                        acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                        acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                        acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                        acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                        acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                        acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                        acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
                    }

                    sumf[row] += dh[0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                          (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                          (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                          (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                                 dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                          sumy[2] * sc8[6] + sumy[3] * sc8[7]);
                }

                q1 += args.nb01 / 2;
                sc += args.nb01 / 2;
                dh += args.nb01 / 2;
            }

            y4 += 4 * QK_K;
        }
    }

    device float *dst_f32 = (device float *)(dst + (uint64_t)token * args.nb1);
    for (int row = 0; row < nr0 && first_row + row < args.ne0; row++) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }

    (void)shmem;
    (void)tiitg;
    (void)tgpig;
}

#define QK_NL 16

// Builds the compact per-expert work map used by batched MoE matmul. DS4 routes
// each token to a small fixed top-k list, so this turns token-major ids into
// expert-major slices that the tiled matmul can consume.
template<short ne20>
kernel void kernel_mul_mm_id_map0(
        constant ds4_metal_args_mul_mm_id_map0 & args,
        device  const char * src2,
        device        char * htpe,
        device        char * hids,
        threadgroup   char * shmem [[threadgroup(0)]],
        ushort tpitg[[thread_position_in_threadgroup]],
        ushort   ntg[[threads_per_threadgroup]]) {
    const short ide = tpitg;

    uint32_t n_all = 0;

    device int32_t * ids_i32 = (device int32_t *) hids + ide*args.ne21;

    for (int i21 = 0; i21 < args.ne21; i21 += ntg) {
        if (i21 + tpitg < args.ne21) {
            device const int32_t * src2_i32 = (device const int32_t *) (src2 + (i21 + tpitg)*args.nb21);

            threadgroup uint16_t * sids = (threadgroup uint16_t *) shmem + tpitg*ne20;

            #pragma unroll(ne20)
            for (short i20 = 0; i20 < ne20; i20++) {
                sids[i20] = src2_i32[i20];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short t = 0; t < ntg; t++) {
            if (i21 + t >= args.ne21) {
                break;
            }

            threadgroup const uint16_t * sids = (threadgroup const uint16_t *) shmem + t*ne20;

            short sel = 0;
            #pragma unroll(ne20)
            for (short i20 = 0; i20 < ne20; i20++) {
                sel += (sids[i20] == ide)*(i20 + 1);
            }

            ids_i32[n_all] = (i21 + t)*ne20 + sel - 1;

            n_all += sel > 0;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device uint32_t * tpe_u32 = (device uint32_t *) (htpe);
    tpe_u32[ide] = n_all;
}

typedef decltype(kernel_mul_mm_id_map0<1>) kernel_mul_mm_id_map0_t;

// Host-visible map builders for the routed-expert counts used by DS4 graph
// shapes. Some arities are generic leftovers retained for nearby batch sizes.
template [[host_name("kernel_mul_mm_id_map0_ne20_1" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<1>;
template [[host_name("kernel_mul_mm_id_map0_ne20_2" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<2>;
template [[host_name("kernel_mul_mm_id_map0_ne20_4" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<4>;
template [[host_name("kernel_mul_mm_id_map0_ne20_5" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<5>;
template [[host_name("kernel_mul_mm_id_map0_ne20_6" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<6>;
template [[host_name("kernel_mul_mm_id_map0_ne20_8" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<8>;
template [[host_name("kernel_mul_mm_id_map0_ne20_10")]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<10>;
template [[host_name("kernel_mul_mm_id_map0_ne20_16")]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<16>;
template [[host_name("kernel_mul_mm_id_map0_ne20_22")]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<22>;

// Qwen selects eight of 256 experts for every token.  The rectangular routed
// MM grid consequently contains many empty expert/route tiles: at 2K tokens
// each expert is dispatched for 64 route tiles even though the mean occupancy
// is only two.  Build an exact compact tile list so gate/up and down can
// consume only populated tiles without a CPU readback or a different
// arithmetic path.  worklist[0].x is the populated count; entries begin at 1.
kernel void kernel_mul_mm_id_map0_compact_qwen8(
        constant ds4_metal_args_mul_mm_id_map0 & args,
        device  const char * src2,
        device        char * htpe,
        device        char * hids,
        device       uint2 * worklist,
        threadgroup   char * shmem [[threadgroup(0)]],
        ushort tpitg[[thread_position_in_threadgroup]],
        ushort   ntg[[threads_per_threadgroup]]) {
    constexpr short ne20 = 8;
    const short ide = tpitg;
    uint32_t n_all = 0;
    device int32_t *ids_i32 =
        (device int32_t *)hids + ide*args.ne21;

    for (int i21 = 0; i21 < args.ne21; i21 += ntg) {
        if (i21 + tpitg < args.ne21) {
            device const int32_t *src2_i32 =
                (device const int32_t *)(src2 +
                    (i21 + tpitg)*args.nb21);
            threadgroup uint16_t *sids =
                (threadgroup uint16_t *)shmem + tpitg*ne20;
            #pragma unroll(ne20)
            for (short i20 = 0; i20 < ne20; i20++) {
                sids[i20] = src2_i32[i20];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short t = 0; t < ntg; t++) {
            if (i21 + t >= args.ne21) break;
            threadgroup const uint16_t *sids =
                (threadgroup const uint16_t *)shmem + t*ne20;
            short sel = 0;
            #pragma unroll(ne20)
            for (short i20 = 0; i20 < ne20; i20++) {
                sel += (sids[i20] == ide)*(i20 + 1);
            }
            ids_i32[n_all] = (i21 + t)*ne20 + sel - 1;
            n_all += sel > 0;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device uint32_t *tpe_u32 = (device uint32_t *)htpe;
    tpe_u32[ide] = n_all;
    threadgroup_barrier(mem_flags::mem_device);

    if (tpitg == 0) {
        uint32_t tile_count = 0;
        for (uint32_t expert = 0; expert < (uint32_t)args.ne02;
             expert++) {
            const uint32_t routes = tpe_u32[expert];
            for (uint32_t route = 0; route < routes; route += 32u) {
                worklist[1u + tile_count++] = uint2(expert, route);
            }
        }
        worklist[0] = uint2(tile_count, 0u);
    }
}

// Batched routed-expert matmul. It reads the expert-major map produced above,
// loads selected expert weights, and writes results back to token-major slots
// so the DS4 FFN can apply SwiGLU, weighting, and the down projection.
template<short NR1, typename S0, typename S0_4x4, typename S0_8x8, typename S1, typename S1_2x4, typename S1_8x8, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread S0_4x4 &), typename T0, typename T0_4x4, typename T1, typename T1_2x4, bool compact = false>
kernel void kernel_mul_mm_id(
        constant ds4_metal_args_mul_mm_id & args,
        device const char * src0,
        device const char * src1,
        device const char * htpe,
        device const char * hids,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr int NR0 = 64;
    static_assert(NR1 == 32, "kernel_mul_mm_id accumulator layout supports only 32 routed rows");

    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;
    constexpr int SA_BYTES = NR0 * NR1 * (int)sizeof(S0);

    threadgroup S0 * sa = (threadgroup S0 *)(shmem);
    threadgroup S1 * sb = (threadgroup S1 *)(shmem + SA_BYTES);

    const int r0 = tgpig.y*NR0;

    device const uint32_t * tpe_u32 = (device const uint32_t *) (htpe);
    device const int32_t  * ids_i32 = (device const int32_t  *) (hids);
    device const uint2 *worklist =
        (device const uint2 *)(ids_i32 + args.ne02*args.ne21);
    if (compact && tgpig.x >= worklist[0].x) return;
    const uint2 compact_tile =
        compact ? worklist[1u + tgpig.x] :
                  uint2((uint)tgpig.z, (uint)tgpig.x*NR1);
    const int im = compact_tile.x;
    const int r1 = compact_tile.y;

    const int32_t neh1 = tpe_u32[im];

    if (r1 >= neh1) {
        return;
    }

    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (    neh1 - r1 < NR1) ? (    neh1 - r1) : NR1;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);

    short il = il0;

    const int id = ids_i32[im*args.ne21 + r1 + lr1];

    const short i11 = (id % args.ne20) % args.ne11;
    const short i12 = (id / args.ne20);
    const short i13 = 0;

    const uint64_t offset0 = im*args.nb02 + i13*args.nb03;
    const short    offset1 = il0/nl;

    device const block_q * x = (device const block_q *)(src0 + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const T1 * y = (device const T1 *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*i11
        + args.nb10*iy);

    S0_8x8 ma[4];
    S1_8x8 mb[2];

    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                *(sa + 64*ib + 8*ly + lx) = loop_k + 16*il + i < args.ne00 ? *((device T0 *) x + i) : 0;
            }
        } else {
            S0_4x4 temp_a;
            dequantize_func(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }

        if (FC_mul_mm_bc_inp) {
            for (short i = 0; i < 8; ++i) {
                const short sx = (tiitg%NL1);
                const short sy = (tiitg/NL1)/8;

                const short lx = i;
                const short ly = (tiitg/NL1)%8;

                const short ib = 4*sx + sy;

                *(sb + 64*ib + 8*ly + lx) = loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) y + i) : 0;
            }
        } else {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

            const short ly = (tiitg/NL1)%8;

            const short ib = 4*sx + sy;

            *(threadgroup S1_2x4 *)(sb + 64*ib + 8*ly) = (S1_2x4)(*((device T1_2x4 *) y));
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const S0 * lsma = (sa + 4*64*(sgitg%2));
        threadgroup const S1 * lsmb = (sb + 2*64*(sgitg/2));

        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }

            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short j = sgitg; j < nr1; j += 4) {
        const int idj = ids_i32[im*args.ne21 + r1 + j];

        const short ide = idj % args.ne20;
        const short idt = idj / args.ne20;

        device float  * D  = (device float  *) dst + r0 + ide*args.ne0 + idt*args.ne1*args.ne0;
        device float4 * D4 = (device float4 *) D;

        threadgroup float  * C  = (threadgroup float  *) shmem + j*NR0;
        threadgroup float4 * C4 = (threadgroup float4 *) C;

        int i = tiisg;
        for (; i < nr0/4; i += 32) {
            *(D4 + i) = *(C4 + i);
        }

        i = (4*(nr0/4)) + tiisg;
        for (; i < nr0; i += 32) {
            *(D + i) = *(C + i);
        }
    }
}

// Address-table variant used by SSD streaming. Routing ids retain their model
// identity while each selected expert resolves through a GPU-address table.
template<short NR1, typename S0, typename S0_4x4, typename S0_8x8, typename S1, typename S1_2x4, typename S1_8x8, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread S0_4x4 &), typename T0, typename T0_4x4, typename T1, typename T1_2x4>
kernel void kernel_mul_mm_id_addr(
        constant ds4_metal_args_mul_mm_id & args,
        device const uint64_t * src0_addrs,
        device const char * src1,
        device const char * htpe,
        device const char * hids,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr int NR0 = 64;
    static_assert(NR1 == 32,
                  "kernel_mul_mm_id_addr supports 32 routed rows");

    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;
    constexpr int SA_BYTES = NR0 * NR1 * (int)sizeof(S0);

    threadgroup S0 *sa = (threadgroup S0 *)(shmem);
    threadgroup S1 *sb = (threadgroup S1 *)(shmem + SA_BYTES);

    const int im = tgpig.z;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;
    device const uint32_t *tpe_u32 =
        (device const uint32_t *)(htpe);
    device const int32_t *ids_i32 =
        (device const int32_t *)(hids);
    const int32_t neh1 = tpe_u32[im];
    if (r1 >= neh1) return;

    const uint64_t base_addr = src0_addrs[im];
    if (base_addr == 0) return;
    device const char *src0 =
        reinterpret_cast<device const char *>(base_addr);

    const short nr0 = (args.ne0 - r0 < NR0) ?
        (args.ne0 - r0) : NR0;
    const short nr1 = (neh1 - r1 < NR1) ? (neh1 - r1) : NR1;
    const short lr0 = ((short)tiitg/NL0) < nr0 ?
        ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ?
        ((short)tiitg/NL1) : nr1 - 1;
    const short il0 = tiitg % NL0;
    short il = il0;

    const int id = ids_i32[im*args.ne21 + r1 + lr1];
    const short i11 = (id % args.ne20) % args.ne11;
    const short i12 = id / args.ne20;
    const short i13 = 0;
    const short offset1 = il0/nl;
    device const block_q *x =
        (device const block_q *)(src0 + args.nb01*(r0 + lr0)) +
        offset1;
    const short iy = 8*(tiitg % NL1);
    device const T1 *y = (device const T1 *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*i11
        + args.nb10*iy);

    S0_8x8 ma[4];
    S1_8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;
                const short ib = 8*sx + sy;
                *(sa + 64*ib + 8*ly + lx) =
                    loop_k + 16*il + i < args.ne00 ?
                    *((device T0 *)x + i) : 0;
            }
        } else {
            S0_4x4 temp_a;
            dequantize_func(x, il, temp_a);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;
                const short ib = 8*sx + sy;
                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }

        if (FC_mul_mm_bc_inp) {
            for (short i = 0; i < 8; i++) {
                const short sx = tiitg%NL1;
                const short sy = (tiitg/NL1)/8;
                const short lx = i;
                const short ly = (tiitg/NL1)%8;
                const short ib = 4*sx + sy;
                *(sb + 64*ib + 8*ly + lx) =
                    loop_k + iy + i < args.ne00 ?
                    (S1)*((device T1 *)y + i) : 0;
            }
        } else {
            const short sx = tiitg%NL1;
            const short sy = (tiitg/NL1)/8;
            const short ly = (tiitg/NL1)%8;
            const short ib = 4*sx + sy;
            *(threadgroup S1_2x4 *)(sb + 64*ib + 8*ly) =
                (S1_2x4)(*((device T1_2x4 *)y));
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x = (il < 2) ? x + (2 + nl - 1)/nl : x;
        y += NK;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const S0 *lsma = sa + 4*64*(sgitg%2);
        threadgroup const S1 *lsmb = sb + 2*64*(sgitg/2);
        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(
                    mc[i], mb[i/4], ma[i%4], mc[i]);
            }
            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *temp_str =
        (threadgroup float *)shmem + 32*(sgitg&1) +
        (16*(sgitg >> 1))*NR0;
    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i],
                        temp_str + 8*(i%4) + 8*NR0*(i/4),
                        NR0, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short j = sgitg; j < nr1; j += 4) {
        const int idj = ids_i32[im*args.ne21 + r1 + j];
        const short ide = idj % args.ne20;
        const short idt = idj / args.ne20;
        device float *D = (device float *)dst + r0 +
            ide*args.ne0 + idt*args.ne1*args.ne0;
        device float4 *D4 = (device float4 *)D;
        threadgroup float *C =
            (threadgroup float *)shmem + j*NR0;
        threadgroup float4 *C4 = (threadgroup float4 *)C;
        int i = tiisg;
        for (; i < nr0/4; i += 32) *(D4 + i) = *(C4 + i);
        i = 4*(nr0/4) + tiisg;
        for (; i < nr0; i += 32) *(D + i) = *(C + i);
    }
}

template<short NR1, short NL, typename block_q,
         void (*dequantize_func)(device const block_q *, short,
                                 thread half4x4 &)>
kernel void kernel_mul_mm_id_pair_swiglu_f16(
        constant ds4_metal_args_mul_mm_id & args,
        constant ds4_metal_dsv4_moe_swiglu_weight_args & act,
        device const char * src0_gate,
        device const char * src0_up,
        device const char * src1,
        device const char * htpe,
        device const char * hids,
        device       char * dst_mid,
        device const char * weights,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    static_assert(NR1 == 16 || NR1 == 32,
                  "paired routed MM supports only 16/32 route tiles");
    constexpr short NR0 = 64;
    constexpr short NK  = 32;
    constexpr short NL0 = NK/16;
    constexpr short NL1 = NK/8;
    constexpr short RHS_BLOCKS = NR1/8;
    constexpr short RHS_MATS = NR1/16;
    constexpr short NACC = 4*RHS_MATS;
    threadgroup half *sa = (threadgroup half *)(shmem);
    threadgroup half *sb =
        (threadgroup half *)(shmem + 4096);

    const int im = tgpig.z;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    device const uint32_t * tpe_u32 = (device const uint32_t *) (htpe);
    device const int32_t  * ids_i32 = (device const int32_t  *) (hids);

    const int32_t neh1 = tpe_u32[im];

    if (r1 >= neh1) {
        return;
    }

    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (    neh1 - r1 < NR1) ? (    neh1 - r1) : NR1;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);
    short il = il0;

    const int id = ids_i32[im*args.ne21 + r1 + lr1];

    const short i11 = (id % args.ne20) % args.ne11;
    const short i12 = (id / args.ne20);
    const short i13 = 0;

    const uint64_t offset0 = im*args.nb02 + i13*args.nb03;
    const short    offset1 = il0/NL;

    device const block_q * xg =
        (device const block_q *)(src0_gate +
                                 args.nb01*(r0 + lr0) + offset0) +
        offset1;
    device const block_q * xu =
        (device const block_q *)(src0_up +
                                 args.nb01*(r0 + lr0) + offset0) +
        offset1;

    const short iy = 8*(tiitg % NL1);

    device const float * y = (device const float *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*i11
        + args.nb10*iy);

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[RHS_MATS];

    simdgroup_float8x8 mc_gate[NACC];
    simdgroup_float8x8 mc_up[NACC];

    for (short i = 0; i < NACC; i++) {
        mc_gate[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
        mc_up[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        const short rhs_row = tiitg/NL1;
        if (rhs_row < NR1) {
            const short sx_b = tiitg%NL1;
            const short sy_b = rhs_row/8;
            const short ly_b = rhs_row%8;
            const short ib_b = RHS_BLOCKS*sx_b + sy_b;
            *(threadgroup half2x4 *)(sb + 64*ib_b + 8*ly_b) =
                (half2x4)(*((device float2x4 *) y));
        }

        half4x4 temp_gate;
        dequantize_func(xg, il, temp_gate);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8;
            const short sy = (tiitg/NL0)/8;
            const short lx = (tiitg/NL0)%8;
            const short ly = i%8;
            const short ib = 8*sx + sy;
            *(sa + 64*ib + 8*ly + lx) = temp_gate[i/4][i%4];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma_gate = (sa + 4*64*(sgitg%2));
        threadgroup const half * lsmb =
            sb + RHS_MATS*64*(sgitg/2);

        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma_gate + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < RHS_MATS; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < NACC; i++) {
                simdgroup_multiply_accumulate(mc_gate[i], mb[i/4], ma[i%4], mc_gate[i]);
            }

            lsma_gate += 8*64;
            lsmb += RHS_BLOCKS*64;
        }

        half4x4 temp_up;
        dequantize_func(xu, il, temp_up);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8;
            const short sy = (tiitg/NL0)/8;
            const short lx = (tiitg/NL0)%8;
            const short ly = i%8;
            const short ib = 8*sx + sy;
            *(sa + 64*ib + 8*ly + lx) =
                temp_up[i/4][i%4];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma_up =
            sa + 4*64*(sgitg%2);
        lsmb = sb + RHS_MATS*64*(sgitg/2);

        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma_up + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < RHS_MATS; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < NACC; i++) {
                simdgroup_multiply_accumulate(mc_up[i], mb[i/4], ma[i%4], mc_up[i]);
            }

            lsma_up += 8*64;
            lsmb += RHS_BLOCKS*64;
        }

        /*
         * All four simdgroups consume the same staged RHS tile.  The next K
         * iteration starts by replacing that tile, so a simdgroup-only
         * barrier is insufficient here: a faster simdgroup could otherwise
         * overwrite sb while a slower one is still accumulating the current
         * up projection.  This race is much easier to hit in the paired
         * kernel because each iteration performs two independent MMAs.
         */
        threadgroup_barrier(mem_flags::mem_threadgroup);

        il = (il + 2 < NL) ? il + 2 : il % 2;
        xg = (il < 2) ? xg + (2 + NL - 1)/NL : xg;
        xu = (il < 2) ? xu + (2 + NL - 1)/NL : xu;
        y += NK;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * temp_gate = (threadgroup float *) shmem;
    threadgroup float * temp_up = temp_gate + NR0*NR1;
    const short route_row_base = (NR1/2)*(sgitg >> 1);
    threadgroup float * temp_gate_str =
        temp_gate + 32*(sgitg&1) + route_row_base*NR0;
    threadgroup float * temp_up_str =
        temp_up + 32*(sgitg&1) + route_row_base*NR0;

    for (short i = 0; i < NACC; i++) {
        simdgroup_store(mc_gate[i], temp_gate_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        simdgroup_store(mc_up[i],   temp_up_str   + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float c = act.clamp_value;
    for (short j = sgitg; j < nr1; j += 4) {
        const int idj = ids_i32[im*args.ne21 + r1 + j];

        const short ide = idj % args.ne20;
        const short idt = idj / args.ne20;

        device half *D = (device half *)(dst_mid +
            ((uint64_t)idt*args.ne1 + (uint64_t)ide)*act.mid_row_stride) + r0;
        device const float *w = (device const float *)(weights + (uint64_t)idj*act.weight_stride);
        const float route_weight = w[0];

        threadgroup float *Cg = temp_gate + j*NR0;
        threadgroup float *Cu = temp_up   + j*NR0;

        int i = tiisg;
        for (; i < nr0; i += 32) {
            float g = Cg[i];
            float u = Cu[i];
            if (c > 1.0e-6f) {
                g = min(g, c);
                u = clamp(u, -c, c);
            }
            const float silu = g / (1.0f + exp(-g));
            D[i] = (half)(silu * u * route_weight);
        }
    }
}

#ifdef DS4_METAL_HAS_TENSOR

// MLX's long-prompt SwitchLinear path sorts routes by expert and evaluates
// affine 4-bit weights with Metal 4 TensorOps.  DS4's map already provides the
// same expert-major route ordering, so this affine-only kernel keeps the
// existing canonical scatter but replaces the inner SIMD-group MMA with the
// cooperative TensorOps primitive.  The GGML quantized paths intentionally do
// not use it: their accumulated arithmetic has a separate semantic baseline.
template<typename T1>
kernel void kernel_mul_mm_id_mlx_affine4_64_nax(
        constant ds4_metal_args_mul_mm_id & args,
        device const char * src0,
        device const char * src1,
        device const char * htpe,
        device const char * hids,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    (void)sgitg;

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;
    constexpr short NL = 4;

    threadgroup half *sa = (threadgroup half *)shmem;
    threadgroup half *sb = sa + NR0*NK;
    threadgroup float *sc = (threadgroup float *)shmem;

    const int expert = tgpig.z;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;
    device const uint32_t *counts = (device const uint32_t *)htpe;
    device const int32_t *route_ids = (device const int32_t *)hids;
    const int route_count = counts[expert];
    if (r1 >= route_count) return;

    const short nr0 = min((int)NR0, args.ne0 - r0);
    const short nr1 = min((int)NR1, route_count - r1);
    const short lr0 = min((short)(tiitg/NL0), (short)(nr0 - 1));
    const short lr1 = min((short)(tiitg/NL1), (short)(nr1 - 1));
    const short il0 = tiitg % NL0;
    short il = il0;

    const int route_id = route_ids[expert*args.ne21 + r1 + lr1];
    const short route_slot = (route_id % args.ne20) % args.ne11;
    const short token = route_id / args.ne20;
    const uint64_t expert_offset = (uint64_t)expert*args.nb02;
    device const block_mlx_affine4_64 *w =
        (device const block_mlx_affine4_64 *)(
            src0 + args.nb01*(r0 + lr0) + expert_offset);
    device const T1 *x = (device const T1 *)(src1
        + args.nb12*token
        + args.nb11*route_slot
        + args.nb10*(8*(tiitg % NL1)));

    auto tA = tensor(sa, dextents<int32_t, 2>(NK, NR0));
    auto tB = tensor(sb, dextents<int32_t, 2>(NR1, NK));
    matmul2d<
        matmul2d_descriptor(NR1, NR0, NK, false, true, false,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;
    auto accum =
        mm.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();

    #pragma unroll
    for (uint16_t i = 0; i < accum.get_capacity(); i++) {
        if (accum.is_valid_element(i)) accum[i] = 0.0f;
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        half4x4 temp_w;
        dequantize_mlx_affine4_64(w, il, temp_w);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8;
            const short sy = (tiitg/NL0)/8;
            const short lx = i%8;
            const short ly = (tiitg/NL0)%8;
            sa[NK*(8*sy + ly) + 8*sx + lx] = temp_w[i/4][i%4];
        }

        const short sx = tiitg%NL1;
        const short sy = (tiitg/NL1)/8;
        const short ly = (tiitg/NL1)%8;
        threadgroup half *tile_x = sb + NK*(8*sy + ly) + 8*sx;
        #pragma unroll
        for (short i = 0; i < 8; ++i) {
            tile_x[i] = (half)x[i];
        }

        il = (il + 2 < NL) ? il + 2 : il % 2;
        w = (il < 2) ? w + (2 + NL - 1)/NL : w;
        x += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto tile_a = tA.slice(0, 0);
        auto tile_b = tB.slice(0, 0);
        mm.run(tile_b, tile_a, accum);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto tile_c = tensor(sc, dextents<int32_t, 2>(NR0, NR1));
    accum.store(tile_c);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short j = tiitg/32; j < nr1; j += 4) {
        const int canonical = route_ids[expert*args.ne21 + r1 + j];
        const short slot = canonical % args.ne20;
        const short tok = canonical / args.ne20;
        device float *out = (device float *)dst + r0 +
            slot*args.ne0 + tok*args.ne1*args.ne0;
        device float4 *out4 = (device float4 *)out;
        threadgroup float *tile = sc + j*NR0;
        threadgroup float4 *tile4 = (threadgroup float4 *)tile;
        int i = tiisg;
        for (; i < nr0/4; i += 32) out4[i] = tile4[i];
        i = 4*(nr0/4) + tiisg;
        for (; i < nr0; i += 32) out[i] = tile[i];
    }
}

typedef decltype(kernel_mul_mm_id_mlx_affine4_64_nax<float>)
    mul_mm_id_mlx_affine4_64_nax_f32_t;
typedef decltype(kernel_mul_mm_id_mlx_affine4_64_nax<half>)
    mul_mm_id_mlx_affine4_64_nax_f16_t;

template [[host_name("kernel_mul_mm_id_mlx_affine4_64_nax_f32")]]
kernel mul_mm_id_mlx_affine4_64_nax_f32_t
kernel_mul_mm_id_mlx_affine4_64_nax<float>;
template [[host_name("kernel_mul_mm_id_mlx_affine4_64_nax_f16")]]
kernel mul_mm_id_mlx_affine4_64_nax_f16_t
kernel_mul_mm_id_mlx_affine4_64_nax<half>;

#endif


typedef decltype(kernel_mul_mm_id_pair_swiglu_f16<
    32, QK_NL, block_iq2_xxs, dequantize_iq2_xxs>)
    mul_mm_id_iq2_xxs_pair_swiglu_f16_n32_t;
typedef decltype(kernel_mul_mm_id_pair_swiglu_f16<
    16, QK_NL, block_iq2_xxs, dequantize_iq2_xxs>)
    mul_mm_id_iq2_xxs_pair_swiglu_f16_n16_t;
typedef decltype(kernel_mul_mm_id_pair_swiglu_f16<
    32, QK_NL, block_q4_K, dequantize_q4_K>)
    mul_mm_id_q4_K_pair_swiglu_f16_n32_t;
typedef decltype(kernel_mul_mm_id_pair_swiglu_f16<
    16, QK_NL, block_q4_K, dequantize_q4_K>)
    mul_mm_id_q4_K_pair_swiglu_f16_n16_t;
typedef decltype(kernel_mul_mm_id_pair_swiglu_f16<
    32, 4, block_mlx_affine4_64, dequantize_mlx_affine4_64>)
    mul_mm_id_mlx_affine4_64_pair_swiglu_f16_n32_t;
typedef decltype(kernel_mul_mm_id_pair_swiglu_f16<
    16, 4, block_mlx_affine4_64, dequantize_mlx_affine4_64>)
    mul_mm_id_mlx_affine4_64_pair_swiglu_f16_n16_t;
template [[host_name("kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16")]]
kernel mul_mm_id_iq2_xxs_pair_swiglu_f16_n32_t
kernel_mul_mm_id_pair_swiglu_f16<
    32, QK_NL, block_iq2_xxs, dequantize_iq2_xxs>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16_n16")]]
kernel mul_mm_id_iq2_xxs_pair_swiglu_f16_n16_t
kernel_mul_mm_id_pair_swiglu_f16<
    16, QK_NL, block_iq2_xxs, dequantize_iq2_xxs>;
template [[host_name("kernel_mul_mm_id_q4_K_pair_swiglu_f16")]]
kernel mul_mm_id_q4_K_pair_swiglu_f16_n32_t
kernel_mul_mm_id_pair_swiglu_f16<
    32, QK_NL, block_q4_K, dequantize_q4_K>;
template [[host_name("kernel_mul_mm_id_q4_K_pair_swiglu_f16_n16")]]
kernel mul_mm_id_q4_K_pair_swiglu_f16_n16_t
kernel_mul_mm_id_pair_swiglu_f16<
    16, QK_NL, block_q4_K, dequantize_q4_K>;
template [[host_name("kernel_mul_mm_id_mlx_affine4_64_pair_swiglu_f16")]]
kernel mul_mm_id_mlx_affine4_64_pair_swiglu_f16_n32_t
kernel_mul_mm_id_pair_swiglu_f16<
    32, 4, block_mlx_affine4_64, dequantize_mlx_affine4_64>;
template [[host_name("kernel_mul_mm_id_mlx_affine4_64_pair_swiglu_f16_n16")]]
kernel mul_mm_id_mlx_affine4_64_pair_swiglu_f16_n16_t
kernel_mul_mm_id_pair_swiglu_f16<
    16, 4, block_mlx_affine4_64, dequantize_mlx_affine4_64>;
typedef decltype(kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K, QK_NL, dequantize_q2_K, float, float4x4, float, float2x4>) mul_mm_id;
typedef decltype(kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K, QK_NL, dequantize_q2_K, half, half4x4, half, half2x4>) mul_mm_id_f16_rhs;
typedef decltype(kernel_mul_mm_id<32, float, float4x4, simdgroup_float8x8, float, float2x4, simdgroup_float8x8, block_q2_K, QK_NL, dequantize_q2_K, float, float4x4, float, float2x4>) mul_mm_id_ff32;
typedef decltype(kernel_mul_mm_id_addr<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K, QK_NL, dequantize_q2_K, float, float4x4, float, float2x4>) mul_mm_id_addr;
typedef decltype(kernel_mul_mm_id_addr<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K, QK_NL, dequantize_q2_K, half, half4x4, half, half2x4>) mul_mm_id_addr_f16_rhs;

// Host-visible batched MoE matmul variants for the DS4 quant formats.
template [[host_name("kernel_mul_mm_id_q8_0_f32")]]         kernel mul_mm_id kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q8_0,    2,     dequantize_q8_0,    float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_q2_K_f32")]]         kernel mul_mm_id kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K,    QK_NL, dequantize_q2_K,    float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_q4_K_f32")]]         kernel mul_mm_id kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q4_K,    QK_NL, dequantize_q4_K,    float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_mlx_affine4_64_f32")]] kernel mul_mm_id kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_mlx_affine4_64, 4, dequantize_mlx_affine4_64, float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_q5_K_f32")]]         kernel mul_mm_id kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q5_K,    QK_NL, dequantize_q5_K,    float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_q6_K_f32")]]         kernel mul_mm_id kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q6_K,    QK_NL, dequantize_q6_K,    float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f32")]]      kernel mul_mm_id kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xs_f32")]]       kernel mul_mm_id kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq3_xxs_f32")]]      kernel mul_mm_id kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq4_xs_f32")]]       kernel mul_mm_id kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_q8_0_f16")]]         kernel mul_mm_id_f16_rhs kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q8_0,    2,     dequantize_q8_0,    half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_q2_K_f16")]]         kernel mul_mm_id_f16_rhs kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K,    QK_NL, dequantize_q2_K,    half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_q4_K_f16")]]         kernel mul_mm_id_f16_rhs kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q4_K,    QK_NL, dequantize_q4_K,    half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_mlx_affine4_64_f16")]] kernel mul_mm_id_f16_rhs kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_mlx_affine4_64, 4, dequantize_mlx_affine4_64, half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_q5_K_f16")]]         kernel mul_mm_id_f16_rhs kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q5_K,    QK_NL, dequantize_q5_K,    half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_q6_K_f16")]]         kernel mul_mm_id_f16_rhs kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q6_K,    QK_NL, dequantize_q6_K,    half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f16")]]      kernel mul_mm_id_f16_rhs kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq2_xxs, QK_NL, dequantize_iq2_xxs, half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xs_f16")]]       kernel mul_mm_id_f16_rhs kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq2_xs,  QK_NL, dequantize_iq2_xs,  half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq3_xxs_f16")]]      kernel mul_mm_id_f16_rhs kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq3_xxs, QK_NL, dequantize_iq3_xxs, half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq4_xs_f16")]]       kernel mul_mm_id_f16_rhs kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq4_xs,  QK_NL, dequantize_iq4_xs,  half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_q4_K_ff32")]]        kernel mul_mm_id_ff32 kernel_mul_mm_id<32, float, float4x4, simdgroup_float8x8, float, float2x4, simdgroup_float8x8, block_q4_K, QK_NL, dequantize_q4_K, float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_q5_K_ff32")]]        kernel mul_mm_id_ff32 kernel_mul_mm_id<32, float, float4x4, simdgroup_float8x8, float, float2x4, simdgroup_float8x8, block_q5_K, QK_NL, dequantize_q5_K, float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_q6_K_ff32")]]        kernel mul_mm_id_ff32 kernel_mul_mm_id<32, float, float4x4, simdgroup_float8x8, float, float2x4, simdgroup_float8x8, block_q6_K, QK_NL, dequantize_q6_K, float, float4x4, float, float2x4>;

// Exact Qwen IQ compact-grid variants.  Only the grid mapping differs from
// the established kernels above; dequantization, half-tile loads, SIMD-group
// MMA sequence, F32 accumulation, and token-major stores are identical.
typedef decltype(kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq2_xs, QK_NL, dequantize_iq2_xs, float, float4x4, float, float2x4, true>) mul_mm_id_compact;
typedef decltype(kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq2_xs, QK_NL, dequantize_iq2_xs, half, half4x4, half, half2x4, true>) mul_mm_id_compact_f16_rhs;
template [[host_name("kernel_mul_mm_id_compact_iq2_xs_f32")]]
kernel mul_mm_id_compact kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq2_xs, QK_NL, dequantize_iq2_xs, float, float4x4, float, float2x4, true>;
template [[host_name("kernel_mul_mm_id_compact_iq3_xxs_f32")]]
kernel mul_mm_id_compact kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float, float4x4, float, float2x4, true>;
template [[host_name("kernel_mul_mm_id_compact_iq4_xs_f32")]]
kernel mul_mm_id_compact kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq4_xs, QK_NL, dequantize_iq4_xs, float, float4x4, float, float2x4, true>;
template [[host_name("kernel_mul_mm_id_compact_iq2_xs_f16")]]
kernel mul_mm_id_compact_f16_rhs kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq2_xs, QK_NL, dequantize_iq2_xs, half, half4x4, half, half2x4, true>;
template [[host_name("kernel_mul_mm_id_compact_iq3_xxs_f16")]]
kernel mul_mm_id_compact_f16_rhs kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq3_xxs, QK_NL, dequantize_iq3_xxs, half, half4x4, half, half2x4, true>;
template [[host_name("kernel_mul_mm_id_compact_iq4_xs_f16")]]
kernel mul_mm_id_compact_f16_rhs kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq4_xs, QK_NL, dequantize_iq4_xs, half, half4x4, half, half2x4, true>;

template [[host_name("kernel_mul_mm_id_addr_q2_K_f32")]]    kernel mul_mm_id_addr kernel_mul_mm_id_addr<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K, QK_NL, dequantize_q2_K, float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_addr_q4_K_f32")]]    kernel mul_mm_id_addr kernel_mul_mm_id_addr<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q4_K, QK_NL, dequantize_q4_K, float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_addr_mlx_affine4_64_f32")]] kernel mul_mm_id_addr kernel_mul_mm_id_addr<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_mlx_affine4_64, 4, dequantize_mlx_affine4_64, float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_addr_iq2_xs_f32")]]  kernel mul_mm_id_addr kernel_mul_mm_id_addr<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq2_xs, QK_NL, dequantize_iq2_xs, float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_addr_iq3_xxs_f32")]] kernel mul_mm_id_addr kernel_mul_mm_id_addr<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_addr_iq4_xs_f32")]]  kernel mul_mm_id_addr kernel_mul_mm_id_addr<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq4_xs, QK_NL, dequantize_iq4_xs, float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_id_addr_q2_K_f16")]]    kernel mul_mm_id_addr_f16_rhs kernel_mul_mm_id_addr<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K, QK_NL, dequantize_q2_K, half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_addr_q4_K_f16")]]    kernel mul_mm_id_addr_f16_rhs kernel_mul_mm_id_addr<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q4_K, QK_NL, dequantize_q4_K, half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_addr_mlx_affine4_64_f16")]] kernel mul_mm_id_addr_f16_rhs kernel_mul_mm_id_addr<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_mlx_affine4_64, 4, dequantize_mlx_affine4_64, half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_addr_iq2_xs_f16")]]  kernel mul_mm_id_addr_f16_rhs kernel_mul_mm_id_addr<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq2_xs, QK_NL, dequantize_iq2_xs, half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_addr_iq3_xxs_f16")]] kernel mul_mm_id_addr_f16_rhs kernel_mul_mm_id_addr<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq3_xxs, QK_NL, dequantize_iq3_xxs, half, half4x4, half, half2x4>;
template [[host_name("kernel_mul_mm_id_addr_iq4_xs_f16")]]  kernel mul_mm_id_addr_f16_rhs kernel_mul_mm_id_addr<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_iq4_xs, QK_NL, dequantize_iq4_xs, half, half4x4, half, half2x4>;

#ifdef DS4_METAL_HAS_TENSOR
// Attention-output low-rank projection retained for Metal4 prefill.  It uses
// the same direct-RHS idea as dense matmul: dequantize the Q8_0 low projection
// weights to a half tile, then let TensorOps read the dense head activations
// directly.  Only the 64-token direct-RHS instantiation is exported because the
// staged-RHS and 32-token variants were benchmark-only experiments.
template<short NR1>
kernel void kernel_attn_out_low_q8_0_mpp_direct_rhs(
        constant ds4_metal_args_mul_mm_id & args,
        device const char * srcA,
        device const char * srcB,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    (void) sgitg;

    constexpr int NR0 = 64;
    constexpr int NK  = 32;
    constexpr int NL  = NK/16;
    constexpr int NUM_THREADS = 128;

    const int K = args.ne00;
    const int M = args.ne0;
    const int N = args.ne21;
    const int G = args.ne1;
    const int group = tgpig.z;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;
    const bool full_tile = r0 + NR0 <= M && r1 + NR1 <= N && (K % NK) == 0;

    threadgroup half *sa = (threadgroup half *)shmem;
    auto tA = tensor(sa, dextents<int32_t, 2>(NK, NR0));

    device float *ptrB = (device float *)(srcB + args.nb11*group);
    const int strideB = args.nb12/sizeof(float);
    auto tB = tensor(ptrB, dextents<int32_t, 2>(K, N), array<int, 2>({1, strideB}));

    matmul2d<
        matmul2d_descriptor(NR1, NR0, NK, false, true, true,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            cT[i] = 0.0f;
        }
    }

    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        for (int work = tiitg; work < NR0*NL; work += NUM_THREADS) {
            const int row = work/NL;
            const int k_chunk = work%NL;
            const int k_pos = loop_k + k_chunk*16;
            const short k_base = k_chunk*16;

            if (full_tile || r0 + row < M) {
                const int block_idx = k_pos/32;
                const short il = (k_pos/16)%2;
                device const block_q8_0 *row_ptr =
                    (device const block_q8_0 *)(srcA + args.nb01*(r0 + row) + group*args.nb02);

                half4x4 temp_a;
                dequantize_q8_0(row_ptr + block_idx, il, temp_a);
                FOR_UNROLL (short i = 0; i < 16; i++) {
                    sa[row*NK + k_base + i] = (full_tile || k_pos + i < K) ? temp_a[i/4][i%4] : (half)0;
                }
            } else {
                FOR_UNROLL (short i = 0; i < 16; i++) {
                    sa[row*NK + k_base + i] = (half)0;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto mA = tA.slice(0, 0);
        auto mB = tB.slice(loop_k, r1);
        mm.run(mB, mA, cT);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float *dst_group = (device float *)dst + group*M;
    if (full_tile) {
        device float *dst_tile = dst_group + r0 + (uint64_t)r1*G*M;
        auto tD = tensor(dst_tile, dextents<int32_t, 2>(NR0, NR1), array<int, 2>({1, G*M}));
        cT.store(tD);
    } else {
        auto tD = tensor(dst_group, dextents<int32_t, 2>(M, N), array<int, 2>({1, G*M}));
        auto mD = tD.slice(r0, r1);
        cT.store(mD);
    }
}

typedef decltype(kernel_attn_out_low_q8_0_mpp_direct_rhs<64>) attn_out_low_q8_0_mpp_direct_rhs_n64_t;

template [[host_name("kernel_attn_out_low_q8_0_mpp_direct_rhs_n64")]] kernel attn_out_low_q8_0_mpp_direct_rhs_n64_t kernel_attn_out_low_q8_0_mpp_direct_rhs<64>;

#endif

#undef QK_NL
#undef kmask_iq2xs
#undef ksigns_iq2xs
#undef iq2xxs_grid
#undef iq2xs_grid
#undef iq3xxs_grid
#undef kvalues_iq4nl_f
#undef QK_K
#undef N_R0_Q2_K
#undef N_R0_Q4_K
#undef N_R0_GLM_Q4_PAIR2_K
#undef N_R0_GLM_Q4_PAIR_K
#undef N_R0_Q5_PAIR_K
#undef N_R0_Q5_K
#undef N_R0_Q6_K
#undef N_R0_IQ2_XXS
#undef N_R0_IQ2_XS
#undef N_R0_IQ3_XXS
#undef N_R0_IQ4_XS
