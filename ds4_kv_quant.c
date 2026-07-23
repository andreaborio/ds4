#include "ds4_kv_quant.h"

#include <float.h>
#include <math.h>
#include <string.h>

/*
 * Standard-normal Lloyd-Max centroids. A vector normalized to unit L2 norm
 * and randomized by an orthogonal transform has coordinate variance 1/d, so
 * the runtime scales this table by 1/sqrt(d).
 *
 * Provenance and the reproducible solver are documented in
 * docs/architecture/KV_QUANTIZATION.md.
 */
/* BEGIN GENERATED DS4_KV_TQ4_CENTROIDS */
static const float ds4_kv_tq4_centroid_std_normal[16] = {
    -2.730922219f, -2.068447119f, -1.617881760f, -1.256257570f,
    -0.942448243f, -0.656879919f, -0.388137791f, -0.128427661f,
     0.128427661f,  0.388137791f,  0.656879919f,  0.942448243f,
     1.256257570f,  1.617881760f,  2.068447119f,  2.730922219f,
};
/* END GENERATED DS4_KV_TQ4_CENTROIDS */

static bool ds4_kv_u64_mul(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out || (a != 0 && b > UINT64_MAX / a)) return false;
    *out = a * b;
    return true;
}

static bool ds4_kv_u64_add(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out || b > UINT64_MAX - a) return false;
    *out = a + b;
    return true;
}

static bool ds4_kv_surface_semantics_valid(
        const ds4_kv_surface *surface) {
    bool family_kind_valid = false;
    switch (surface->family) {
    case DS4_KV_FAMILY_QWEN35:
        family_kind_valid =
            surface->kind == DS4_KV_SURFACE_FULL_KEY ||
            surface->kind == DS4_KV_SURFACE_FULL_VALUE;
        break;
    case DS4_KV_FAMILY_DEEPSEEK4:
        family_kind_valid =
            surface->kind == DS4_KV_SURFACE_RAW_MLA ||
            surface->kind == DS4_KV_SURFACE_COMPRESSED_ATTN ||
            surface->kind == DS4_KV_SURFACE_INDEXER_KEY;
        break;
    case DS4_KV_FAMILY_GLM52:
        family_kind_valid =
            surface->kind == DS4_KV_SURFACE_COMPACT_KV_LORA ||
            surface->kind == DS4_KV_SURFACE_COMPACT_K_ROPE ||
            surface->kind == DS4_KV_SURFACE_INDEXER_KEY;
        break;
    default:
        return false;
    }
    if (!family_kind_valid) return false;

    if (surface->storage == DS4_KV_STORAGE_TQ4_KEY) {
        return (surface->kind == DS4_KV_SURFACE_FULL_KEY ||
                surface->kind == DS4_KV_SURFACE_INDEXER_KEY) &&
               (surface->vector_dim &
                (surface->vector_dim - 1u)) == 0u;
    }
    if (surface->storage == DS4_KV_STORAGE_TQ4_VALUE) {
        return surface->kind == DS4_KV_SURFACE_FULL_VALUE;
    }
    return true;
}

static bool ds4_kv_storage_vector_bytes(
        ds4_kv_storage storage,
        uint32_t       vector_dim,
        uint64_t      *data_bytes,
        uint64_t      *metadata_bytes) {
    if (!data_bytes || !metadata_bytes || vector_dim == 0) return false;

    switch (storage) {
    case DS4_KV_STORAGE_F32:
        *metadata_bytes = 0;
        return ds4_kv_u64_mul(vector_dim, sizeof(float), data_bytes);
    case DS4_KV_STORAGE_F16:
        *metadata_bytes = 0;
        return ds4_kv_u64_mul(vector_dim, sizeof(uint16_t), data_bytes);
    case DS4_KV_STORAGE_Q8_SCALE_F16:
        *data_bytes = vector_dim;
        *metadata_bytes = sizeof(uint16_t);
        return true;
    case DS4_KV_STORAGE_TQ4_KEY:
        *data_bytes = ((uint64_t)vector_dim + 1u) / 2u;
        *metadata_bytes = sizeof(uint16_t);
        return true;
    case DS4_KV_STORAGE_TQ4_VALUE:
        *data_bytes = ((uint64_t)vector_dim + 1u) / 2u;
        *metadata_bytes = 2u * sizeof(uint16_t);
        return true;
    default:
        return false;
    }
}

bool ds4_kv_surface_plan_checked(
        const ds4_kv_surface *surface,
        ds4_kv_surface_plan  *plan) {
    if (!surface || !plan || surface->family < DS4_KV_FAMILY_QWEN35 ||
        surface->family > DS4_KV_FAMILY_GLM52 ||
        surface->kind < DS4_KV_SURFACE_FULL_KEY ||
        surface->kind > DS4_KV_SURFACE_INDEXER_KEY ||
        surface->layer_count == 0 || surface->capacity_rows == 0 ||
        surface->vectors_per_row == 0 || surface->vector_dim == 0 ||
        !ds4_kv_surface_semantics_valid(surface)) {
        return false;
    }

    uint64_t vector_data = 0;
    uint64_t vector_metadata = 0;
    uint64_t vector_stride = 0;
    uint64_t row_vectors = surface->vectors_per_row;
    uint64_t row_data = 0;
    uint64_t row_metadata = 0;
    uint64_t row_stride = 0;
    uint64_t surface_rows = 0;
    uint64_t total_data = 0;
    uint64_t total_metadata = 0;
    uint64_t total = 0;

    if (!ds4_kv_storage_vector_bytes(
             surface->storage, surface->vector_dim,
             &vector_data, &vector_metadata) ||
        !ds4_kv_u64_add(vector_data, vector_metadata, &vector_stride) ||
        !ds4_kv_u64_mul(vector_data, row_vectors, &row_data) ||
        !ds4_kv_u64_mul(vector_metadata, row_vectors, &row_metadata) ||
        !ds4_kv_u64_add(row_data, row_metadata, &row_stride) ||
        !ds4_kv_u64_mul(
             surface->layer_count, surface->capacity_rows, &surface_rows) ||
        !ds4_kv_u64_mul(surface_rows, row_data, &total_data) ||
        !ds4_kv_u64_mul(surface_rows, row_metadata, &total_metadata) ||
        !ds4_kv_u64_add(total_data, total_metadata, &total)) {
        return false;
    }

    *plan = (ds4_kv_surface_plan){
        .packed_data_bytes = total_data,
        .metadata_bytes = total_metadata,
        .total_bytes = total,
        .vector_stride_bytes = vector_stride,
        .row_stride_bytes = row_stride,
    };
    return true;
}

bool ds4_kv_plan_add_checked(
        ds4_kv_plan_total         *total,
        const ds4_kv_surface_plan *surface) {
    if (!total || !surface) return false;
    ds4_kv_plan_total next = *total;
    return ds4_kv_u64_add(
               next.packed_data_bytes, surface->packed_data_bytes,
               &next.packed_data_bytes) &&
           ds4_kv_u64_add(
               next.metadata_bytes, surface->metadata_bytes,
               &next.metadata_bytes) &&
           ds4_kv_u64_add(
               next.total_bytes, surface->total_bytes,
               &next.total_bytes) &&
           (*total = next, true);
}

size_t ds4_kv_tq4_key_bytes(uint32_t vector_dim) {
    if (vector_dim == 0 ||
        (uint64_t)vector_dim + 1u > 2u * (uint64_t)SIZE_MAX) {
        return 0;
    }
    const size_t data = ((size_t)vector_dim + 1u) / 2u;
    return data > SIZE_MAX - sizeof(uint16_t)
        ? 0 : data + sizeof(uint16_t);
}

size_t ds4_kv_tq4_value_bytes(uint32_t vector_dim) {
    if (vector_dim == 0 ||
        (uint64_t)vector_dim + 1u > 2u * (uint64_t)SIZE_MAX) {
        return 0;
    }
    const size_t data = ((size_t)vector_dim + 1u) / 2u;
    return data > SIZE_MAX - 2u * sizeof(uint16_t)
        ? 0 : data + 2u * sizeof(uint16_t);
}

static bool ds4_kv_is_power_of_two(uint32_t n) {
    return n != 0 && (n & (n - 1u)) == 0;
}

static uint32_t ds4_kv_mix32(uint32_t x) {
    x ^= x >> 16;
    x *= UINT32_C(0x7feb352d);
    x ^= x >> 15;
    x *= UINT32_C(0x846ca68b);
    x ^= x >> 16;
    return x;
}

static float ds4_kv_sign(uint32_t seed, uint32_t dim) {
    const uint32_t mixed =
        ds4_kv_mix32(seed ^ (UINT32_C(0x9e3779b9) * (dim + 1u)));
    return (mixed & 1u) ? 1.0f : -1.0f;
}

static void ds4_kv_wht(float *values, uint32_t n) {
    for (uint32_t width = 1u; width < n; width <<= 1u) {
        const uint32_t span = width << 1u;
        for (uint32_t base = 0; base < n; base += span) {
            for (uint32_t j = 0; j < width; j++) {
                const float a = values[base + j];
                const float b = values[base + width + j];
                values[base + j] = a + b;
                values[base + width + j] = a - b;
            }
        }
    }
    const float scale = 1.0f / sqrtf((float)n);
    for (uint32_t i = 0; i < n; i++) values[i] *= scale;
}

static uint16_t ds4_kv_f32_to_f16(float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & UINT32_C(0x8000);
    const uint32_t magnitude = bits & UINT32_C(0x7fffffff);

    if (magnitude >= UINT32_C(0x7f800000)) {
        const uint16_t nan = (magnitude > UINT32_C(0x7f800000))
            ? UINT16_C(0x0200) : UINT16_C(0);
        return (uint16_t)(sign | UINT32_C(0x7c00) | nan);
    }
    if (magnitude > UINT32_C(0x477fefff)) {
        return (uint16_t)(sign | UINT32_C(0x7c00));
    }
    if (magnitude < UINT32_C(0x33000001)) {
        return (uint16_t)sign;
    }

    int32_t exponent = (int32_t)(magnitude >> 23) - 127 + 15;
    uint32_t mantissa = magnitude & UINT32_C(0x007fffff);
    if (exponent <= 0) {
        mantissa |= UINT32_C(0x00800000);
        const uint32_t shift = (uint32_t)(14 - exponent);
        uint32_t half_mantissa = mantissa >> shift;
        const uint32_t remainder = mantissa & ((UINT32_C(1) << shift) - 1u);
        const uint32_t halfway = UINT32_C(1) << (shift - 1u);
        if (remainder > halfway ||
            (remainder == halfway && (half_mantissa & 1u))) {
            half_mantissa++;
        }
        return (uint16_t)(sign | half_mantissa);
    }

    uint32_t half_mantissa = mantissa >> 13;
    const uint32_t remainder = mantissa & UINT32_C(0x1fff);
    if (remainder > UINT32_C(0x1000) ||
        (remainder == UINT32_C(0x1000) && (half_mantissa & 1u))) {
        half_mantissa++;
        if (half_mantissa == UINT32_C(0x0400)) {
            half_mantissa = 0;
            exponent++;
        }
    }
    if (exponent >= 31) return (uint16_t)(sign | UINT32_C(0x7c00));
    return (uint16_t)(
        sign | ((uint32_t)exponent << 10) | half_mantissa);
}

static float ds4_kv_f16_to_f32(uint16_t value) {
    const uint32_t sign = ((uint32_t)value & UINT32_C(0x8000)) << 16;
    uint32_t exponent = ((uint32_t)value >> 10) & UINT32_C(0x1f);
    uint32_t mantissa = (uint32_t)value & UINT32_C(0x03ff);
    uint32_t bits = 0;

    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int32_t shift = 0;
            while ((mantissa & UINT32_C(0x0400)) == 0) {
                mantissa <<= 1;
                shift++;
            }
            mantissa &= UINT32_C(0x03ff);
            exponent = (uint32_t)(127 - 15 - shift + 1);
            bits = sign | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | UINT32_C(0x7f800000) | (mantissa << 13);
    } else {
        exponent = exponent + (127u - 15u);
        bits = sign | (exponent << 23) | (mantissa << 13);
    }

    float result = 0.0f;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static void ds4_kv_store_u16_le(uint8_t *dst, uint16_t value) {
    dst[0] = (uint8_t)(value & UINT16_C(0xff));
    dst[1] = (uint8_t)(value >> 8);
}

static uint16_t ds4_kv_load_u16_le(const uint8_t *src) {
    return (uint16_t)((uint16_t)src[0] | ((uint16_t)src[1] << 8));
}

static uint8_t ds4_kv_tq4_nearest(float value, float centroid_scale) {
    uint8_t best = 0;
    float best_distance = FLT_MAX;
    for (uint8_t i = 0; i < 16; i++) {
        const float centroid =
            ds4_kv_tq4_centroid_std_normal[i] * centroid_scale;
        const float distance = fabsf(value - centroid);
        if (distance < best_distance) {
            best = i;
            best_distance = distance;
        }
    }
    return best;
}

bool ds4_kv_tq4_key_encode_reference(
        uint8_t       *packed,
        size_t         packed_bytes,
        const float   *input,
        uint32_t       vector_dim,
        uint32_t       seed,
        float         *scratch,
        size_t         scratch_count) {
    const size_t required = ds4_kv_tq4_key_bytes(vector_dim);
    if (!packed || !input || !scratch || required == 0 ||
        packed_bytes < required || scratch_count < vector_dim ||
        !ds4_kv_is_power_of_two(vector_dim)) {
        return false;
    }

    double norm2 = 0.0;
    for (uint32_t i = 0; i < vector_dim; i++) {
        if (!isfinite(input[i])) return false;
        norm2 += (double)input[i] * input[i];
    }
    const float norm = (float)sqrt(norm2);
    const float inverse_norm = norm > 0.0f ? 1.0f / norm : 0.0f;
    for (uint32_t i = 0; i < vector_dim; i++) {
        scratch[i] = input[i] * inverse_norm * ds4_kv_sign(seed, i);
    }
    ds4_kv_wht(scratch, vector_dim);

    const float centroid_scale = 1.0f / sqrtf((float)vector_dim);
    const size_t data_bytes = ((size_t)vector_dim + 1u) / 2u;
    memset(packed, 0, data_bytes);
    for (uint32_t i = 0; i < vector_dim; i++) {
        const uint8_t code =
            ds4_kv_tq4_nearest(scratch[i], centroid_scale);
        packed[i / 2u] |= (uint8_t)(
            code << ((i & 1u) ? 4u : 0u));
    }
    ds4_kv_store_u16_le(packed + data_bytes, ds4_kv_f32_to_f16(norm));
    return true;
}

bool ds4_kv_tq4_key_decode_reference(
        float         *output,
        uint32_t       vector_dim,
        const uint8_t *packed,
        size_t         packed_bytes,
        uint32_t       seed,
        float         *scratch,
        size_t         scratch_count) {
    const size_t required = ds4_kv_tq4_key_bytes(vector_dim);
    if (!output || !packed || !scratch || required == 0 ||
        packed_bytes < required || scratch_count < vector_dim ||
        !ds4_kv_is_power_of_two(vector_dim)) {
        return false;
    }

    const size_t data_bytes = ((size_t)vector_dim + 1u) / 2u;
    const float norm =
        ds4_kv_f16_to_f32(ds4_kv_load_u16_le(packed + data_bytes));
    const float centroid_scale = 1.0f / sqrtf((float)vector_dim);
    for (uint32_t i = 0; i < vector_dim; i++) {
        const uint8_t byte = packed[i / 2u];
        const uint8_t code = (uint8_t)(
            (byte >> ((i & 1u) ? 4u : 0u)) & UINT8_C(0x0f));
        scratch[i] =
            ds4_kv_tq4_centroid_std_normal[code] * centroid_scale;
    }
    ds4_kv_wht(scratch, vector_dim);
    for (uint32_t i = 0; i < vector_dim; i++) {
        output[i] = scratch[i] * ds4_kv_sign(seed, i) * norm;
    }
    return true;
}

bool ds4_kv_tq4_value_encode_reference(
        uint8_t       *packed,
        size_t         packed_bytes,
        const float   *input,
        uint32_t       vector_dim) {
    const size_t required = ds4_kv_tq4_value_bytes(vector_dim);
    if (!packed || !input || required == 0 || packed_bytes < required) {
        return false;
    }

    float minimum = FLT_MAX;
    float maximum = -FLT_MAX;
    for (uint32_t i = 0; i < vector_dim; i++) {
        if (!isfinite(input[i])) return false;
        minimum = fminf(minimum, input[i]);
        maximum = fmaxf(maximum, input[i]);
    }
    float scale = (maximum - minimum) / 15.0f;
    if (!(scale > 1.0e-8f)) scale = 1.0e-8f;

    const size_t data_bytes = ((size_t)vector_dim + 1u) / 2u;
    memset(packed, 0, data_bytes);
    for (uint32_t i = 0; i < vector_dim; i++) {
        long code = lroundf((input[i] - minimum) / scale);
        if (code < 0) code = 0;
        if (code > 15) code = 15;
        packed[i / 2u] |= (uint8_t)(
            (uint8_t)code << ((i & 1u) ? 4u : 0u));
    }
    ds4_kv_store_u16_le(
        packed + data_bytes, ds4_kv_f32_to_f16(scale));
    ds4_kv_store_u16_le(
        packed + data_bytes + sizeof(uint16_t),
        ds4_kv_f32_to_f16(minimum));
    return true;
}

bool ds4_kv_tq4_value_decode_reference(
        float         *output,
        uint32_t       vector_dim,
        const uint8_t *packed,
        size_t         packed_bytes) {
    const size_t required = ds4_kv_tq4_value_bytes(vector_dim);
    if (!output || !packed || required == 0 || packed_bytes < required) {
        return false;
    }

    const size_t data_bytes = ((size_t)vector_dim + 1u) / 2u;
    const float scale =
        ds4_kv_f16_to_f32(ds4_kv_load_u16_le(packed + data_bytes));
    const float minimum = ds4_kv_f16_to_f32(ds4_kv_load_u16_le(
        packed + data_bytes + sizeof(uint16_t)));
    if (!isfinite(scale) || !isfinite(minimum) || scale < 0.0f) return false;

    for (uint32_t i = 0; i < vector_dim; i++) {
        const uint8_t byte = packed[i / 2u];
        const uint8_t code = (uint8_t)(
            (byte >> ((i & 1u) ? 4u : 0u)) & UINT8_C(0x0f));
        output[i] = minimum + scale * code;
    }
    return true;
}
