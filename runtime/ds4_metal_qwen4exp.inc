/* Qwen4Exp Phase-5 Metal correctness partition.
 *
 * This file is a textual implementation partition for ds4_metal.m.  It is
 * intentionally inert until the including translation unit calls the
 * prefixed compile/encode helpers below.  It declares no public interface and
 * contains no family dispatch, support selector, codec, SSD, or graph policy.
 *
 * Required graph order for the eventual caller:
 *   embedding -> optional PLE(add before the layer read) -> attention GR
 *   prepare/block/apply -> FFN GR prepare/MoE/apply -> next layer -> final GR
 *   mix -> unnormalized output head.
 * Stateful encoding takes one private snapshot of the complete recurrent row;
 * GDN-conv and PLE-conv bind disjoint offsets into that same snapshot (never a
 * second gather that can clobber the first write-back). QSA raw keys use the
 * main KV slot map, pool complete logical groups before norm/RoPE, and leave the
 * incomplete tail unpooled. Nothing is published before completed status and
 * all per-kernel status words are clean.
 */

typedef struct {
    uint32_t n_token;
    uint32_t n_row;
    uint32_t n_stream;
    uint32_t dim;
} ds4_metal_qwen4exp_embedding_args;

typedef struct {
    uint32_t n_token;
    uint32_t n_stream;
    uint32_t dim;
    uint32_t rank;
    float    epsilon;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
} ds4_metal_qwen4exp_gr_args;

typedef struct {
    uint32_t n_token;
    uint32_t n_channel;
    uint32_t kernel_size;
    uint32_t dilation;
    uint32_t n_sequence;
    uint32_t reserved;
} ds4_metal_qwen4exp_conv_args;

typedef struct {
    uint32_t n_token;
    uint32_t n_value_head;
} ds4_metal_qwen4exp_controls_args;

typedef struct {
    uint32_t n_token;
    uint32_t n_key_head;
    uint32_t n_value_head;
    uint32_t key_dim;
    uint32_t value_dim;
    uint32_t reserved;
} ds4_metal_qwen4exp_gdn_args;

typedef struct {
    uint32_t n_token;
    uint32_t n_expert;
    uint32_t n_selected;
    uint32_t reserved;
} ds4_metal_qwen4exp_router_args;

typedef struct {
    uint32_t n_token;
    uint32_t n_expert;
    uint32_t n_selected;
    uint32_t input_dim;
    uint32_t expert_dim;
    uint32_t output_dim;
    uint32_t reserved0;
    uint32_t reserved1;
} ds4_metal_qwen4exp_moe_args;

typedef struct {
    uint32_t n_group;
    uint32_t compression;
    uint32_t head_dim;
    uint32_t n_rot;
    float    theta;
    float    epsilon;
    uint32_t position0;
    uint32_t n_slot;
} ds4_metal_qwen4exp_qsa_group_args;

typedef struct {
    uint32_t n_query;
    uint32_t n_query_head;
    uint32_t head_dim;
    uint32_t n_visible_max;
    uint32_t compression;
    uint32_t group_budget;
    uint32_t max_width;
    uint32_t reserved;
} ds4_metal_qwen4exp_qsa_select_args;

typedef struct {
    uint32_t n_query;
    uint32_t n_query_head;
    uint32_t n_kv_head;
    uint32_t head_dim;
    uint32_t n_key;
    uint32_t max_selected;
    uint32_t reserved0;
    uint32_t reserved1;
} ds4_metal_qwen4exp_qsa_attention_args;

typedef struct {
    uint32_t n_token;
    uint32_t n_head;
    uint32_t row_dim;
    uint32_t n_row;
} ds4_metal_qwen4exp_ple_gather_args;

typedef struct {
    uint32_t n_token;
    uint32_t n_stream;
    uint32_t dim;
    uint32_t reserved;
} ds4_metal_qwen4exp_ple_gate_args;

typedef struct {
    uint32_t n_token;
    uint32_t input_dim;
    uint32_t output_dim;
    uint32_t reserved;
} ds4_metal_qwen4exp_head_args;

_Static_assert(sizeof(ds4_metal_qwen4exp_embedding_args) == 16,
               "Qwen4Exp embedding Metal ABI drift");
_Static_assert(sizeof(ds4_metal_qwen4exp_gr_args) == 32,
               "Qwen4Exp GR Metal ABI drift");
_Static_assert(sizeof(ds4_metal_qwen4exp_conv_args) == 24,
               "Qwen4Exp conv Metal ABI drift");
_Static_assert(sizeof(ds4_metal_qwen4exp_controls_args) == 8,
               "Qwen4Exp controls Metal ABI drift");
_Static_assert(sizeof(ds4_metal_qwen4exp_gdn_args) == 24,
               "Qwen4Exp GDN Metal ABI drift");
_Static_assert(sizeof(ds4_metal_qwen4exp_router_args) == 16,
               "Qwen4Exp router Metal ABI drift");
_Static_assert(sizeof(ds4_metal_qwen4exp_moe_args) == 32,
               "Qwen4Exp MoE Metal ABI drift");
_Static_assert(sizeof(ds4_metal_qwen4exp_qsa_group_args) == 32,
               "Qwen4Exp QSA-group Metal ABI drift");
_Static_assert(sizeof(ds4_metal_qwen4exp_qsa_select_args) == 32,
               "Qwen4Exp QSA-select Metal ABI drift");
_Static_assert(sizeof(ds4_metal_qwen4exp_qsa_attention_args) == 32,
               "Qwen4Exp QSA-attention Metal ABI drift");
_Static_assert(sizeof(ds4_metal_qwen4exp_ple_gather_args) == 16,
               "Qwen4Exp PLE-gather Metal ABI drift");
_Static_assert(sizeof(ds4_metal_qwen4exp_ple_gate_args) == 16,
               "Qwen4Exp PLE-gate Metal ABI drift");
_Static_assert(sizeof(ds4_metal_qwen4exp_head_args) == 16,
               "Qwen4Exp head Metal ABI drift");

typedef NS_ENUM(uint32_t, ds4_metal_qwen4exp_kernel) {
    DS4_METAL_QWEN4EXP_EMBEDDING = 0,
    DS4_METAL_QWEN4EXP_GR_PREPARE,
    DS4_METAL_QWEN4EXP_GR_APPLY,
    DS4_METAL_QWEN4EXP_GR_FINAL,
    DS4_METAL_QWEN4EXP_CONV,
    DS4_METAL_QWEN4EXP_CONTROLS,
    DS4_METAL_QWEN4EXP_GDN,
    DS4_METAL_QWEN4EXP_ROUTER,
    DS4_METAL_QWEN4EXP_MOE,
    DS4_METAL_QWEN4EXP_QSA_GROUP,
    DS4_METAL_QWEN4EXP_QSA_SELECT,
    DS4_METAL_QWEN4EXP_QSA_ATTENTION,
    DS4_METAL_QWEN4EXP_PLE_GATHER,
    DS4_METAL_QWEN4EXP_PLE_GATE,
    DS4_METAL_QWEN4EXP_HEAD,
    DS4_METAL_QWEN4EXP_KERNEL_COUNT,
};

typedef struct {
    __strong id<MTLLibrary> library;
    __strong id<MTLComputePipelineState>
        pipeline[DS4_METAL_QWEN4EXP_KERNEL_COUNT];
} ds4_metal_qwen4exp_cache;

typedef struct {
    __strong id<MTLBuffer> buffer;
    NSUInteger offset;
    NSUInteger minimum_length;
} ds4_metal_qwen4exp_binding;

typedef struct {
    __strong id<MTLBuffer> public_state;
    __strong id<MTLBuffer> private_state;
    NSUInteger byte_count;
} ds4_metal_qwen4exp_state_transaction;

static inline NSString *ds4_metal_qwen4exp_kernel_name(
        ds4_metal_qwen4exp_kernel kernel) {
    switch (kernel) {
        case DS4_METAL_QWEN4EXP_EMBEDDING:
            return @"kernel_qwen4exp_embedding_four_stream_f32";
        case DS4_METAL_QWEN4EXP_GR_PREPARE:
            return @"kernel_qwen4exp_gr_prepare_f32";
        case DS4_METAL_QWEN4EXP_GR_APPLY:
            return @"kernel_qwen4exp_gr_apply_f32";
        case DS4_METAL_QWEN4EXP_GR_FINAL:
            return @"kernel_qwen4exp_gr_final_mix_f32";
        case DS4_METAL_QWEN4EXP_CONV:
            return @"kernel_qwen4exp_depthwise_conv_sequence_silu_f32";
        case DS4_METAL_QWEN4EXP_CONTROLS:
            return @"kernel_qwen4exp_gdn_controls_f32";
        case DS4_METAL_QWEN4EXP_GDN:
            return @"kernel_qwen4exp_gdn_sequence_f32";
        case DS4_METAL_QWEN4EXP_ROUTER:
            return @"kernel_qwen4exp_router_softmax_top10_f32";
        case DS4_METAL_QWEN4EXP_MOE:
            return @"kernel_qwen4exp_moe_resident_f32";
        case DS4_METAL_QWEN4EXP_QSA_GROUP:
            return @"kernel_qwen4exp_qsa_group_keys_f32";
        case DS4_METAL_QWEN4EXP_QSA_SELECT:
            return @"kernel_qwen4exp_qsa_select_positions_f32";
        case DS4_METAL_QWEN4EXP_QSA_ATTENTION:
            return @"kernel_qwen4exp_qsa_attention_f32";
        case DS4_METAL_QWEN4EXP_PLE_GATHER:
            return @"kernel_qwen4exp_ple_gather_f32";
        case DS4_METAL_QWEN4EXP_PLE_GATE:
            return @"kernel_qwen4exp_ple_gate_f32";
        case DS4_METAL_QWEN4EXP_HEAD:
            return @"kernel_qwen4exp_head_f32";
        case DS4_METAL_QWEN4EXP_KERNEL_COUNT:
            break;
    }
    return nil;
}

static inline NSUInteger ds4_metal_qwen4exp_argument_size(
        ds4_metal_qwen4exp_kernel kernel) {
    switch (kernel) {
        case DS4_METAL_QWEN4EXP_EMBEDDING:
            return sizeof(ds4_metal_qwen4exp_embedding_args);
        case DS4_METAL_QWEN4EXP_GR_PREPARE:
        case DS4_METAL_QWEN4EXP_GR_APPLY:
        case DS4_METAL_QWEN4EXP_GR_FINAL:
            return sizeof(ds4_metal_qwen4exp_gr_args);
        case DS4_METAL_QWEN4EXP_CONV:
            return sizeof(ds4_metal_qwen4exp_conv_args);
        case DS4_METAL_QWEN4EXP_CONTROLS:
            return sizeof(ds4_metal_qwen4exp_controls_args);
        case DS4_METAL_QWEN4EXP_GDN:
            return sizeof(ds4_metal_qwen4exp_gdn_args);
        case DS4_METAL_QWEN4EXP_ROUTER:
            return sizeof(ds4_metal_qwen4exp_router_args);
        case DS4_METAL_QWEN4EXP_MOE:
            return sizeof(ds4_metal_qwen4exp_moe_args);
        case DS4_METAL_QWEN4EXP_QSA_GROUP:
            return sizeof(ds4_metal_qwen4exp_qsa_group_args);
        case DS4_METAL_QWEN4EXP_QSA_SELECT:
            return sizeof(ds4_metal_qwen4exp_qsa_select_args);
        case DS4_METAL_QWEN4EXP_QSA_ATTENTION:
            return sizeof(ds4_metal_qwen4exp_qsa_attention_args);
        case DS4_METAL_QWEN4EXP_PLE_GATHER:
            return sizeof(ds4_metal_qwen4exp_ple_gather_args);
        case DS4_METAL_QWEN4EXP_PLE_GATE:
            return sizeof(ds4_metal_qwen4exp_ple_gate_args);
        case DS4_METAL_QWEN4EXP_HEAD:
            return sizeof(ds4_metal_qwen4exp_head_args);
        case DS4_METAL_QWEN4EXP_KERNEL_COUNT:
            break;
    }
    return 0;
}

static inline NSUInteger ds4_metal_qwen4exp_binding_count(
        ds4_metal_qwen4exp_kernel kernel) {
    switch (kernel) {
        case DS4_METAL_QWEN4EXP_EMBEDDING: return 4u;
        case DS4_METAL_QWEN4EXP_GR_PREPARE: return 8u;
        case DS4_METAL_QWEN4EXP_GR_APPLY: return 4u;
        case DS4_METAL_QWEN4EXP_GR_FINAL: return 6u;
        case DS4_METAL_QWEN4EXP_CONV: return 5u;
        case DS4_METAL_QWEN4EXP_CONTROLS: return 7u;
        case DS4_METAL_QWEN4EXP_GDN: return 8u;
        case DS4_METAL_QWEN4EXP_ROUTER: return 4u;
        case DS4_METAL_QWEN4EXP_MOE: return 12u;
        case DS4_METAL_QWEN4EXP_QSA_GROUP: return 6u;
        case DS4_METAL_QWEN4EXP_QSA_SELECT: return 6u;
        case DS4_METAL_QWEN4EXP_QSA_ATTENTION: return 7u;
        case DS4_METAL_QWEN4EXP_PLE_GATHER: return 4u;
        case DS4_METAL_QWEN4EXP_PLE_GATE: return 5u;
        case DS4_METAL_QWEN4EXP_HEAD: return 4u;
        case DS4_METAL_QWEN4EXP_KERNEL_COUNT: break;
    }
    return 0u;
}

static inline void ds4_metal_qwen4exp_set_error(
        NSError **error_out,
        NSInteger code,
        NSString *description) {
    if (!error_out) return;
    *error_out = [NSError errorWithDomain:@"org.ds4.qwen4exp.metal"
                                     code:code
                                 userInfo:@{NSLocalizedDescriptionKey:
                                                description != nil
                                                    ? description
                                                    : @"unknown"}];
}

static inline bool ds4_metal_qwen4exp_compile_cache(
        id<MTLDevice> device,
        NSString *source,
        ds4_metal_qwen4exp_cache *cache,
        NSError **error_out) {
    if (!device || source.length == 0u || !cache) {
        ds4_metal_qwen4exp_set_error(error_out, 1,
                                     @"missing device, source, or cache");
        return false;
    }
    MTLCompileOptions *options = [MTLCompileOptions new];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    options.fastMathEnabled = NO;
#pragma clang diagnostic pop
    NSError *error = nil;
    id<MTLLibrary> library =
        [device newLibraryWithSource:source options:options error:&error];
    if (!library) {
        if (error_out) *error_out = error;
        return false;
    }
    id<MTLComputePipelineState> built[DS4_METAL_QWEN4EXP_KERNEL_COUNT];
    for (uint32_t index = 0u;
         index < (uint32_t)DS4_METAL_QWEN4EXP_KERNEL_COUNT; index++) {
        const ds4_metal_qwen4exp_kernel kernel =
            (ds4_metal_qwen4exp_kernel)index;
        NSString *name = ds4_metal_qwen4exp_kernel_name(kernel);
        id<MTLFunction> function = [library newFunctionWithName:name];
        if (!function) {
            ds4_metal_qwen4exp_set_error(
                error_out, 2, [NSString stringWithFormat:
                    @"missing Metal function %@", name]);
            return false;
        }
        built[index] =
            [device newComputePipelineStateWithFunction:function error:&error];
        if (!built[index]) {
            if (error_out) *error_out = error;
            return false;
        }
    }
    cache->library = library;
    for (uint32_t index = 0u;
         index < (uint32_t)DS4_METAL_QWEN4EXP_KERNEL_COUNT; index++) {
        cache->pipeline[index] = built[index];
    }
    return true;
}

static inline bool ds4_metal_qwen4exp_checked_binding(
        const ds4_metal_qwen4exp_binding *binding) {
    if (!binding || !binding->buffer) return false;
    if (binding->offset > binding->buffer.length) return false;
    return binding->minimum_length <= binding->buffer.length - binding->offset;
}

/* Encodes only after proving argument ABI, buffer count, every offset/span,
 * grid, and threadgroup limits.  Completion status remains the caller's
 * transaction boundary; this helper never commits or publishes state. */
static inline bool ds4_metal_qwen4exp_encode(
        const ds4_metal_qwen4exp_cache *cache,
        ds4_metal_qwen4exp_kernel kernel,
        id<MTLCommandBuffer> command,
        const void *arguments,
        NSUInteger argument_size,
        const ds4_metal_qwen4exp_binding *bindings,
        NSUInteger n_binding,
        MTLSize grid,
        MTLSize threads,
        NSError **error_out) {
    if (!cache || kernel >= DS4_METAL_QWEN4EXP_KERNEL_COUNT || !command ||
        !arguments || argument_size != ds4_metal_qwen4exp_argument_size(kernel) ||
        !bindings || n_binding != ds4_metal_qwen4exp_binding_count(kernel) ||
        grid.width == 0u || grid.height == 0u ||
        grid.depth == 0u || threads.width == 0u || threads.height == 0u ||
        threads.depth == 0u) {
        ds4_metal_qwen4exp_set_error(error_out, 3,
                                     @"invalid Metal encode arguments");
        return false;
    }
    id<MTLComputePipelineState> pipeline = cache->pipeline[kernel];
    if (!pipeline || threads.width > pipeline.maxTotalThreadsPerThreadgroup ||
        threads.height > pipeline.maxTotalThreadsPerThreadgroup / threads.width ||
        threads.depth > pipeline.maxTotalThreadsPerThreadgroup /
            (threads.width * threads.height)) {
        ds4_metal_qwen4exp_set_error(error_out, 4,
                                     @"invalid Qwen4Exp pipeline/threadgroup");
        return false;
    }
    for (NSUInteger index = 0u; index < n_binding; index++) {
        if (!ds4_metal_qwen4exp_checked_binding(&bindings[index])) {
            ds4_metal_qwen4exp_set_error(
                error_out, 5, [NSString stringWithFormat:
                    @"Qwen4Exp buffer %lu is out of bounds",
                    (unsigned long)index]);
            return false;
        }
    }
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (!encoder) {
        ds4_metal_qwen4exp_set_error(error_out, 6,
                                     @"could not create compute encoder");
        return false;
    }
    [encoder setComputePipelineState:pipeline];
    [encoder setBytes:arguments length:argument_size atIndex:0];
    for (NSUInteger index = 0u; index < n_binding; index++) {
        [encoder setBuffer:bindings[index].buffer
                    offset:bindings[index].offset
                   atIndex:index + 1u];
    }
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return true;
}

static inline bool ds4_metal_qwen4exp_finite_f32(
        const float *values,
        size_t count) {
    if (!values || count == 0u) return false;
    for (size_t index = 0u; index < count; index++)
        if (!isfinite(values[index])) return false;
    return true;
}

/* Phase-5 state is shared/managed F32 memory so a fresh private transaction
 * can be seeded byte-for-byte. allocation_ceiling is normally the device or
 * graph budget and is injectable in the standalone OOM rollback test. */
static inline bool ds4_metal_qwen4exp_prepare_state_transaction(
        id<MTLDevice> device,
        id<MTLBuffer> public_state,
        NSUInteger byte_count,
        NSUInteger allocation_ceiling,
        ds4_metal_qwen4exp_state_transaction *transaction,
        NSError **error_out) {
    if (!device || !public_state || !transaction || byte_count == 0u ||
        public_state.length < byte_count || byte_count > allocation_ceiling ||
        byte_count > device.maxBufferLength ||
        (public_state.storageMode != MTLStorageModeShared &&
         public_state.storageMode != MTLStorageModeManaged)) {
        ds4_metal_qwen4exp_set_error(
            error_out, 7, @"Qwen4Exp private state allocation rejected");
        return false;
    }
    id<MTLBuffer> private_state =
        [device newBufferWithLength:byte_count
                            options:public_state.resourceOptions];
    if (!private_state) {
        ds4_metal_qwen4exp_set_error(
            error_out, 8, @"Qwen4Exp private state allocation failed");
        return false;
    }
    memcpy(private_state.contents, public_state.contents, byte_count);
    if (private_state.storageMode == MTLStorageModeManaged) {
        [private_state didModifyRange:NSMakeRange(0u, byte_count)];
    }
    transaction->public_state = public_state;
    transaction->private_state = private_state;
    transaction->byte_count = byte_count;
    return true;
}

/* The graph binds private_state to stateful kernels and swaps only after a
 * completed, error-free command.  A false completion leaves public_state
 * byte-for-byte and identity-wise unchanged. */
static inline bool ds4_metal_qwen4exp_publish_state(
        ds4_metal_qwen4exp_state_transaction *transaction,
        bool command_succeeded,
        const uint32_t *status,
        size_t status_count) {
    if (!transaction || !transaction->public_state ||
        !transaction->private_state || transaction->byte_count == 0u ||
        transaction->public_state == transaction->private_state ||
        transaction->public_state.length < transaction->byte_count ||
        transaction->private_state.length < transaction->byte_count ||
        !command_succeeded || !status || status_count == 0u) {
        return false;
    }
    for (size_t index = 0u; index < status_count; index++)
        if (status[index] != 0u) return false;
    id<MTLBuffer> previous = transaction->public_state;
    transaction->public_state = transaction->private_state;
    transaction->private_state = previous;
    return true;
}

static inline bool ds4_metal_qwen4exp_publish_state_after_command(
        ds4_metal_qwen4exp_state_transaction *transaction,
        id<MTLCommandBuffer> command,
        const uint32_t *status,
        size_t status_count) {
    const bool completed = command != nil &&
        command.status == MTLCommandBufferStatusCompleted &&
        command.error == nil;
    return ds4_metal_qwen4exp_publish_state(
        transaction, completed, status, status_count);
}
