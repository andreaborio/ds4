#include "ds4_profile.h"

void ds4_metal_ssd_profile_resolve(
        bool is_metal,
        bool is_glm,
        bool ssd_streaming,
        ds4_metal_ssd_profile *out) {
    if (!out) return;

    const bool glm_gold = is_metal && is_glm && ssd_streaming;
    *out = (ds4_metal_ssd_profile) {
        .name = glm_gold ? "glm52-metal-ssd-gold-v1" : "default",
        .glm_indexed_prefill_prepare = glm_gold,
        .glm_router_ahead = glm_gold ?
            DS4_GLM_ROUTER_AHEAD_ADVISORY : DS4_GLM_ROUTER_AHEAD_OFF,
        .glm_router_ahead_lookahead = 1,
        .streaming_expert_readahead = !glm_gold,
    };
}
