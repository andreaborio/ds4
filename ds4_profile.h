#ifndef DS4_PROFILE_H
#define DS4_PROFILE_H

#include <stdbool.h>
#include <stdint.h>

typedef enum {
    DS4_GLM_ROUTER_AHEAD_OFF = 0,
    DS4_GLM_ROUTER_AHEAD_ADVISORY,
} ds4_glm_router_ahead_mode;

typedef struct {
    const char *name;
    bool glm_indexed_prefill_prepare;
    ds4_glm_router_ahead_mode glm_router_ahead;
    uint32_t glm_router_ahead_lookahead;
    bool streaming_expert_readahead;
} ds4_metal_ssd_profile;

void ds4_metal_ssd_profile_resolve(
        bool is_metal,
        bool is_glm,
        bool ssd_streaming,
        ds4_metal_ssd_profile *out);

#endif
