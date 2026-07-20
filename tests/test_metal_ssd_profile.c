#include "ds4_profile.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

static void clear_profile_env(void) {
    unsetenv("DS4_METAL_ENABLE_GLM_INDEXED_PREFILL_PREPARE");
    unsetenv("DS4_METAL_DISABLE_GLM_INDEXED_PREFILL_PREPARE");
    unsetenv("DS4_GLM_ROUTER_AHEAD_PREFETCH");
    unsetenv("DS4_GLM_ROUTER_AHEAD_LOOKAHEAD");
    unsetenv("DS4_METAL_ENABLE_STREAMING_EXPERT_READAHEAD");
    unsetenv("DS4_METAL_DISABLE_STREAMING_EXPERT_READAHEAD");
}

int main(void) {
    bool toggle = false;
    assert(ds4_profile_toggle_parse("0", &toggle) && !toggle);
    assert(ds4_profile_toggle_parse("false", &toggle) && !toggle);
    assert(ds4_profile_toggle_parse("ON", &toggle) && toggle);
    assert(!ds4_profile_toggle_parse("sometimes", &toggle));

    ds4_glm_router_ahead_mode router = DS4_GLM_ROUTER_AHEAD_OFF;
    assert(ds4_profile_router_ahead_parse("off", &router) &&
           router == DS4_GLM_ROUTER_AHEAD_OFF);
    assert(ds4_profile_router_ahead_parse("1", &router) &&
           router == DS4_GLM_ROUTER_AHEAD_ADVISORY);
    assert(!ds4_profile_router_ahead_parse("install", &router));

    clear_profile_env();
    ds4_metal_ssd_profile profile;
    ds4_metal_ssd_profile_resolve(true, true, false, &profile);
    assert(!profile.glm_indexed_prefill_prepare);
    assert(profile.glm_router_ahead == DS4_GLM_ROUTER_AHEAD_OFF);
    assert(profile.streaming_expert_readahead);

    ds4_metal_ssd_profile_resolve(true, true, true, &profile);
    assert(profile.glm_indexed_prefill_prepare);
    assert(profile.glm_router_ahead == DS4_GLM_ROUTER_AHEAD_ADVISORY);
    assert(profile.glm_router_ahead_lookahead == 1);
    assert(!profile.streaming_expert_readahead);

    setenv("DS4_METAL_ENABLE_GLM_INDEXED_PREFILL_PREPARE", "0", 1);
    setenv("DS4_GLM_ROUTER_AHEAD_PREFETCH", "false", 1);
    setenv("DS4_METAL_ENABLE_STREAMING_EXPERT_READAHEAD", "on", 1);
    setenv("DS4_GLM_ROUTER_AHEAD_LOOKAHEAD", "3", 1);
    ds4_metal_ssd_profile_resolve(true, true, true, &profile);
    assert(!profile.glm_indexed_prefill_prepare);
    assert(profile.glm_router_ahead == DS4_GLM_ROUTER_AHEAD_OFF);
    assert(profile.glm_router_ahead_lookahead == 3);
    assert(profile.streaming_expert_readahead);

    clear_profile_env();
    ds4_metal_ssd_profile_resolve(true, false, true, &profile);
    assert(!profile.glm_indexed_prefill_prepare);
    assert(profile.glm_router_ahead == DS4_GLM_ROUTER_AHEAD_OFF);
    assert(profile.streaming_expert_readahead);

    setenv("DS4_METAL_DISABLE_STREAMING_EXPERT_READAHEAD", "0", 1);
    ds4_metal_ssd_profile_resolve(true, false, true, &profile);
    assert(profile.streaming_expert_readahead);
    setenv("DS4_METAL_DISABLE_STREAMING_EXPERT_READAHEAD", "true", 1);
    ds4_metal_ssd_profile_resolve(true, false, true, &profile);
    assert(!profile.streaming_expert_readahead);

    clear_profile_env();
    puts("metal SSD profile parser: ok");
    return 0;
}
