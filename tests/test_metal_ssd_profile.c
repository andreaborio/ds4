#include "ds4_profile.h"

#include <assert.h>
#include <stdio.h>

int main(void) {
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

    ds4_metal_ssd_profile_resolve(true, false, true, &profile);
    assert(!profile.glm_indexed_prefill_prepare);
    assert(profile.glm_router_ahead == DS4_GLM_ROUTER_AHEAD_OFF);
    assert(profile.glm_router_ahead_lookahead == 1);
    assert(profile.streaming_expert_readahead);

    ds4_metal_ssd_profile_resolve(false, true, true, &profile);
    assert(!profile.glm_indexed_prefill_prepare);
    assert(profile.glm_router_ahead == DS4_GLM_ROUTER_AHEAD_OFF);
    assert(profile.glm_router_ahead_lookahead == 1);
    assert(profile.streaming_expert_readahead);

    puts("metal SSD profile resolver: ok");
    return 0;
}
