#include "ds4_profile.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

static bool word_is(const char *value, const char *word) {
    return value && word && strcasecmp(value, word) == 0;
}

bool ds4_profile_toggle_parse(const char *value, bool *out) {
    if (!value || !out) return false;
    if (!value[0] || word_is(value, "1") || word_is(value, "true") ||
        word_is(value, "on") || word_is(value, "yes")) {
        *out = true;
        return true;
    }
    if (word_is(value, "0") || word_is(value, "false") ||
        word_is(value, "off") || word_is(value, "no")) {
        *out = false;
        return true;
    }
    return false;
}

bool ds4_profile_router_ahead_parse(
        const char *value,
        ds4_glm_router_ahead_mode *out) {
    if (!value || !out) return false;
    if (word_is(value, "0") || word_is(value, "false") ||
        word_is(value, "off") || word_is(value, "no")) {
        *out = DS4_GLM_ROUTER_AHEAD_OFF;
        return true;
    }
    if (!value[0] || word_is(value, "1") || word_is(value, "true") ||
        word_is(value, "on") || word_is(value, "yes") ||
        word_is(value, "advisory")) {
        *out = DS4_GLM_ROUTER_AHEAD_ADVISORY;
        return true;
    }
    if (word_is(value, "2") || word_is(value, "install")) {
        *out = DS4_GLM_ROUTER_AHEAD_INSTALL;
        return true;
    }
    return false;
}

static bool env_toggle(const char *name, bool *out) {
    const char *value = getenv(name);
    if (!value) return false;
    if (ds4_profile_toggle_parse(value, out)) return true;
    fprintf(stderr,
            "ds4: ignoring invalid %s=%s (expected on/off or 1/0)\n",
            name,
            value);
    return false;
}

static bool env_router_mode(
        const char *name,
        ds4_glm_router_ahead_mode *out) {
    const char *value = getenv(name);
    if (!value) return false;
    if (ds4_profile_router_ahead_parse(value, out)) return true;
    fprintf(stderr,
            "ds4: ignoring invalid %s=%s (expected off/advisory/install)\n",
            name,
            value);
    return false;
}

static uint32_t env_lookahead(const char *name, uint32_t fallback) {
    const char *value = getenv(name);
    if (!value || !value[0]) return fallback;
    char *end = NULL;
    errno = 0;
    const long parsed = strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' ||
        parsed < 1 || parsed > 4) {
        fprintf(stderr,
                "ds4: ignoring invalid %s=%s (expected 1..4)\n",
                name,
                value);
        return fallback;
    }
    return (uint32_t)parsed;
}

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

    /* These are diagnostic A/B overrides, resolved once at startup.  An
     * explicit zero means false instead of the older presence-only behavior. */
    bool toggle = false;
    if (env_toggle("DS4_METAL_ENABLE_GLM_INDEXED_PREFILL_PREPARE", &toggle)) {
        out->glm_indexed_prefill_prepare = toggle;
    }
    if (env_toggle("DS4_METAL_DISABLE_GLM_INDEXED_PREFILL_PREPARE", &toggle) &&
        toggle) {
        out->glm_indexed_prefill_prepare = false;
    }

    (void)env_router_mode("DS4_GLM_ROUTER_AHEAD_PREFETCH",
                          &out->glm_router_ahead);
    out->glm_router_ahead_lookahead = env_lookahead(
            "DS4_GLM_ROUTER_AHEAD_LOOKAHEAD",
            out->glm_router_ahead_lookahead);

    if (env_toggle("DS4_METAL_ENABLE_STREAMING_EXPERT_READAHEAD", &toggle)) {
        out->streaming_expert_readahead = toggle;
    }
    if (env_toggle("DS4_METAL_DISABLE_STREAMING_EXPERT_READAHEAD", &toggle) &&
        toggle) {
        out->streaming_expert_readahead = false;
    }
}
