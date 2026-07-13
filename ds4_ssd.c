#include "ds4_ssd.h"

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif

static const uint64_t DS4_GIB = 1024ull * 1024ull * 1024ull;

static uint64_t saturating_add_u64(uint64_t a, uint64_t b) {
    return a > UINT64_MAX - b ? UINT64_MAX : a + b;
}

const char *ds4_residency_mode_name(ds4_residency_mode mode) {
    switch (mode) {
    case DS4_RESIDENCY_AUTO:     return "auto";
    case DS4_RESIDENCY_RESIDENT: return "resident";
    case DS4_RESIDENCY_SSD:      return "ssd";
    }
    return "invalid";
}

const char *ds4_residency_reason_name(ds4_residency_reason reason) {
    switch (reason) {
    case DS4_RESIDENCY_REASON_EXPLICIT_RESIDENT:
        return "explicit resident override";
    case DS4_RESIDENCY_REASON_EXPLICIT_SSD:
        return "explicit SSD override";
    case DS4_RESIDENCY_REASON_NON_METAL_AUTO:
        return "AUTO currently preserves resident mode outside Metal";
    case DS4_RESIDENCY_REASON_METAL_FITS:
        return "estimated Metal residency plan fits the conservative budget";
    case DS4_RESIDENCY_REASON_METAL_EXCEEDS:
        return "estimated Metal residency plan exceeds the conservative budget";
    case DS4_RESIDENCY_REASON_METAL_BUDGET_UNAVAILABLE:
        return "Metal recommended working-set budget is unavailable";
    case DS4_RESIDENCY_REASON_INSPECT_ONLY:
        return "model inspection defers runtime residency selection";
    }
    return "unknown reason";
}

bool ds4_residency_plan_make(bool                metal_backend,
                             ds4_residency_mode  requested,
                             uint64_t            model_bytes,
                             uint64_t            runtime_bytes,
                             uint64_t            recommended_bytes,
                             uint64_t            external_reserved_bytes,
                             ds4_residency_plan *out) {
    if (!out || requested < DS4_RESIDENCY_AUTO ||
        requested > DS4_RESIDENCY_SSD) {
        return false;
    }

    memset(out, 0, sizeof(*out));
    out->requested = requested;
    out->recommended_bytes = recommended_bytes;
    out->external_reserved_bytes = external_reserved_bytes;
    out->model_bytes = model_bytes;
    out->runtime_bytes = runtime_bytes;

    if (recommended_bytes > external_reserved_bytes) {
        out->budget_bytes = recommended_bytes - external_reserved_bytes;
    }

    if (metal_backend && recommended_bytes != 0) {
        out->headroom_bytes = recommended_bytes / 5u;
        if (out->headroom_bytes < 2u * DS4_GIB) {
            out->headroom_bytes = 2u * DS4_GIB;
        }
    }
    out->required_bytes = saturating_add_u64(model_bytes, runtime_bytes);
    out->required_bytes = saturating_add_u64(out->required_bytes,
                                             out->headroom_bytes);

    if (requested == DS4_RESIDENCY_RESIDENT) {
        out->resolved = DS4_RESIDENCY_RESIDENT;
        out->reason = DS4_RESIDENCY_REASON_EXPLICIT_RESIDENT;
        return true;
    }
    if (requested == DS4_RESIDENCY_SSD) {
        out->resolved = DS4_RESIDENCY_SSD;
        out->reason = DS4_RESIDENCY_REASON_EXPLICIT_SSD;
        return true;
    }
    if (!metal_backend) {
        out->resolved = DS4_RESIDENCY_RESIDENT;
        out->reason = DS4_RESIDENCY_REASON_NON_METAL_AUTO;
        return true;
    }
    if (recommended_bytes == 0) {
        /* AUTO is conservative: if Metal cannot report a budget, do not risk
         * attempting to map a model whose residency has not been proven. */
        out->resolved = DS4_RESIDENCY_SSD;
        out->reason = DS4_RESIDENCY_REASON_METAL_BUDGET_UNAVAILABLE;
        return true;
    }

    if (out->required_bytes <= out->budget_bytes) {
        out->resolved = DS4_RESIDENCY_RESIDENT;
        out->reason = DS4_RESIDENCY_REASON_METAL_FITS;
    } else {
        out->resolved = DS4_RESIDENCY_SSD;
        out->reason = DS4_RESIDENCY_REASON_METAL_EXCEEDS;
    }
    return true;
}

bool ds4_parse_gib_arg(const char *s, uint64_t *bytes) {
    if (bytes) *bytes = 0;
    if (!s || !s[0] || !bytes) return false;

    size_t len = strlen(s);
    if (len > 2 &&
        (s[len - 2] == 'g' || s[len - 2] == 'G') &&
        (s[len - 1] == 'b' || s[len - 1] == 'B')) {
        len -= 2;
    }
    if (len == 0) return false;
    for (size_t i = 0; i < len; i++) {
        if (!isdigit((unsigned char)s[i])) return false;
    }

    char numbuf[32];
    if (len >= sizeof(numbuf)) return false;
    memcpy(numbuf, s, len);
    numbuf[len] = '\0';

    errno = 0;
    unsigned long long v = strtoull(numbuf, NULL, 10);
    if (errno != 0 || v == 0 || v > UINT64_MAX / DS4_GIB) return false;

    *bytes = (uint64_t)v * DS4_GIB;
    return true;
}

bool ds4_parse_streaming_cache_experts_arg(const char *s,
                                           uint32_t   *experts,
                                           uint64_t   *bytes) {
    if (experts) *experts = 0;
    if (bytes) *bytes = 0;
    if (!s || !s[0] || !experts || !bytes) return false;

    const size_t len = strlen(s);
    if (len > 2 &&
        (s[len - 2] == 'g' || s[len - 2] == 'G') &&
        (s[len - 1] == 'b' || s[len - 1] == 'B')) {
        return ds4_parse_gib_arg(s, bytes);
    }

    for (size_t i = 0; i < len; i++) {
        if (!isdigit((unsigned char)s[i])) return false;
    }

    errno = 0;
    unsigned long v = strtoul(s, NULL, 10);
    if (errno != 0 || v == 0 || v > UINT32_MAX) return false;

    *experts = (uint32_t)v;
    return true;
}

uint32_t ds4_ssd_cache_experts_for_byte_budget(uint64_t bytes,
                                               uint64_t per_expert_bytes) {
    if (bytes == 0 || per_expert_bytes == 0) return 0;
    const uint64_t experts = bytes / per_expert_bytes;
    if (experts == 0 || experts > UINT32_MAX) return 0;
    return (uint32_t)experts;
}

bool ds4_ssd_auto_cache_plan(uint64_t            recommended_bytes,
                             uint64_t            non_routed_bytes,
                             uint64_t            per_expert_bytes,
                             uint64_t            max_model_experts,
                             ds4_ssd_cache_plan *out) {
    if (recommended_bytes == 0 || per_expert_bytes == 0) return false;

    const uint64_t model_target_bytes =
        recommended_bytes > UINT64_MAX / 4ull ?
            UINT64_MAX : (recommended_bytes * 4ull) / 5ull;
    return ds4_ssd_cache_plan_for_model_target(model_target_bytes,
                                                non_routed_bytes,
                                                per_expert_bytes,
                                                max_model_experts,
                                                out);
}

bool ds4_ssd_cache_plan_for_model_target(uint64_t            model_target_bytes,
                                         uint64_t            non_routed_bytes,
                                         uint64_t            per_expert_bytes,
                                         uint64_t            max_model_experts,
                                         ds4_ssd_cache_plan *out) {
    if (!out) return false;
    memset(out, 0, sizeof(*out));
    if (model_target_bytes == 0 || per_expert_bytes == 0) return false;

    out->model_target_bytes = model_target_bytes;
    if (out->model_target_bytes <= non_routed_bytes) return false;
    out->cache_bytes = out->model_target_bytes - non_routed_bytes;

    uint64_t cache_experts = out->cache_bytes / per_expert_bytes;
    if (cache_experts == 0) return false;
    if (max_model_experts != 0 && cache_experts > max_model_experts) {
        cache_experts = max_model_experts;
    }
    if (cache_experts > UINT32_MAX) cache_experts = UINT32_MAX;

    out->cache_experts = (uint32_t)cache_experts;
    out->effective_cache_bytes = cache_experts * per_expert_bytes;
    return out->cache_experts != 0;
}

bool ds4_ssd_expert_cache_floor_make(
        uint64_t                    cacheable_routed_layers,
        uint64_t                    experts_per_token,
        uint64_t                    per_expert_bytes,
        ds4_ssd_expert_cache_floor *out) {
    if (!out) return false;
    memset(out, 0, sizeof(*out));
    if (cacheable_routed_layers == 0 ||
        experts_per_token == 0 ||
        per_expert_bytes == 0 ||
        cacheable_routed_layers > UINT64_MAX / experts_per_token) {
        return false;
    }

    const uint64_t working_set =
        cacheable_routed_layers * experts_per_token;
    if (working_set == UINT64_MAX || working_set > UINT64_MAX / 2u) {
        return false;
    }
    const uint64_t minimum_cache = working_set + 1u;
    if (minimum_cache > UINT64_MAX / per_expert_bytes) return false;

    out->working_set_experts = working_set;
    out->minimum_cache_experts = minimum_cache;
    out->minimum_cache_bytes = minimum_cache * per_expert_bytes;
    out->warning_cache_experts = working_set * 2u;
    return true;
}

bool ds4_ssd_adaptive_cache_plan_make(
        const ds4_ssd_host_memory   *memory,
        uint64_t                     runtime_bytes,
        uint64_t                     cacheable_routed_layers,
        uint64_t                     experts_per_token,
        uint64_t                     per_expert_bytes,
        uint64_t                     max_cacheable_experts,
        ds4_ssd_adaptive_cache_plan *out) {
    if (!out) return false;
    memset(out, 0, sizeof(*out));
    if (!memory ||
        memory->physical_bytes == 0 ||
        memory->recommended_bytes == 0 ||
        max_cacheable_experts == 0 ||
        !ds4_ssd_expert_cache_floor_make(cacheable_routed_layers,
                                          experts_per_token,
                                          per_expert_bytes,
                                          &out->floor)) {
        return false;
    }

    const uint64_t file_inactive_bytes =
        memory->inactive_bytes < memory->file_backed_bytes ?
            memory->inactive_bytes : memory->file_backed_bytes;
    uint64_t reclaimable = saturating_add_u64(memory->free_bytes,
                                               memory->purgeable_bytes);
    reclaimable = saturating_add_u64(reclaimable,
                                      file_inactive_bytes / 2u);
    out->reclaimable_bytes = reclaimable;

    const uint64_t two_gib = 2u * DS4_GIB;
    const uint64_t quarter_gib = DS4_GIB / 4u;
    out->current_headroom_bytes = memory->physical_bytes / 16u;
    if (out->current_headroom_bytes < two_gib) {
        out->current_headroom_bytes = two_gib;
    }
    out->pressure_margin_bytes = memory->physical_bytes / 64u;
    if (out->pressure_margin_bytes < quarter_gib) {
        out->pressure_margin_bytes = quarter_gib;
    }
    out->platform_headroom_bytes = memory->physical_bytes / 8u;
    if (out->platform_headroom_bytes < two_gib) {
        out->platform_headroom_bytes = two_gib;
    }

    const uint64_t current_reserve =
        saturating_add_u64(out->current_headroom_bytes,
                           out->pressure_margin_bytes);
    if (reclaimable > current_reserve) {
        out->current_wire_budget_bytes = reclaimable - current_reserve;
    }

    const uint64_t platform_reserve =
        saturating_add_u64(runtime_bytes, out->platform_headroom_bytes);
    if (memory->recommended_bytes > platform_reserve) {
        out->platform_wire_budget_bytes =
            memory->recommended_bytes - platform_reserve;
    }

    out->wire_budget_bytes =
        out->current_wire_budget_bytes < out->platform_wire_budget_bytes ?
            out->current_wire_budget_bytes : out->platform_wire_budget_bytes;
    uint64_t raw_experts = out->wire_budget_bytes / per_expert_bytes;
    if (raw_experts > max_cacheable_experts) {
        raw_experts = max_cacheable_experts;
    }
    if (raw_experts > UINT32_MAX) raw_experts = UINT32_MAX;
    if (raw_experts < out->floor.minimum_cache_experts) return false;

    /* Grow only by complete per-token working sets.  Besides leaving useful
     * pressure slack, this prevents small changes in free pages from buying a
     * cache which still cannot retain one more token's routed-expert cycle. */
    const uint64_t cache_experts =
        1u + out->floor.working_set_experts *
                 ((raw_experts - 1u) / out->floor.working_set_experts);
    if (cache_experts < out->floor.minimum_cache_experts ||
        cache_experts > UINT32_MAX ||
        cache_experts > UINT64_MAX / per_expert_bytes) {
        return false;
    }

    out->cache_experts = (uint32_t)cache_experts;
    out->cache_bytes = cache_experts * per_expert_bytes;
    return true;
}

bool ds4_ssd_working_set_after_reserve(uint64_t  recommended_bytes,
                                       uint64_t  runtime_bytes,
                                       uint64_t  external_reserved_bytes,
                                       uint64_t *available_bytes,
                                       uint64_t *reserved_bytes) {
    if (available_bytes) *available_bytes = 0;
    if (reserved_bytes) *reserved_bytes = 0;
    if (!available_bytes || !reserved_bytes || recommended_bytes == 0) {
        return false;
    }

    const uint64_t reserved = saturating_add_u64(runtime_bytes,
                                                  external_reserved_bytes);
    *reserved_bytes = reserved;
    if (reserved >= recommended_bytes) return false;
    *available_bytes = recommended_bytes - reserved;
    return true;
}

bool ds4_ssd_memory_lock_acquire(ds4_ssd_memory_lock *lock,
                                 uint64_t             bytes) {
    if (!lock) return false;
    lock->ptr = NULL;
    lock->bytes = 0;
    if (bytes == 0) return true;
    if (bytes > (uint64_t)SIZE_MAX) {
        fprintf(stderr,
                "ds4: --simulate-used-memory is too large for this process\n");
        return false;
    }

    void *ptr = mmap(NULL,
                     (size_t)bytes,
                     PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS,
                     -1,
                     0);
    if (ptr == MAP_FAILED) {
        fprintf(stderr,
                "ds4: --simulate-used-memory mmap %.2f GiB failed: %s\n",
                (double)bytes / (double)DS4_GIB,
                strerror(errno));
        return false;
    }

    const long page_long = sysconf(_SC_PAGESIZE);
    const uint64_t page = page_long > 0 ? (uint64_t)page_long : 4096ull;
    const uint64_t chunk_bytes = 256ull * 1024ull * 1024ull;
    volatile unsigned char *p = (volatile unsigned char *)ptr;

    /*
     * Touch and lock in bounded chunks.  A single very large mlock() is harder
     * to diagnose when it fails and can create long uninterruptible VM work on
     * macOS; chunking mirrors the standalone diagnostic utility.
     */
    uint64_t locked = 0;
    for (uint64_t off = 0; off < bytes; off += chunk_bytes) {
        uint64_t len = bytes - off;
        if (len > chunk_bytes) len = chunk_bytes;

        for (uint64_t pos = off; pos < off + len; pos += page) {
            p[pos] = (unsigned char)(pos / page);
        }
        if (len != 0) p[off + len - 1u] = 1;

        if (mlock((void *)(p + off), (size_t)len) != 0) {
            fprintf(stderr,
                    "ds4: --simulate-used-memory mlock failed after %.2f/%.2f GiB: %s\n",
                    (double)locked / (double)DS4_GIB,
                    (double)bytes / (double)DS4_GIB,
                    strerror(errno));
            if (locked != 0) munlock(ptr, (size_t)locked);
            munmap(ptr, (size_t)bytes);
            return false;
        }
        locked += len;
    }

    lock->ptr = ptr;
    lock->bytes = bytes;
    fprintf(stderr,
            "ds4: simulated used memory: locked %.2f GiB before model load\n",
            (double)bytes / (double)DS4_GIB);
    return true;
}

void ds4_ssd_memory_lock_release(ds4_ssd_memory_lock *lock) {
    if (!lock || !lock->ptr || lock->bytes == 0) return;
    munlock(lock->ptr, (size_t)lock->bytes);
    munmap(lock->ptr, (size_t)lock->bytes);
    lock->ptr = NULL;
    lock->bytes = 0;
}
