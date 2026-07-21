/* Qwen Metal residency orchestration. Included by ds4.c after ds4_engine is
 * defined so the monolithic engine keeps only the policy call sites. */
static bool ds4_qwen35_metal_session_context_fits_runtime_plan(
        const ds4_engine   *e,
        int                 ctx_size,
        ds4_context_memory *requested) {
    if (requested) memset(requested, 0, sizeof(*requested));
    if (!e || !requested ||
        !ds4_engine_context_memory_estimate_private(
            e, ctx_size, e->prefill_chunk, requested)) {
        return false;
    }
    return requested->total_bytes <= e->residency_plan.runtime_bytes;
}

#if defined(__APPLE__) && !defined(DS4_NO_GPU)
static bool ds4_engine_apply_qwen35_metal_hardware_policy(
        ds4_engine         *e,
        ds4_ssd_host_memory *memory) {
    if (!e || !memory || e->backend != DS4_BACKEND_METAL ||
        e->model.family != DS4_MODEL_FAMILY_QWEN35_MOE ||
        !ds4_gpu_host_memory_snapshot(memory)) {
        return false;
    }

    ds4_qwen_metal_hardware_policy policy = {0};
    if (!ds4_qwen_metal_hardware_policy_make(
            memory->physical_bytes,
            memory->recommended_bytes,
            &policy) ||
        !ds4_residency_plan_apply_qwen_metal_hardware_policy(
            &policy, &e->residency_plan)) {
        return false;
    }

    fprintf(stderr,
            "ds4: Qwen Metal memory profile %u GiB: %.2f GiB physical, "
            "%.2f GiB recommended working set, %.2f GiB resident reserve\n",
            policy.profile_gib,
            (double)policy.physical_bytes / 1073741824.0,
            (double)policy.recommended_bytes / 1073741824.0,
            (double)policy.resident_headroom_bytes / 1073741824.0);
    return true;
}
#endif
