#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
RUNNER=$ROOT/speed-bench/run_m5_dsflash_arm.sh

fail() {
    echo "benchmark-env-guard: FAIL: $*" >&2
    exit 1
}

run_and_capture() {
    set +e
    OUTPUT=$(env "$@" 2>&1)
    STATUS=$?
    set -e
}

run_and_capture \
    DS4_GLM_STREAMING_DECODE_FULL_LAYER_MAP=1 \
    DS4_M5_CACHE_STATE=warm \
    "$RUNNER" env-guard-acceptance auto 128
[ "$STATUS" -eq 2 ] ||
    fail "acceptance arm with a hidden GLM flag exited $STATUS instead of 2"
case "$OUTPUT" in
    *"acceptance arm refuses unexpected DS4 runtime environment:"*\
*"DS4_GLM_STREAMING_DECODE_FULL_LAYER_MAP"*) ;;
    *) fail "acceptance arm did not identify the hidden GLM flag" ;;
esac

run_and_capture \
    DS4_GLM_STREAMING_DECODE_FULL_LAYER_MAP=1 \
    DS4_M5_EXPLORATORY=1 \
    "$RUNNER" env-guard-exploratory auto 128
[ "$STATUS" -eq 2 ] ||
    fail "model-free exploratory probe exited $STATUS instead of 2"
case "$OUTPUT" in
    *"exploratory arm records unexpected DS4 runtime environment:"*\
*"DS4_GLM_STREAMING_DECODE_FULL_LAYER_MAP"*\
*"set DS4_M5_MODEL to the ExpertMajor v2 GGUF path"*) ;;
    *) fail "exploratory arm did not retain and report the hidden GLM flag" ;;
esac

echo "benchmark-env-guard: PASS"
