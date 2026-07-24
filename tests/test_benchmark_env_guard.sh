#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
RUNNER=$ROOT/speed-bench/run_m5_dsflash_arm.sh

if ! command -v zsh >/dev/null 2>&1; then
    echo "benchmark-runner-guard: PASS (zsh-only M5 runner skipped on this host)"
    exit 0
fi

fail() {
    echo "benchmark-runner-guard: FAIL: $*" >&2
    exit 1
}

run_and_capture() {
    set +e
    if [ -n "${DEFAULT_MODEL_HASH_EVIDENCE:-}" ]; then
        OUTPUT=$(env -i \
            DS4_M5_MODEL_HASH_EVIDENCE="$DEFAULT_MODEL_HASH_EVIDENCE" \
            FAKE_FORBIDDEN_HASH_PATH="$MODEL" \
            "$@" 2>&1)
    else
        OUTPUT=$(env -i "$@" 2>&1)
    fi
    STATUS=$?
    set -e
}

assert_contains() {
    case $1 in
        *"$2"*) ;;
        *) fail "output does not contain: $2" ;;
    esac
}

TMP_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/hebrus-benchmark-runner.XXXXXX")
trap 'rm -rf "$TMP_ROOT"' EXIT HUP INT TERM
STUB_BIN=$TMP_ROOT/bin
mkdir -p "$STUB_BIN"

MODEL=$TMP_ROOT/model.gguf
PROMPT=$TMP_ROOT/prompt.txt
FAKE_BENCH=$TMP_ROOT/hebrus-bench
printf 'model-free benchmark runner fixture\n' >"$MODEL"
printf 'fixture prompt\n' >"$PROMPT"
MODEL_SHA256=$(shasum -a 256 "$MODEL" | awk '{print $1}')

cat >"$FAKE_BENCH" <<'EOF'
#!/bin/sh
set -eu

if [ "${1:-}" = "--build-info" ]; then
    printf '%s\n' 'hebrus fake benchmark build'
    exit 0
fi

printf '%s\n' "$@" >"$FAKE_ARGS_FILE"
[ -n "${FAKE_CHILD_PID_FILE:-}" ] && printf '%s\n' "$$" >"$FAKE_CHILD_PID_FILE"
[ -z "${FAKE_RUNTIME_STARTED_FILE:-}" ] ||
    printf '%s\n' started >"$FAKE_RUNTIME_STARTED_FILE"
[ "${FAKE_BIN_SLEEP_SECONDS:-0}" = 0 ] || sleep "$FAKE_BIN_SLEEP_SECONDS"

logits=
evidence=
csv=
while [ "$#" -gt 0 ]; do
    case $1 in
        --dump-frontier-logits-dir) logits=$2; shift 2 ;;
        --dump-decode-evidence-dir) evidence=$2; shift 2 ;;
        --csv) csv=$2; shift 2 ;;
        *) shift ;;
    esac
done

mkdir -p "$logits" "$evidence"
printf '%s\n' '{"frontier":128}' >"$logits/frontier.json"
printf '%s\n' '{"decode":128}' >"$evidence/decode.json"
printf '%s\n' 'ctx,prefill_tps,gen_tps' '128,1,1' >"$csv"
if [ -n "${DS4_QWEN_TELEMETRY_JSONL:-}" ] &&
   [ "${FAKE_SKIP_TELEMETRY:-0}" != 1 ]; then
    if [ "${FAKE_PARTIAL_TELEMETRY:-0}" = 1 ]; then
        printf '%s\n' '{"schema":1,"event":"prefill_commit"}' \
            >"$DS4_QWEN_TELEMETRY_JSONL"
    else
        printf '%s\n' '{"schema":1,"event":"runtime_close"}' \
            >"$DS4_QWEN_TELEMETRY_JSONL"
    fi
fi
printf '%s\n' 'ds4: metal_library sha256=fake compile_mode=fixture' >&2
printf '%s\n' 'ds4: residency requested=ssd resolved=ssd' >&2
if [ "${FAKE_TELEMETRY_WRITE_FAILURE:-0}" = 1 ]; then
    printf '%s\n' \
        'ds4: Qwen telemetry write failed; disabling telemetry' >&2
fi
if [ -n "${FAKE_MUTATE_MODEL_AT_EXIT:-}" ]; then
    printf '%s\n' changed >>"$FAKE_MUTATE_MODEL_AT_EXIT"
fi
EOF
chmod +x "$FAKE_BENCH"

cat >"$STUB_BIN/pmset" <<'EOF'
#!/bin/sh
printf "%s\n" "Now drawing from 'AC Power'"
EOF

cat >"$STUB_BIN/sysctl" <<'EOF'
#!/bin/sh
case "${1:-} ${2:-}" in
    '-n hw.memsize') printf '%s\n' "$FAKE_HW_MEMSIZE" ;;
    '-n vm.swapusage') printf '%s\n' 'total = 0.00M  used = 0.00M  free = 0.00M' ;;
    *) exit 1 ;;
esac
EOF

cat >"$STUB_BIN/vm_stat" <<'EOF'
#!/bin/sh
swapouts=0
wired=1000
if [ -n "${FAKE_RUNTIME_STARTED_FILE:-}" ] &&
   [ -f "$FAKE_RUNTIME_STARTED_FILE" ]; then
    swapouts=${FAKE_SWAPOUT_AFTER_START:-0}
    wired=${FAKE_WIRED_AFTER_START:-1000}
fi
cat <<'OUT'
Mach Virtual Memory Statistics: (page size of 16384 bytes)
OUT
printf '%s\n' "Pages wired down: ${wired}."
cat <<OUT
Pageins: 0.
Swapouts: ${swapouts}.
OUT
EOF

cat >"$STUB_BIN/memory_pressure" <<'EOF'
#!/bin/sh
value=${FAKE_FREE_PERCENT:-80}
if [ -n "${FAKE_FREE_COUNTER_FILE:-}" ]; then
    count=0
    [ ! -f "$FAKE_FREE_COUNTER_FILE" ] || count=$(cat "$FAKE_FREE_COUNTER_FILE")
    count=$((count + 1))
    printf '%s\n' "$count" >"$FAKE_FREE_COUNTER_FILE"
    if [ "$count" -gt 1 ] && [ -n "${FAKE_FREE_PERCENT_AFTER_FIRST:-}" ]; then
        value=$FAKE_FREE_PERCENT_AFTER_FIRST
    fi
fi
printf '%s\n' "System-wide memory free percentage: ${value}%"
EOF

cat >"$STUB_BIN/pagesize" <<'EOF'
#!/bin/sh
printf '%s\n' 16384
EOF

cat >"$STUB_BIN/sw_vers" <<'EOF'
#!/bin/sh
printf '%s\n' 'fixture-build'
EOF

cat >"$STUB_BIN/shasum" <<'EOF'
#!/bin/sh
if [ -n "${FAKE_FORBIDDEN_HASH_PATH:-}" ]; then
    for arg in "$@"; do
        if [ "$arg" = "$FAKE_FORBIDDEN_HASH_PATH" ]; then
            printf '%s\n' "unexpected repeated full-model hash: $arg" >&2
            exit 99
        fi
    done
fi
exec /usr/bin/shasum "$@"
EOF

cat >"$STUB_BIN/git" <<'EOF'
#!/bin/sh
if [ -n "${FAKE_MUTATE_MODEL_ON_GIT:-}" ] &&
   [ ! -e "$FAKE_MUTATE_MODEL_ON_GIT.done" ]; then
    printf '%s\n' changed >>"$FAKE_MUTATE_MODEL_ON_GIT"
    : >"$FAKE_MUTATE_MODEL_ON_GIT.done"
fi
exec /usr/bin/git "$@"
EOF

cat >"$STUB_BIN/ps" <<'EOF'
#!/bin/sh
if [ "${1:-}" = "-axo" ]; then
    if [ "${FAKE_PS_LIST_FAIL_AFTER_START:-0}" = 1 ] &&
       [ -n "${FAKE_RUNTIME_STARTED_FILE:-}" ] &&
       [ -f "$FAKE_RUNTIME_STARTED_FILE" ]; then
        exit 1
    fi
    if [ -n "${FAKE_PS_SELF_ROLE:-}" ]; then
        runner_pid=$(/bin/ps -o ppid= -p "$PPID" | awk '{print $1}')
        printf '%s %s %s\n' "$runner_pid" 1 "$FAKE_PS_SELF_ROLE"
    fi
    if [ -n "${FAKE_PS_COMPETITOR_FILE:-}" ] &&
       [ -f "$FAKE_PS_COMPETITOR_FILE" ]; then
        cat "$FAKE_PS_COMPETITOR_FILE"
    fi
    if [ -n "${FAKE_CHILD_PID_FILE:-}" ] &&
       [ -f "$FAKE_CHILD_PID_FILE" ]; then
        child=$(cat "$FAKE_CHILD_PID_FILE")
        if kill -0 "$child" 2>/dev/null; then
            printf '%s %s %s\n' "$child" "$PPID" hebrus-bench
        fi
    fi
    exit 0
fi
if [ "${1:-}" = "-o" ]; then
    if [ -n "${FAKE_PS_RSS_DELAY_FAILURE:-}" ]; then
        [ -z "${FAKE_PS_RSS_DELAY_MARKER_FILE:-}" ] ||
            printf '%s\n' sampled >"$FAKE_PS_RSS_DELAY_MARKER_FILE"
        sleep "$FAKE_PS_RSS_DELAY_FAILURE"
        exit 1
    fi
    [ "${FAKE_PS_RSS_FAIL:-0}" != 1 ] || exit 1
    printf '%s\n' 1024
    exit 0
fi
exec /bin/ps "$@"
EOF
chmod +x "$STUB_BIN"/*

COMMON_PATH=$STUB_BIN:/usr/bin:/bin:/usr/sbin:/sbin
HOME_DIR=${HOME:-$TMP_ROOT}

# Acceptance admits the exact Qwen telemetry sink, while every other
# non-runner DS4_* flag remains forbidden.
run_and_capture \
    PATH="$COMMON_PATH" HOME="$HOME_DIR" \
    DS4_GLM_STREAMING_DECODE_FULL_LAYER_MAP=1 \
    DS4_M5_CACHE_STATE=warm \
    "$RUNNER" env-guard-acceptance auto 128
[ "$STATUS" -eq 2 ] ||
    fail "acceptance arm with a hidden GLM flag exited $STATUS instead of 2"
assert_contains "$OUTPUT" "acceptance arm refuses unexpected DS4 runtime environment:"
assert_contains "$OUTPUT" "DS4_GLM_STREAMING_DECODE_FULL_LAYER_MAP"

PROBE_PREFIX=$TMP_ROOT/telemetry-probe
TELEMETRY_PROBE=$PROBE_PREFIX.qwen-telemetry.jsonl
run_and_capture \
    PATH="$COMMON_PATH" HOME="$HOME_DIR" \
    DS4_QWEN_TELEMETRY_JSONL="$TELEMETRY_PROBE" \
    DS4_M5_PREFIX="$PROBE_PREFIX" \
    DS4_M5_CACHE_STATE=warm \
    "$RUNNER" env-guard-qwen-telemetry auto 128
[ "$STATUS" -eq 2 ] ||
    fail "model-free Qwen telemetry probe exited $STATUS instead of 2"
assert_contains "$OUTPUT" "set DS4_M5_MODEL to the ExpertMajor v2 GGUF path"
case $OUTPUT in
    *"unexpected DS4 runtime environment"*)
        fail "the controlled Qwen telemetry sink was rejected" ;;
esac

run_and_capture \
    PATH="$COMMON_PATH" HOME="$HOME_DIR" \
    DS4_QWEN_TELEMETRY_JSONL=relative.jsonl \
    DS4_M5_CACHE_STATE=warm \
    "$RUNNER" env-guard-relative-telemetry auto 128
[ "$STATUS" -eq 2 ] ||
    fail "relative Qwen telemetry path exited $STATUS instead of 2"
assert_contains "$OUTPUT" "DS4_QWEN_TELEMETRY_JSONL must be an absolute path"

COLLISION_PREFIX=$TMP_ROOT/telemetry-collision
run_and_capture \
    PATH="$COMMON_PATH" HOME="$HOME_DIR" \
    DS4_QWEN_TELEMETRY_JSONL="$COLLISION_PREFIX.summary" \
    DS4_M5_PREFIX="$COLLISION_PREFIX" \
    DS4_M5_CACHE_STATE=warm \
    "$RUNNER" env-guard-telemetry-collision auto 128
[ "$STATUS" -eq 2 ] ||
    fail "colliding Qwen telemetry path exited $STATUS instead of 2"
assert_contains "$OUTPUT" \
    "DS4_QWEN_TELEMETRY_JSONL must equal DS4_M5_PREFIX.qwen-telemetry.jsonl"

EXISTING_PREFIX=$TMP_ROOT/telemetry-existing
EXISTING_TELEMETRY=$EXISTING_PREFIX.qwen-telemetry.jsonl
printf '%s\n' existing >"$EXISTING_TELEMETRY"
run_and_capture \
    PATH="$COMMON_PATH" HOME="$HOME_DIR" \
    DS4_QWEN_TELEMETRY_JSONL="$EXISTING_TELEMETRY" \
    DS4_M5_PREFIX="$EXISTING_PREFIX" \
    DS4_M5_CACHE_STATE=warm \
    "$RUNNER" env-guard-existing-telemetry auto 128
[ "$STATUS" -eq 2 ] ||
    fail "existing Qwen telemetry path exited $STATUS instead of 2"
assert_contains "$OUTPUT" "refusing to append to existing Qwen telemetry"

DANGLING_PREFIX=$TMP_ROOT/telemetry-dangling
DANGLING_TELEMETRY=$DANGLING_PREFIX.qwen-telemetry.jsonl
ln -s "$TMP_ROOT/missing-telemetry-target" "$DANGLING_TELEMETRY"
run_and_capture \
    PATH="$COMMON_PATH" HOME="$HOME_DIR" \
    DS4_QWEN_TELEMETRY_JSONL="$DANGLING_TELEMETRY" \
    DS4_M5_PREFIX="$DANGLING_PREFIX" \
    DS4_M5_CACHE_STATE=warm \
    "$RUNNER" env-guard-dangling-telemetry auto 128
[ "$STATUS" -eq 2 ] ||
    fail "dangling Qwen telemetry symlink exited $STATUS instead of 2"
assert_contains "$OUTPUT" "refusing to append to existing Qwen telemetry"

run_and_capture \
    PATH="$COMMON_PATH" HOME="$HOME_DIR" \
    DS4_GLM_STREAMING_DECODE_FULL_LAYER_MAP=1 \
    DS4_M5_EXPLORATORY=1 \
    "$RUNNER" env-guard-exploratory auto 128
[ "$STATUS" -eq 2 ] ||
    fail "model-free exploratory probe exited $STATUS instead of 2"
assert_contains "$OUTPUT" "exploratory arm records unexpected DS4 runtime environment:"
assert_contains "$OUTPUT" "DS4_GLM_STREAMING_DECODE_FULL_LAYER_MAP"
assert_contains "$OUTPUT" "set DS4_M5_MODEL to the ExpertMajor v2 GGUF path"

# A one-shot full-hash preparation fails closed on mismatch, reports the actual
# digest, and emits no evidence. The successful preparation is then reused by
# every fixture arm without rereading the model payload.
ZERO_SHA=0000000000000000000000000000000000000000000000000000000000000000
MODEL_HASH_EVIDENCE=$TMP_ROOT/model-hash.evidence
run_and_capture \
    PATH="$COMMON_PATH" HOME="$HOME_DIR" \
    DS4_M5_MODEL="$MODEL" \
    DS4_M5_MODEL_SHA256="$ZERO_SHA" \
    DS4_M5_MODEL_HASH_EVIDENCE="$MODEL_HASH_EVIDENCE" \
    "$RUNNER" --prepare-model-hash-evidence
[ "$STATUS" -eq 2 ] ||
    fail "hash mismatch exited $STATUS instead of 2"
assert_contains "$OUTPUT" "model SHA-256 mismatch"
assert_contains "$OUTPUT" "expected=$ZERO_SHA"
assert_contains "$OUTPUT" "actual=$MODEL_SHA256"
[ ! -e "$MODEL_HASH_EVIDENCE" ] ||
    fail "hash mismatch left apparently valid evidence"

run_and_capture \
    PATH="$COMMON_PATH" HOME="$HOME_DIR" \
    DS4_M5_MODEL="$MODEL" \
    DS4_M5_MODEL_SHA256="$MODEL_SHA256" \
    DS4_M5_MODEL_HASH_EVIDENCE="$MODEL_HASH_EVIDENCE" \
    "$RUNNER" --prepare-model-hash-evidence
[ "$STATUS" -eq 0 ] ||
    fail "model hash evidence preparation exited $STATUS: $OUTPUT"
[ -s "$MODEL_HASH_EVIDENCE" ] ||
    fail "model hash evidence preparation produced no evidence"
grep -Fx -- "model_sha256_actual=$MODEL_SHA256" \
    "$MODEL_HASH_EVIDENCE" >/dev/null ||
    fail "model hash evidence did not retain the actual digest"

PREP_COMPETITOR=$TMP_ROOT/prepare-competitor.txt
PREP_BLOCKED_EVIDENCE=$TMP_ROOT/blocked-model-hash.evidence
printf '%s %s %s\n' 4242 1 hebrus-server >"$PREP_COMPETITOR"
run_and_capture \
    PATH="$COMMON_PATH" HOME="$HOME_DIR" \
    FAKE_PS_COMPETITOR_FILE="$PREP_COMPETITOR" \
    DS4_M5_MODEL="$MODEL" \
    DS4_M5_MODEL_SHA256="$MODEL_SHA256" \
    DS4_M5_MODEL_HASH_EVIDENCE="$PREP_BLOCKED_EVIDENCE" \
    "$RUNNER" --prepare-model-hash-evidence
[ "$STATUS" -eq 2 ] ||
    fail "hash preparation beside a competitor exited $STATUS instead of 2"
assert_contains "$OUTPUT" "refusing to run beside another inference process:"
[ ! -e "$PREP_BLOCKED_EVIDENCE" ] ||
    fail "competitor-blocked hash preparation emitted evidence"

DEFAULT_MODEL_HASH_EVIDENCE=$MODEL_HASH_EVIDENCE

MODEL_COPY=$TMP_ROOT/model-copy.gguf
cp "$MODEL" "$MODEL_COPY"
run_and_capture \
    PATH="$COMMON_PATH" HOME="$HOME_DIR" \
    FAKE_HW_MEMSIZE=68719476736 \
    FAKE_ARGS_FILE="$TMP_ROOT/model-copy.args" \
    DS4_M5_BIN="$FAKE_BENCH" \
    DS4_M5_MODEL="$MODEL_COPY" \
    DS4_M5_MODEL_SHA256="$MODEL_SHA256" \
    DS4_M5_PROMPT="$PROMPT" \
    DS4_M5_CACHE_STATE=warm \
    "$RUNNER" stale-model-evidence auto 128
[ "$STATUS" -eq 2 ] ||
    fail "evidence bound to another model identity exited $STATUS instead of 2"
assert_contains "$OUTPUT" \
    "model hash evidence does not match the configured model identity"

# The 64 GiB profile preserves the historical 46 GiB wired ceiling and
# DeepSeek's explicit 4096-expert preload default.
PREFIX_64=$TMP_ROOT/run-64
ARGS_64=$TMP_ROOT/run-64.args
run_and_capture \
    PATH="$COMMON_PATH" HOME="$HOME_DIR" \
    FAKE_HW_MEMSIZE=68719476736 \
    FAKE_ARGS_FILE="$ARGS_64" \
    DS4_M5_ROOT="$ROOT" \
    DS4_M5_BIN="$FAKE_BENCH" \
    DS4_M5_MODEL="$MODEL" \
    DS4_M5_MODEL_SHA256="$MODEL_SHA256" \
    DS4_M5_PROMPT="$PROMPT" \
    DS4_M5_PREFIX="$PREFIX_64" \
    DS4_M5_CACHE_STATE=warm \
    "$RUNNER" profile-64 auto 128
[ "$STATUS" -eq 0 ] ||
    fail "64 GiB fixture arm exited $STATUS: $OUTPUT"
grep -Fx -- '--ssd-streaming-preload-experts' "$ARGS_64" >/dev/null ||
    fail "64 GiB default omitted the preload option"
grep -Fx -- '4096' "$ARGS_64" >/dev/null ||
    fail "64 GiB default did not retain the 4096-expert preload"
grep -Fx -- 'host_memory_profile=64g-plus' "$PREFIX_64.summary" >/dev/null ||
    fail "64 GiB summary did not retain its host profile"
grep -Fx -- 'max_wired_gib=46' "$PREFIX_64.summary" >/dev/null ||
    fail "64 GiB summary did not retain the historical wired ceiling"

# A transient live free-memory drop remains recorded but does not retrofit the
# 24 GiB watchdog policy onto the historical 64 GiB lane.
PREFIX_64_PRESSURE=$TMP_ROOT/run-64-pressure
run_and_capture \
    PATH="$COMMON_PATH" HOME="$HOME_DIR" \
    FAKE_HW_MEMSIZE=68719476736 \
    FAKE_ARGS_FILE="$TMP_ROOT/run-64-pressure.args" \
    FAKE_BIN_SLEEP_SECONDS=2 \
    FAKE_FREE_COUNTER_FILE="$TMP_ROOT/run-64-pressure.counter" \
    FAKE_FREE_PERCENT=80 \
    FAKE_FREE_PERCENT_AFTER_FIRST=10 \
    DS4_M5_ROOT="$ROOT" \
    DS4_M5_BIN="$FAKE_BENCH" \
    DS4_M5_MODEL="$MODEL" \
    DS4_M5_MODEL_SHA256="$MODEL_SHA256" \
    DS4_M5_PROMPT="$PROMPT" \
    DS4_M5_PREFIX="$PREFIX_64_PRESSURE" \
    DS4_M5_CACHE_STATE=warm \
    "$RUNNER" profile-64-pressure auto 128
[ "$STATUS" -eq 0 ] ||
    fail "64 GiB transient pressure drop exited $STATUS: $OUTPUT"
grep -Fx -- 'pressure_min=10' "$PREFIX_64_PRESSURE.summary" >/dev/null ||
    fail "64 GiB pressure minimum was not retained"
grep -Fx -- 'abort_reason=none' "$PREFIX_64_PRESSURE.summary" >/dev/null ||
    fail "64 GiB transient pressure drop incorrectly invalidated the arm"

# A child may disappear between kill -0 and its RSS sample. An unavailable RSS
# sample must not bypass final evidence collection or truncate the summary.
PREFIX_RSS_RACE=$TMP_ROOT/run-rss-race
run_and_capture \
    PATH="$COMMON_PATH" HOME="$HOME_DIR" \
    FAKE_HW_MEMSIZE=68719476736 \
    FAKE_ARGS_FILE="$TMP_ROOT/run-rss-race.args" \
    FAKE_BIN_SLEEP_SECONDS=1 \
    FAKE_PS_RSS_DELAY_FAILURE=2 \
    FAKE_PS_RSS_DELAY_MARKER_FILE="$TMP_ROOT/run-rss-race.sampled" \
    DS4_M5_ROOT="$ROOT" \
    DS4_M5_BIN="$FAKE_BENCH" \
    DS4_M5_MODEL="$MODEL" \
    DS4_M5_MODEL_SHA256="$MODEL_SHA256" \
    DS4_M5_PROMPT="$PROMPT" \
    DS4_M5_PREFIX="$PREFIX_RSS_RACE" \
    DS4_M5_CACHE_STATE=warm \
    "$RUNNER" rss-exit-race auto 128
[ "$STATUS" -eq 0 ] ||
    fail "unavailable child RSS sample exited $STATUS: $OUTPUT"
[ -s "$PREFIX_RSS_RACE.summary" ] ||
    fail "unavailable child RSS sample truncated the summary"
[ -s "$TMP_ROOT/run-rss-race.sampled" ] ||
    fail "RSS exit-race fixture never exercised the delayed failing sample"

# The same unavailable RSS signal is fail-closed while the child remains live.
PREFIX_RSS_FAILURE=$TMP_ROOT/run-rss-failure
run_and_capture \
    PATH="$COMMON_PATH" HOME="$HOME_DIR" \
    FAKE_HW_MEMSIZE=68719476736 \
    FAKE_ARGS_FILE="$TMP_ROOT/run-rss-failure.args" \
    FAKE_BIN_SLEEP_SECONDS=3 \
    FAKE_PS_RSS_FAIL=1 \
    DS4_M5_ROOT="$ROOT" \
    DS4_M5_BIN="$FAKE_BENCH" \
    DS4_M5_MODEL="$MODEL" \
    DS4_M5_MODEL_SHA256="$MODEL_SHA256" \
    DS4_M5_PROMPT="$PROMPT" \
    DS4_M5_PREFIX="$PREFIX_RSS_FAILURE" \
    DS4_M5_CACHE_STATE=warm \
    "$RUNNER" rss-sensor-failure auto 128
[ "$STATUS" -eq 125 ] ||
    fail "live child RSS failure exited $STATUS instead of 125: $OUTPUT"
grep -Fx -- 'abort_reason=telemetry_rss_unavailable' \
    "$PREFIX_RSS_FAILURE.summary" >/dev/null ||
    fail "live child RSS failure was not retained in the summary"

# On 24 GiB, forced SSD can omit preload entirely. The fake process advertises
# both the runner PID and current child under canonical names; neither is a
# competitor, while the requested Qwen telemetry remains part of the evidence.
PREFIX_24=$TMP_ROOT/run-24
ARGS_24=$TMP_ROOT/run-24.args
TELEMETRY_24=$PREFIX_24.qwen-telemetry.jsonl
CHILD_24=$TMP_ROOT/run-24.child
run_and_capture \
    PATH="$COMMON_PATH" HOME="$HOME_DIR" \
    FAKE_HW_MEMSIZE=25769803776 \
    FAKE_ARGS_FILE="$ARGS_24" \
    FAKE_CHILD_PID_FILE="$CHILD_24" \
    FAKE_BIN_SLEEP_SECONDS=2 \
    FAKE_PS_SELF_ROLE=hebrus-server \
    DS4_QWEN_TELEMETRY_JSONL="$TELEMETRY_24" \
    DS4_M5_ROOT="$ROOT" \
    DS4_M5_BIN="$FAKE_BENCH" \
    DS4_M5_MODEL="$MODEL" \
    DS4_M5_MODEL_SHA256="$MODEL_SHA256" \
    DS4_M5_PROMPT="$PROMPT" \
    DS4_M5_PREFIX="$PREFIX_24" \
    DS4_M5_CACHE_STATE=warm \
    DS4_M5_PRELOAD_POLICY=omit \
    "$RUNNER" profile-24 exact3521 128
[ "$STATUS" -eq 0 ] ||
    fail "24 GiB no-preload fixture arm exited $STATUS: $OUTPUT"
grep -Fx -- '--ssd-streaming' "$ARGS_24" >/dev/null ||
    fail "24 GiB forced-SSD arm omitted --ssd-streaming"
if grep -Fx -- '--ssd-streaming-preload-experts' "$ARGS_24" >/dev/null; then
    fail "24 GiB no-preload arm still passed the preload option"
fi
grep -Fx -- 'host_memory_profile=24g' "$PREFIX_24.summary" >/dev/null ||
    fail "24 GiB summary did not retain its host profile"
grep -Fx -- 'max_wired_gib=17' "$PREFIX_24.summary" >/dev/null ||
    fail "24 GiB summary did not apply the 17 GiB ceiling"
grep -Fx -- 'preload_policy=omit' "$PREFIX_24.summary" >/dev/null ||
    fail "24 GiB summary did not retain the omit policy"
grep -Fx -- 'preload=omitted' "$PREFIX_24.summary" >/dev/null ||
    fail "24 GiB summary claimed a preload value"
grep -Fx -- "model_sha256_actual=$MODEL_SHA256" "$PREFIX_24.summary" >/dev/null ||
    fail "summary did not retain the complete actual model hash"
grep -Fx -- 'model_sha256_verification=one-shot-evidence-match' \
    "$PREFIX_24.summary" >/dev/null ||
    fail "summary did not record full hash verification"
cmp "$MODEL_HASH_EVIDENCE" "$PREFIX_24.model-hash-evidence" >/dev/null ||
    fail "runner did not retain the exact one-shot model hash evidence"
grep -Fx -- "DS4_QWEN_TELEMETRY_JSONL=$TELEMETRY_24" "$PREFIX_24.env" >/dev/null ||
    fail "environment evidence omitted the controlled Qwen telemetry sink"
[ -s "$TELEMETRY_24" ] || fail "Qwen telemetry evidence is empty"
TELEMETRY_24_SHA256=$(shasum -a 256 "$TELEMETRY_24" | awk '{print $1}')
grep -Fx -- "qwen_telemetry_sha256=$TELEMETRY_24_SHA256" \
    "$PREFIX_24.summary" >/dev/null ||
    fail "summary did not retain the Qwen telemetry digest"
grep -Fx -- 'qwen_telemetry_validation=valid' \
    "$PREFIX_24.summary" >/dev/null ||
    fail "summary did not retain successful Qwen telemetry validation"
grep -Fx -- 'qwen_telemetry_records=1' "$PREFIX_24.summary" >/dev/null ||
    fail "summary did not retain the validated Qwen telemetry record count"
grep -Fx -- '--ssd-streaming-cache-experts' "$ARGS_24" >/dev/null ||
    fail "exact3521 boundary omitted the explicit cache option"
grep -Fx -- '3521' "$ARGS_24" >/dev/null ||
    fail "exact3521 boundary did not reach the fake benchmark"

PREFIX_EMPTY_TELEMETRY=$TMP_ROOT/run-empty-telemetry
EMPTY_TELEMETRY=$PREFIX_EMPTY_TELEMETRY.qwen-telemetry.jsonl
run_and_capture \
    PATH="$COMMON_PATH" HOME="$HOME_DIR" \
    FAKE_HW_MEMSIZE=25769803776 \
    FAKE_ARGS_FILE="$TMP_ROOT/run-empty-telemetry.args" \
    FAKE_SKIP_TELEMETRY=1 \
    DS4_QWEN_TELEMETRY_JSONL="$EMPTY_TELEMETRY" \
    DS4_M5_ROOT="$ROOT" \
    DS4_M5_BIN="$FAKE_BENCH" \
    DS4_M5_MODEL="$MODEL" \
    DS4_M5_MODEL_SHA256="$MODEL_SHA256" \
    DS4_M5_PROMPT="$PROMPT" \
    DS4_M5_PREFIX="$PREFIX_EMPTY_TELEMETRY" \
    DS4_M5_CACHE_STATE=warm \
    DS4_M5_PRELOAD_POLICY=omit \
    "$RUNNER" empty-telemetry auto 128
[ "$STATUS" -eq 126 ] ||
    fail "missing requested telemetry exited $STATUS instead of 126: $OUTPUT"
grep -Fx -- 'result_error=missing_qwen_telemetry' \
    "$PREFIX_EMPTY_TELEMETRY.summary" >/dev/null ||
    fail "missing requested telemetry was not retained as a result error"

PREFIX_PARTIAL_TELEMETRY=$TMP_ROOT/run-partial-telemetry
PARTIAL_TELEMETRY=$PREFIX_PARTIAL_TELEMETRY.qwen-telemetry.jsonl
run_and_capture \
    PATH="$COMMON_PATH" HOME="$HOME_DIR" \
    FAKE_HW_MEMSIZE=25769803776 \
    FAKE_ARGS_FILE="$TMP_ROOT/run-partial-telemetry.args" \
    FAKE_PARTIAL_TELEMETRY=1 \
    DS4_QWEN_TELEMETRY_JSONL="$PARTIAL_TELEMETRY" \
    DS4_M5_ROOT="$ROOT" \
    DS4_M5_BIN="$FAKE_BENCH" \
    DS4_M5_MODEL="$MODEL" \
    DS4_M5_MODEL_SHA256="$MODEL_SHA256" \
    DS4_M5_PROMPT="$PROMPT" \
    DS4_M5_PREFIX="$PREFIX_PARTIAL_TELEMETRY" \
    DS4_M5_CACHE_STATE=warm \
    DS4_M5_PRELOAD_POLICY=omit \
    "$RUNNER" partial-telemetry auto 128
[ "$STATUS" -eq 126 ] ||
    fail "partial requested telemetry exited $STATUS instead of 126: $OUTPUT"
grep -Fx -- 'result_error=invalid_qwen_telemetry' \
    "$PREFIX_PARTIAL_TELEMETRY.summary" >/dev/null ||
    fail "partial requested telemetry was not retained as invalid"
grep -Fx -- 'qwen_telemetry_validation=invalid' \
    "$PREFIX_PARTIAL_TELEMETRY.summary" >/dev/null ||
    fail "partial requested telemetry lacked validation status"

PREFIX_FAILED_TELEMETRY=$TMP_ROOT/run-failed-telemetry
FAILED_TELEMETRY=$PREFIX_FAILED_TELEMETRY.qwen-telemetry.jsonl
run_and_capture \
    PATH="$COMMON_PATH" HOME="$HOME_DIR" \
    FAKE_HW_MEMSIZE=25769803776 \
    FAKE_ARGS_FILE="$TMP_ROOT/run-failed-telemetry.args" \
    FAKE_TELEMETRY_WRITE_FAILURE=1 \
    DS4_QWEN_TELEMETRY_JSONL="$FAILED_TELEMETRY" \
    DS4_M5_ROOT="$ROOT" \
    DS4_M5_BIN="$FAKE_BENCH" \
    DS4_M5_MODEL="$MODEL" \
    DS4_M5_MODEL_SHA256="$MODEL_SHA256" \
    DS4_M5_PROMPT="$PROMPT" \
    DS4_M5_PREFIX="$PREFIX_FAILED_TELEMETRY" \
    DS4_M5_CACHE_STATE=warm \
    DS4_M5_PRELOAD_POLICY=omit \
    "$RUNNER" failed-telemetry auto 128
[ "$STATUS" -eq 126 ] ||
    fail "runtime telemetry failure exited $STATUS instead of 126: $OUTPUT"
grep -Fx -- 'result_error=qwen_telemetry_runtime_failure' \
    "$PREFIX_FAILED_TELEMETRY.summary" >/dev/null ||
    fail "runtime telemetry failure was not retained as a result error"
grep -Fx -- 'qwen_telemetry_validation=runtime-failure' \
    "$PREFIX_FAILED_TELEMETRY.summary" >/dev/null ||
    fail "runtime telemetry failure lacked validation status"

# Every canonical inference role, every compatibility role, and llama-server
# invalidates an arm. The runner and its current benchmark child were covered
# by the successful 24 GiB fixture above.
COMPETITOR_FILE=$TMP_ROOT/competitor.txt
for role in \
    hebrus hebrus-server hebrus-agent hebrus-bench hebrus-eval \
    ds4 ds4-server ds4-agent ds4-bench ds4-eval llama-server
do
    printf '%s %s %s\n' 4242 1 "$role" >"$COMPETITOR_FILE"
    run_and_capture \
        PATH="$COMMON_PATH" HOME="$HOME_DIR" \
        FAKE_HW_MEMSIZE=25769803776 \
        FAKE_ARGS_FILE="$TMP_ROOT/competitor.args" \
        FAKE_PS_COMPETITOR_FILE="$COMPETITOR_FILE" \
        DS4_M5_ROOT="$ROOT" \
        DS4_M5_BIN="$FAKE_BENCH" \
        DS4_M5_MODEL="$MODEL" \
        DS4_M5_MODEL_SHA256="$MODEL_SHA256" \
        DS4_M5_PROMPT="$PROMPT" \
        DS4_M5_PREFIX="$TMP_ROOT/competitor-$role" \
        DS4_M5_CACHE_STATE=warm \
        DS4_M5_PRELOAD_POLICY=omit \
        "$RUNNER" "competitor-$role" auto 128
    [ "$STATUS" -eq 2 ] ||
        fail "competitor $role exited $STATUS instead of 2"
    assert_contains "$OUTPUT" "refusing to run beside another inference process:"
    assert_contains "$OUTPUT" "$role"
done

# The 24 GiB preset is a hard safety boundary, not a 46 GiB inherited default.
run_and_capture \
    PATH="$COMMON_PATH" HOME="$HOME_DIR" \
    FAKE_HW_MEMSIZE=25769803776 \
    FAKE_ARGS_FILE="$TMP_ROOT/invalid.args" \
    DS4_M5_BIN="$FAKE_BENCH" \
    DS4_M5_MODEL="$MODEL" \
    DS4_M5_MODEL_SHA256="$MODEL_SHA256" \
    DS4_M5_PROMPT="$PROMPT" \
    DS4_M5_CACHE_STATE=warm \
    DS4_M5_PRELOAD_POLICY=omit \
    DS4_M5_MAX_WIRED_GIB=18 \
    "$RUNNER" invalid-wired auto 128
[ "$STATUS" -eq 2 ] || fail "unsafe 24 GiB wired cap exited $STATUS instead of 2"
assert_contains "$OUTPUT" "24 GiB arms require DS4_M5_MAX_WIRED_GIB in 1..17"

run_and_capture \
    PATH="$COMMON_PATH" HOME="$HOME_DIR" \
    FAKE_HW_MEMSIZE=25769803776 \
    FAKE_ARGS_FILE="$TMP_ROOT/invalid.args" \
    DS4_M5_BIN="$FAKE_BENCH" \
    DS4_M5_MODEL="$MODEL" \
    DS4_M5_MODEL_SHA256="$MODEL_SHA256" \
    DS4_M5_PROMPT="$PROMPT" \
    DS4_M5_CACHE_STATE=warm \
    DS4_M5_PRELOAD_POLICY=omit \
    DS4_M5_MAX_SWAPOUT_PAGES=1 \
    "$RUNNER" invalid-swap auto 128
[ "$STATUS" -eq 2 ] || fail "unsafe 24 GiB swap cap exited $STATUS instead of 2"
assert_contains "$OUTPUT" "24 GiB arms require DS4_M5_MAX_SWAPOUT_PAGES=0"

run_and_capture \
    PATH="$COMMON_PATH" HOME="$HOME_DIR" \
    FAKE_HW_MEMSIZE=25769803776 \
    FAKE_ARGS_FILE="$TMP_ROOT/invalid.args" \
    DS4_M5_BIN="$FAKE_BENCH" \
    DS4_M5_MODEL="$MODEL" \
    DS4_M5_MODEL_SHA256="$MODEL_SHA256" \
    DS4_M5_PROMPT="$PROMPT" \
    DS4_M5_CACHE_STATE=warm \
    DS4_M5_PRELOAD_POLICY=omit \
    DS4_M5_MIN_FREE_PERCENT=19 \
    "$RUNNER" invalid-free-floor auto 128
[ "$STATUS" -eq 2 ] ||
    fail "unsafe 24 GiB free-percent floor exited $STATUS instead of 2"
assert_contains "$OUTPUT" "24 GiB arms require DS4_M5_MIN_FREE_PERCENT >= 20"

run_and_capture \
    PATH="$COMMON_PATH" HOME="$HOME_DIR" \
    FAKE_HW_MEMSIZE=25769803776 \
    FAKE_ARGS_FILE="$TMP_ROOT/invalid.args" \
    DS4_M5_BIN="$FAKE_BENCH" \
    DS4_M5_MODEL="$MODEL" \
    DS4_M5_MODEL_SHA256="$MODEL_SHA256" \
    DS4_M5_PROMPT="$PROMPT" \
    DS4_M5_CACHE_STATE=warm \
    DS4_M5_PRELOAD_POLICY=omit \
    "$RUNNER" invalid-cache exact3522 128
[ "$STATUS" -eq 2 ] || fail "unsafe 24 GiB cache cap exited $STATUS instead of 2"
assert_contains "$OUTPUT" "24 GiB arms reject an exact cache above 3521 experts"

run_and_capture \
    PATH="$COMMON_PATH" HOME="$HOME_DIR" \
    FAKE_HW_MEMSIZE=25769803776 \
    FAKE_ARGS_FILE="$TMP_ROOT/invalid.args" \
    DS4_M5_BIN="$FAKE_BENCH" \
    DS4_M5_MODEL="$MODEL" \
    DS4_M5_MODEL_SHA256="$MODEL_SHA256" \
    DS4_M5_PROMPT="$PROMPT" \
    DS4_M5_CACHE_STATE=warm \
    "$RUNNER" invalid-preload auto 128
[ "$STATUS" -eq 2 ] || fail "24 GiB default preload exited $STATUS instead of 2"
assert_contains "$OUTPUT" "24 GiB forced-SSD arms require DS4_M5_PRELOAD_POLICY=omit"

# A free-percentage breach after launch terminates the arm and is retained in
# the summary instead of being telemetry-only.
PREFIX_PRESSURE=$TMP_ROOT/run-pressure
run_and_capture \
    PATH="$COMMON_PATH" HOME="$HOME_DIR" \
    FAKE_HW_MEMSIZE=25769803776 \
    FAKE_ARGS_FILE="$TMP_ROOT/run-pressure.args" \
    FAKE_CHILD_PID_FILE="$TMP_ROOT/run-pressure.child" \
    FAKE_BIN_SLEEP_SECONDS=3 \
    FAKE_FREE_COUNTER_FILE="$TMP_ROOT/run-pressure.counter" \
    FAKE_FREE_PERCENT=80 \
    FAKE_FREE_PERCENT_AFTER_FIRST=10 \
    DS4_M5_ROOT="$ROOT" \
    DS4_M5_BIN="$FAKE_BENCH" \
    DS4_M5_MODEL="$MODEL" \
    DS4_M5_MODEL_SHA256="$MODEL_SHA256" \
    DS4_M5_PROMPT="$PROMPT" \
    DS4_M5_PREFIX="$PREFIX_PRESSURE" \
    DS4_M5_CACHE_STATE=warm \
    DS4_M5_PRELOAD_POLICY=omit \
    "$RUNNER" pressure-abort auto 128
[ "$STATUS" -eq 125 ] ||
    fail "post-launch pressure breach exited $STATUS instead of 125: $OUTPUT"
grep -Fx -- 'abort_reason=free_memory_below_20_percent' \
    "$PREFIX_PRESSURE.summary" >/dev/null ||
    fail "post-launch pressure breach was not retained in the summary"

PREFIX_SWAP=$TMP_ROOT/run-swap
run_and_capture \
    PATH="$COMMON_PATH" HOME="$HOME_DIR" \
    FAKE_HW_MEMSIZE=25769803776 \
    FAKE_ARGS_FILE="$TMP_ROOT/run-swap.args" \
    FAKE_CHILD_PID_FILE="$TMP_ROOT/run-swap.child" \
    FAKE_BIN_SLEEP_SECONDS=3 \
    FAKE_RUNTIME_STARTED_FILE="$TMP_ROOT/run-swap.started" \
    FAKE_SWAPOUT_AFTER_START=1 \
    DS4_M5_ROOT="$ROOT" \
    DS4_M5_BIN="$FAKE_BENCH" \
    DS4_M5_MODEL="$MODEL" \
    DS4_M5_MODEL_SHA256="$MODEL_SHA256" \
    DS4_M5_PROMPT="$PROMPT" \
    DS4_M5_PREFIX="$PREFIX_SWAP" \
    DS4_M5_CACHE_STATE=warm \
    DS4_M5_PRELOAD_POLICY=omit \
    "$RUNNER" swap-abort auto 128
[ "$STATUS" -eq 125 ] ||
    fail "post-launch swapout exited $STATUS instead of 125: $OUTPUT"
grep -Fx -- 'abort_reason=swapout_over_0_pages' \
    "$PREFIX_SWAP.summary" >/dev/null ||
    fail "post-launch swapout was not retained in the summary"

PREFIX_WIRED=$TMP_ROOT/run-wired
run_and_capture \
    PATH="$COMMON_PATH" HOME="$HOME_DIR" \
    FAKE_HW_MEMSIZE=25769803776 \
    FAKE_ARGS_FILE="$TMP_ROOT/run-wired.args" \
    FAKE_CHILD_PID_FILE="$TMP_ROOT/run-wired.child" \
    FAKE_BIN_SLEEP_SECONDS=3 \
    FAKE_RUNTIME_STARTED_FILE="$TMP_ROOT/run-wired.started" \
    FAKE_WIRED_AFTER_START=1200000 \
    DS4_M5_ROOT="$ROOT" \
    DS4_M5_BIN="$FAKE_BENCH" \
    DS4_M5_MODEL="$MODEL" \
    DS4_M5_MODEL_SHA256="$MODEL_SHA256" \
    DS4_M5_PROMPT="$PROMPT" \
    DS4_M5_PREFIX="$PREFIX_WIRED" \
    DS4_M5_CACHE_STATE=warm \
    DS4_M5_PRELOAD_POLICY=omit \
    "$RUNNER" wired-abort auto 128
[ "$STATUS" -eq 125 ] ||
    fail "post-launch wired breach exited $STATUS instead of 125: $OUTPUT"
grep -Fx -- 'abort_reason=wired_memory_over_17_GiB' \
    "$PREFIX_WIRED.summary" >/dev/null ||
    fail "post-launch wired breach was not retained in the summary"

# Process enumeration is a safety signal, not optional telemetry. A failure
# after launch must stop the child and still produce a fail-closed summary.
PREFIX_PS_FAILURE=$TMP_ROOT/run-process-list-failure
run_and_capture \
    PATH="$COMMON_PATH" HOME="$HOME_DIR" \
    FAKE_HW_MEMSIZE=25769803776 \
    FAKE_ARGS_FILE="$TMP_ROOT/run-process-list-failure.args" \
    FAKE_BIN_SLEEP_SECONDS=3 \
    FAKE_RUNTIME_STARTED_FILE="$TMP_ROOT/run-process-list-failure.started" \
    FAKE_PS_LIST_FAIL_AFTER_START=1 \
    DS4_M5_ROOT="$ROOT" \
    DS4_M5_BIN="$FAKE_BENCH" \
    DS4_M5_MODEL="$MODEL" \
    DS4_M5_MODEL_SHA256="$MODEL_SHA256" \
    DS4_M5_PROMPT="$PROMPT" \
    DS4_M5_PREFIX="$PREFIX_PS_FAILURE" \
    DS4_M5_CACHE_STATE=warm \
    DS4_M5_PRELOAD_POLICY=omit \
    "$RUNNER" process-list-failure auto 128
[ "$STATUS" -eq 125 ] ||
    fail "process-list failure exited $STATUS instead of 125: $OUTPUT"
grep -Fx -- 'abort_reason=telemetry_process_list_unavailable' \
    "$PREFIX_PS_FAILURE.summary" >/dev/null ||
    fail "process-list failure was not retained in the summary"

MUTATING_MODEL=$TMP_ROOT/mutating-model.gguf
MUTATING_EVIDENCE=$TMP_ROOT/mutating-model-hash.evidence
cp "$MODEL" "$MUTATING_MODEL"
run_and_capture \
    PATH="$COMMON_PATH" HOME="$HOME_DIR" \
    DS4_M5_MODEL="$MUTATING_MODEL" \
    DS4_M5_MODEL_SHA256="$MODEL_SHA256" \
    DS4_M5_MODEL_HASH_EVIDENCE="$MUTATING_EVIDENCE" \
    "$RUNNER" --prepare-model-hash-evidence
[ "$STATUS" -eq 0 ] ||
    fail "mutating-model evidence preparation exited $STATUS: $OUTPUT"
run_and_capture \
    PATH="$COMMON_PATH" HOME="$HOME_DIR" \
    FAKE_HW_MEMSIZE=68719476736 \
    FAKE_ARGS_FILE="$TMP_ROOT/mutating-model.args" \
    FAKE_MUTATE_MODEL_ON_GIT="$MUTATING_MODEL" \
    DS4_M5_MODEL_HASH_EVIDENCE="$MUTATING_EVIDENCE" \
    DS4_M5_ROOT="$ROOT" \
    DS4_M5_BIN="$FAKE_BENCH" \
    DS4_M5_MODEL="$MUTATING_MODEL" \
    DS4_M5_MODEL_SHA256="$MODEL_SHA256" \
    DS4_M5_PROMPT="$PROMPT" \
    DS4_M5_PREFIX="$TMP_ROOT/mutating-model-run" \
    DS4_M5_CACHE_STATE=warm \
    "$RUNNER" mutating-model auto 128
[ "$STATUS" -eq 2 ] ||
    fail "model mutation before exec exited $STATUS instead of 2"
assert_contains "$OUTPUT" "model identity changed after hash-evidence verification"

POST_MUTATING_MODEL=$TMP_ROOT/post-mutating-model.gguf
POST_MUTATING_EVIDENCE=$TMP_ROOT/post-mutating-model-hash.evidence
cp "$MODEL" "$POST_MUTATING_MODEL"
run_and_capture \
    PATH="$COMMON_PATH" HOME="$HOME_DIR" \
    DS4_M5_MODEL="$POST_MUTATING_MODEL" \
    DS4_M5_MODEL_SHA256="$MODEL_SHA256" \
    DS4_M5_MODEL_HASH_EVIDENCE="$POST_MUTATING_EVIDENCE" \
    "$RUNNER" --prepare-model-hash-evidence
[ "$STATUS" -eq 0 ] ||
    fail "post-mutation evidence preparation exited $STATUS: $OUTPUT"
POST_MUTATING_PREFIX=$TMP_ROOT/post-mutating-model-run
run_and_capture \
    PATH="$COMMON_PATH" HOME="$HOME_DIR" \
    FAKE_HW_MEMSIZE=68719476736 \
    FAKE_ARGS_FILE="$TMP_ROOT/post-mutating-model.args" \
    FAKE_MUTATE_MODEL_AT_EXIT="$POST_MUTATING_MODEL" \
    DS4_M5_MODEL_HASH_EVIDENCE="$POST_MUTATING_EVIDENCE" \
    DS4_M5_ROOT="$ROOT" \
    DS4_M5_BIN="$FAKE_BENCH" \
    DS4_M5_MODEL="$POST_MUTATING_MODEL" \
    DS4_M5_MODEL_SHA256="$MODEL_SHA256" \
    DS4_M5_PROMPT="$PROMPT" \
    DS4_M5_PREFIX="$POST_MUTATING_PREFIX" \
    DS4_M5_CACHE_STATE=warm \
    "$RUNNER" post-mutating-model auto 128
[ "$STATUS" -eq 125 ] ||
    fail "model mutation during arm exited $STATUS instead of 125"
grep -Fx -- 'abort_reason=model_identity_changed_during_arm' \
    "$POST_MUTATING_PREFIX.summary" >/dev/null ||
    fail "model mutation during arm was not retained in the summary"

echo "benchmark-runner-guard: PASS"
