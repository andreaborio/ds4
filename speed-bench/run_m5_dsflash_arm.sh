#!/bin/zsh

# Bounded M5 Pro / 64 GiB Metal benchmark arm. The default remains DeepSeek
# SSD streaming; DS4_M5_RESIDENCY=auto|resident reuses the same guard for GLM
# and Qwen. Every run is killed before sustained swap or a wired-memory level
# close to the host user-wire limit; transient pressure is recorded as evidence.

set -euo pipefail

if (( $# < 2 || $# > 3 )); then
    print -u2 -- "usage: $0 LABEL auto|auto_pin|exactN|exactN_pin [GEN_TOKENS]"
    exit 2
fi

label=$1
mode=$2
gen_tokens=${3:-128}
ctx_start=${DS4_M5_CTX_START:-128}
ctx_max=${DS4_M5_CTX_MAX:-$ctx_start}
ctx_alloc=${DS4_M5_CTX_ALLOC:-32768}
step_mul=${DS4_M5_STEP_MUL:-1}
root=${DS4_M5_ROOT:-${0:A:h:h}}
bin=${DS4_M5_BIN:-$root/build/metal-arm64/bin/ds4-bench}
model=${DS4_M5_MODEL:-}
model_sha256_expected=${DS4_M5_MODEL_SHA256:-}
prompt=${DS4_M5_PROMPT:-$root/tests/long_context_security_prompt.txt}
prompt_source=$prompt
prefix=${DS4_M5_PREFIX:-${TMPDIR:-/tmp}/ds4-m5-${label}}
preload=${DS4_M5_PRELOAD_EXPERTS:-4096}
residency=${DS4_M5_RESIDENCY:-ssd}
max_seconds=${DS4_M5_MAX_SECONDS:-240}
min_free_percent=${DS4_M5_MIN_FREE_PERCENT:-20}
max_swapout_pages=${DS4_M5_MAX_SWAPOUT_PAGES:-0}
max_wired_gib=${DS4_M5_MAX_WIRED_GIB:-46}
cache_state=${DS4_M5_CACHE_STATE:-unclassified}
exploratory=${DS4_M5_EXPLORATORY:-0}

case $mode in
    auto)          cache=auto; pin=0 ;;
    auto_pin)      cache=auto; pin=1 ;;
    exact<->)      cache=${mode#exact}; pin=0 ;;
    exact<->_pin)  cache=${mode#exact}; cache=${cache%_pin}; pin=1 ;;
    *)
        print -u2 -- "invalid mode: $mode"
        exit 2
        ;;
esac

case $residency in
    ssd|auto|resident) ;;
    *) print -u2 -- "DS4_M5_RESIDENCY must be ssd, auto, or resident"; exit 2 ;;
esac
if [[ $residency != ssd && ( $cache != auto || $pin != 0 ) ]]; then
    print -u2 -- "exact cache and pin modes require DS4_M5_RESIDENCY=ssd"
    exit 2
fi

case $gen_tokens in
    ''|*[!0-9]*) print -u2 -- "GEN_TOKENS must be a positive integer"; exit 2 ;;
esac
(( gen_tokens > 0 )) || { print -u2 -- "GEN_TOKENS must be positive"; exit 2; }
case $exploratory in
    0|1) ;;
    *) print -u2 -- "DS4_M5_EXPLORATORY must be 0 or 1"; exit 2 ;;
esac
if (( ! exploratory && gen_tokens < 128 )); then
    print -u2 -- "acceptance arms require at least 128 decode tokens; set DS4_M5_EXPLORATORY=1 for a smoke run"
    exit 2
fi
case $cache_state in
    warm|cold|unclassified|exploratory) ;;
    *) print -u2 -- "DS4_M5_CACHE_STATE must be warm, cold, unclassified, or exploratory"; exit 2 ;;
esac
if (( ! exploratory )) && [[ $cache_state == exploratory ]]; then
    print -u2 -- "cache_state=exploratory requires DS4_M5_EXPLORATORY=1"
    exit 2
fi
if (( exploratory )) && [[ $cache_state == unclassified ]]; then
    cache_state=exploratory
elif (( ! exploratory )) && [[ $cache_state == unclassified ]]; then
    print -u2 -- "acceptance arms require DS4_M5_CACHE_STATE=warm or cold"
    exit 2
fi

# Acceptance evidence must be hermetic.  A flag inherited from an interactive
# profiling shell can otherwise change GLM/MoE/Metal behavior without appearing
# in the benchmark label.  Runner controls use DS4_M5_*; the two telemetry
# variables below are overwritten by this script.  Exploratory arms may retain
# other DS4_* flags, but record every one of them in the environment artifact.
unexpected_ds4_env=$(
    env | awk -F= '
        /^DS4_/ {
            key = $1
            if (key !~ /^DS4_M5_/ &&
                key != "DS4_METAL_MEMORY_REPORT" &&
                key != "DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY") {
                print key
            }
        }' | LC_ALL=C sort -u
)
if [[ -n $unexpected_ds4_env ]] && (( ! exploratory )); then
    print -u2 -- "acceptance arm refuses unexpected DS4 runtime environment:"
    print -u2 -- "$unexpected_ds4_env"
    print -u2 -- "unset these variables or set DS4_M5_EXPLORATORY=1 for a recorded experiment"
    exit 2
fi
if [[ -n $unexpected_ds4_env ]]; then
    print -u2 -- "exploratory arm records unexpected DS4 runtime environment:"
    print -u2 -- "$unexpected_ds4_env"
fi

[[ -n $model ]] || { print -u2 -- "set DS4_M5_MODEL to the ExpertMajor v2 GGUF path"; exit 2; }
[[ ${#model_sha256_expected} == 64 && $model_sha256_expected != *[!0-9a-fA-F]* ]] || {
    print -u2 -- "set DS4_M5_MODEL_SHA256 to the expected 64-digit GGUF SHA-256 from the campaign manifest"
    exit 2
}
[[ -x $bin ]] || { print -u2 -- "missing executable: $bin"; exit 2; }
[[ -f $model ]] || { print -u2 -- "missing model: $model"; exit 2; }
[[ -f $prompt ]] || { print -u2 -- "missing prompt: $prompt"; exit 2; }
pmset -g batt | head -n 1 | grep -q "AC Power" || {
    print -u2 -- "M5 benchmark requires AC power"
    exit 2
}

pid=
inference_processes() {
    ps -axo pid=,ppid=,comm= | awk -v self="$$" -v child="${pid:-0}" '
        {
            process_pid = $1
            process_ppid = $2
            command = $3
            sub(/^.*\//, "", command)
            if (process_pid != self && process_pid != child &&
                command ~ /^(ds4|llama-server)/) {
                print process_pid, process_ppid, command
            }
        }'
}

competitors=$(inference_processes)
if [[ -n $competitors ]]; then
    print -u2 -- "refusing to run beside another inference process:"
    print -u2 -- "$competitors"
    exit 2
fi

vm_value() {
    vm_stat | awk -v key="$1:" '
        index($0, key) == 1 {
            line = $0
            sub(/^[^:]*:[[:space:]]*/, "", line)
            gsub(/\./, "", line)
            gsub(/[[:space:]]/, "", line)
            if (line ~ /^[0-9]+$/) print line
            exit
        }'
}

free_percent() {
    memory_pressure -Q 2>/dev/null |
        awk '/System-wide memory free percentage:/ {
            gsub(/%/, "", $5)
            if ($5 ~ /^[0-9]+$/) print $5
            exit
        }'
}

is_uint() {
    case ${1:-} in
        ''|*[!0-9]*) return 1 ;;
        *) return 0 ;;
    esac
}

require_uint() {
    local name=$1
    local value=${2:-}
    is_uint "$value" || {
        print -u2 -- "invalid or unavailable telemetry/config: $name=${value:-<empty>}"
        exit 2
    }
}

require_uint DS4_M5_MAX_SECONDS "$max_seconds"
require_uint DS4_M5_MIN_FREE_PERCENT "$min_free_percent"
require_uint DS4_M5_MAX_SWAPOUT_PAGES "$max_swapout_pages"
require_uint DS4_M5_MAX_WIRED_GIB "$max_wired_gib"
require_uint DS4_M5_CTX_START "$ctx_start"
require_uint DS4_M5_CTX_MAX "$ctx_max"
require_uint DS4_M5_CTX_ALLOC "$ctx_alloc"
require_uint DS4_M5_STEP_MUL "$step_mul"
if (( ! exploratory && min_free_percent < 20 )); then
    print -u2 -- "acceptance arms require DS4_M5_MIN_FREE_PERCENT >= 20"
    exit 2
fi
(( ctx_start > 0 )) || { print -u2 -- "DS4_M5_CTX_START must be positive"; exit 2; }
(( ctx_max >= ctx_start )) || { print -u2 -- "DS4_M5_CTX_MAX must be >= DS4_M5_CTX_START"; exit 2; }
(( step_mul > 0 )) || { print -u2 -- "DS4_M5_STEP_MUL must be positive"; exit 2; }
(( ctx_alloc > ctx_max + gen_tokens )) || {
    print -u2 -- "DS4_M5_CTX_ALLOC must exceed ctx_max + gen_tokens"
    exit 2
}

page_size=$(pagesize)
require_uint page_size "$page_size"
max_wired_bytes=$((max_wired_gib * 1024 * 1024 * 1024))

unset DS4_METAL_STREAMING_PIN_NON_ROUTED
unset DS4_METAL_STREAMING_PIN_STATIC
unset DS4_METAL_STREAMING_EXPERT_NO_RDAHEAD
unset DS4_METAL_DISABLE_STREAMING_EXPERT_READAHEAD
if (( pin )); then
    export DS4_METAL_STREAMING_PIN_NON_ROUTED=1
fi
export DS4_METAL_MEMORY_REPORT=1
export DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY=1

runtime_args=()
case $residency in
    ssd)
        runtime_args=(--ssd-streaming --ssd-streaming-preload-experts "$preload")
        if [[ $cache != auto ]]; then
            runtime_args+=(--ssd-streaming-cache-experts "$cache")
        fi
        ;;
    resident) runtime_args=(--resident) ;;
    auto) ;;
esac

if [[ -e $prefix.logits || -e $prefix.evidence || -e $prefix.summary ||
      -e $prefix.csv || -e $prefix.prompt.txt ]]; then
    print -u2 -- "refusing to overwrite existing result prefix: $prefix"
    exit 2
fi

# The canonical checked-in prompt is intentionally compact. For the additional
# 65K/100K gates, extend it deterministically instead of committing a duplicated
# megabyte-scale fixture. Eight source bytes per requested token is deliberately
# conservative for the qualified model tokenizers; ds4-bench remains the final
# token-count authority and fails closed if the generated prompt is insufficient.
prompt_expanded=0
prompt_minimum_bytes=$((ctx_max * 8))
prompt_source_bytes=$(stat -f %z "$prompt_source")
if (( ctx_max > 32768 && prompt_source_bytes < prompt_minimum_bytes )); then
    prompt="$prefix.prompt.txt"
    python3 "$root/speed-bench/build_long_context_prompt.py" \
        --source "$prompt_source" \
        --output "$prompt" \
        --minimum-bytes "$prompt_minimum_bytes"
    prompt_expanded=1
fi
mkdir -p -- "$prefix.logits"
mkdir -p -- "$prefix.evidence"

# Make every artifact self-identifying.  Labels alone cannot distinguish two
# dirty builds or reconstruct which opt-in ablation flags were active.
bin_sha256=$(shasum -a 256 "$bin" | awk '{print $1}')
repo_head=$(git -C "$root" rev-parse HEAD 2>/dev/null || print unknown)
repo_diff_sha256=$(
    git -C "$root" diff HEAD --binary 2>/dev/null | shasum -a 256 | awk '{print $1}'
)
# `git diff` excludes untracked implementation partitions. Preserve a separate
# content manifest and fold it into one source-state identity so a dirty
# benchmark cannot silently omit code that the compiler consumed.
repo_untracked_manifest="$prefix.repo-untracked.sha256"
(
    cd "$root"
    git ls-files --others --exclude-standard -z |
        sort -z |
        while IFS= read -r -d '' untracked_path; do
            shasum -a 256 -- "$untracked_path"
        done
) >"$repo_untracked_manifest"
repo_untracked_count=$(wc -l <"$repo_untracked_manifest" | tr -d ' ')
repo_untracked_manifest_sha256=$(
    shasum -a 256 "$repo_untracked_manifest" | awk '{print $1}'
)
repo_source_state_sha256=$(
    {
        print -r -- "tracked_diff $repo_diff_sha256"
        print -r -- "untracked_manifest $repo_untracked_manifest_sha256"
    } | shasum -a 256 | awk '{print $1}'
)
git -C "$root" status --porcelain=v1 >"$prefix.git-status"
metal_files=("$root"/metal/**/*.metal(N))
(( ${#metal_files} > 0 )) || {
    print -u2 -- "no runtime Metal sources found under $root/metal"
    exit 2
}
metal_file_set_manifest_sha256=$(
    cd "$root"
    relative_metal_files=(metal/**/*.metal(N))
    printf '%s\0' "${relative_metal_files[@]}" |
        sort -z | xargs -0 shasum -a 256 |
        shasum -a 256 | awk '{print $1}'
)
model_bytes=$(stat -f %z "$model")
model_mtime=$(stat -f %m "$model")
prompt_source_sha256=$(shasum -a 256 "$prompt_source" | awk '{print $1}')
prompt_sha256=$(shasum -a 256 "$prompt" | awk '{print $1}')
prompt_bytes=$(stat -f %z "$prompt")
"$bin" --build-info >"$prefix.build-info"
[[ -s $prefix.build-info ]] || {
    print -u2 -- "benchmark executable returned no build identity"
    exit 2
}
env | LC_ALL=C sort |
    awk -F= '/^DS4_/ { print }' >"$prefix.env"
vm_stat >"$prefix.vm.before"
sysctl -n vm.swapusage >"$prefix.swap.before"
os_build=$(sw_vers -buildVersion)
power_source=$(pmset -g batt | head -n 1 | sed -E "s/^Now drawing from '(.*)'$/\1/")

swapout_before=$(vm_value Swapouts)
pagein_before=$(vm_value Pageins)
wired_before=$(vm_value "Pages wired down")
pressure_before=$(free_percent)
require_uint swapout_before "$swapout_before"
require_uint pagein_before "$pagein_before"
require_uint wired_before "$wired_before"
require_uint pressure_before "$pressure_before"
print -- "$pressure_before" >"$prefix.pressure.before"
wired_before_bytes=$((wired_before * page_size))
if (( pressure_before < min_free_percent )); then
    print -u2 -- "refusing to launch: free memory ${pressure_before}% is below ${min_free_percent}%"
    exit 125
fi
if (( wired_before_bytes > max_wired_bytes )); then
    print -u2 -- "refusing to launch: wired memory exceeds ${max_wired_gib} GiB"
    exit 125
fi
peak_wired_pages=$wired_before
peak_rss_kib=0
pressure_min=$pressure_before
start_epoch=$(date +%s)
abort_reason=
process_contamination=

terminate_tree() {
    local target=${pid:-}
    [[ -n $target ]] || return 0
    if kill -0 "$target" 2>/dev/null; then
        pkill -TERM -P "$target" 2>/dev/null || true
        kill -TERM "$target" 2>/dev/null || true
        local attempt
        for attempt in {1..20}; do
            kill -0 "$target" 2>/dev/null || break
            sleep 0.1
        done
        if kill -0 "$target" 2>/dev/null; then
            pkill -KILL -P "$target" 2>/dev/null || true
            kill -KILL "$target" 2>/dev/null || true
        fi
    fi
    wait "$target" 2>/dev/null || true
    pid=
}

trap 'terminate_tree' EXIT
trap 'abort_reason=signal_hup; terminate_tree; exit 130' HUP
trap 'abort_reason=signal_int; terminate_tree; exit 130' INT
trap 'abort_reason=signal_term; terminate_tree; exit 130' TERM

print -- "START label=$label residency=$residency mode=$mode cache=$cache preload=$preload gen=$gen_tokens ctx_start=$ctx_start ctx_max=$ctx_max step_mul=$step_mul ctx_alloc=$ctx_alloc"
(
    cd "$root" || exit 2
    "$bin" \
        --metal \
        "${runtime_args[@]}" \
        -m "$model" \
        --prompt-file "$prompt" \
        --ctx-start "$ctx_start" --ctx-max "$ctx_max" --step-mul "$step_mul" \
        --ctx-alloc "$ctx_alloc" \
        --gen-tokens "$gen_tokens" \
        --dump-frontier-logits-dir "$prefix.logits" \
        --dump-decode-evidence-dir "$prefix.evidence" \
        --csv "$prefix.csv"
) >"$prefix.stdout" 2>"$prefix.stderr" &
pid=$!
run_pid=$pid
print -- "$pid" >"$prefix.pid"

while kill -0 "$pid" 2>/dev/null; do
    now=$(date +%s)
    elapsed=$((now - start_epoch))
    swapout_now=$(vm_value Swapouts)
    free_now=$(free_percent)
    wired_now=$(vm_value "Pages wired down")
    if ! is_uint "$swapout_now"; then
        abort_reason=telemetry_swapout_unavailable
    elif ! is_uint "$free_now"; then
        abort_reason=telemetry_memory_pressure_unavailable
    elif ! is_uint "$wired_now"; then
        abort_reason=telemetry_wired_unavailable
    elif (( swapout_now < swapout_before )); then
        abort_reason=telemetry_swapout_counter_reset
    else
        swapout_delta=$((swapout_now - swapout_before))
        (( free_now < pressure_min )) && pressure_min=$free_now
        (( wired_now > peak_wired_pages )) && peak_wired_pages=$wired_now
        wired_bytes=$((wired_now * page_size))
        rss_now=$(ps -o rss= -p "$pid" 2>/dev/null | awk '{print $1 + 0}')
        if is_uint "$rss_now" && (( rss_now > peak_rss_kib )); then
            peak_rss_kib=$rss_now
        fi

        if (( swapout_delta > max_swapout_pages )); then
            abort_reason="swapout_over_${max_swapout_pages}_pages"
        elif (( wired_bytes > max_wired_bytes )); then
            abort_reason="wired_memory_over_${max_wired_gib}_GiB"
        elif (( elapsed > max_seconds )); then
            abort_reason="timeout_over_${max_seconds}_seconds"
        fi
    fi

    competitors=$(inference_processes)
    if [[ -n $competitors ]]; then
        process_contamination=$competitors
        abort_reason=concurrent_inference_process
    fi

    if [[ -n $abort_reason ]]; then
        break
    fi
    sleep 1
done

if [[ -n $abort_reason ]]; then
    terminate_tree
    rc=125
else
    if wait "$pid"; then
        rc=0
    else
        rc=$?
    fi
    pid=
fi

end_epoch=$(date +%s)
vm_stat >"$prefix.vm.after"
sysctl -n vm.swapusage >"$prefix.swap.after"
swapout_after=$(vm_value Swapouts)
pagein_after=$(vm_value Pageins)
wired_after=$(vm_value "Pages wired down")
pressure_after=$(free_percent)
require_uint swapout_after "$swapout_after"
require_uint pagein_after "$pagein_after"
require_uint wired_after "$wired_after"
require_uint pressure_after "$pressure_after"
print -- "$pressure_after" >"$prefix.pressure.after"
(( pressure_after < pressure_min )) && pressure_min=$pressure_after
print -- "$pressure_min" >"$prefix.pressure.min"

competitors=$(inference_processes)
if [[ -n $competitors ]]; then
    process_contamination=$competitors
    abort_reason=concurrent_inference_process
    rc=125
fi
print -- "$process_contamination" >"$prefix.process-contamination"

# Swapouts are monotonic across process exit, unlike pressure and wired memory,
# so this final sample closes the race between the last monitor poll and exit.
if (( swapout_after < swapout_before )); then
    abort_reason=telemetry_swapout_counter_reset
    rc=125
elif (( swapout_after - swapout_before > max_swapout_pages )); then
    abort_reason="swapout_over_${max_swapout_pages}_pages"
    rc=125
fi

logit_files=("$prefix.logits"/**/*.json(N))
result_error=
if (( ${#logit_files} == 0 )); then
    : >"$prefix.logits.sha256"
    result_error=missing_frontier_logits
    print -u2 -- "benchmark produced no frontier-logits JSON"
elif ! (
    set -o pipefail
    printf '%s\0' "${logit_files[@]}" |
        sort -z | xargs -0 shasum -a 256 >"$prefix.logits.sha256"
) || [[ ! -s $prefix.logits.sha256 ]]; then
    result_error=frontier_logits_hash_failed
    print -u2 -- "could not produce a complete frontier-logits checksum"
fi
if [[ -s $prefix.logits.sha256 ]]; then
    awk '{ print $1 }' "$prefix.logits.sha256" |
        shasum -a 256 | awk '{ print $1 }' >"$prefix.logits.content.sha256"
else
    : >"$prefix.logits.content.sha256"
fi

evidence_files=("$prefix.evidence"/**/*.json(N))
if (( ${#evidence_files} == 0 )); then
    : >"$prefix.evidence.sha256"
    [[ -n $result_error ]] || result_error=missing_decode_evidence
    print -u2 -- "benchmark produced no decode-evidence JSON"
elif ! (
    set -o pipefail
    printf '%s\0' "${evidence_files[@]}" |
        sort -z | xargs -0 shasum -a 256 >"$prefix.evidence.sha256"
) || [[ ! -s $prefix.evidence.sha256 ]]; then
    [[ -n $result_error ]] || result_error=decode_evidence_hash_failed
    print -u2 -- "could not produce a complete decode-evidence checksum"
fi
if [[ -s $prefix.evidence.sha256 ]]; then
    awk '{ print $1 }' "$prefix.evidence.sha256" |
        shasum -a 256 | awk '{ print $1 }' >"$prefix.evidence.content.sha256"
else
    : >"$prefix.evidence.content.sha256"
fi

grep -m 1 '^ds4: metal_library ' "$prefix.stderr" >"$prefix.metal-library" || true
if [[ ! -s $prefix.metal-library ]]; then
    [[ -n $result_error ]] || result_error=missing_runtime_metal_identity
    print -u2 -- "benchmark produced no runtime Metal library identity"
fi

grep -E '^ds4: (build |residency requested=|effective profile=|SSD streaming adaptive cache budget|  host physical |  safety cache budget |  cached expert count:|  .*ExpertMajor AUTO selected |  native prefill cache phase:|DeepSeek native cache phase |memory:|memory detail:)' \
    "$prefix.stderr" >"$prefix.resolved-plan" || true
if [[ -s $prefix.resolved-plan ]]; then
    resolved_plan_sha256=$(shasum -a 256 "$prefix.resolved-plan" | awk '{ print $1 }')
else
    resolved_plan_sha256=missing
    [[ -n $result_error ]] || result_error=missing_resolved_plan
fi

final_rc=$rc
if [[ -n $abort_reason ]]; then
    final_rc=125
elif (( rc == 0 )) && [[ -n $result_error ]]; then
    final_rc=126
fi

{
    print -- "label=$label"
    print -- "mode=$mode"
    print -- "residency=$residency"
    print -- "cache=$cache"
    print -- "preload=$preload"
    print -- "gen_tokens=$gen_tokens"
    print -- "ctx_start=$ctx_start"
    print -- "ctx_max=$ctx_max"
    print -- "step_mul=$step_mul"
    print -- "ctx_alloc=$ctx_alloc"
    print -- "bin=$bin"
    print -- "bin_sha256=$bin_sha256"
    print -- "repo_head=$repo_head"
    print -- "repo_diff_sha256=$repo_diff_sha256"
    print -- "repo_untracked_count=$repo_untracked_count"
    print -- "repo_untracked_manifest=$repo_untracked_manifest"
    print -- "repo_untracked_manifest_sha256=$repo_untracked_manifest_sha256"
    print -- "repo_source_state_sha256=$repo_source_state_sha256"
    print -- "repo_status_file=$prefix.git-status"
    print -- "metal_file_set_manifest_sha256=$metal_file_set_manifest_sha256"
    print -- "metal_library_identity_file=$prefix.metal-library"
    print -- "resolved_plan_file=$prefix.resolved-plan"
    print -- "resolved_plan_sha256=$resolved_plan_sha256"
    print -- "model=$model"
    print -- "model_sha256_expected=${model_sha256_expected:l}"
    print -- "model_sha256_verification=expected-only"
    print -- "model_bytes=$model_bytes"
    print -- "model_mtime_epoch=$model_mtime"
    print -- "prompt_source=$prompt_source"
    print -- "prompt_source_sha256=$prompt_source_sha256"
    print -- "prompt=$prompt"
    print -- "prompt_sha256=$prompt_sha256"
    print -- "prompt_bytes=$prompt_bytes"
    print -- "prompt_expanded=$prompt_expanded"
    print -- "os_build=$os_build"
    print -- "power_source=$power_source"
    print -- "cache_state=$cache_state"
    print -- "exploratory=$exploratory"
    print -- "env_file=$prefix.env"
    print -- "pid=$run_pid"
    print -- "process_rc=$rc"
    print -- "rc=$final_rc"
    print -- "elapsed_seconds=$((end_epoch - start_epoch))"
    print -- "abort_reason=${abort_reason:-none}"
    print -- "result_error=${result_error:-none}"
    print -- "pagein_pages_delta=$((pagein_after - pagein_before))"
    print -- "swapout_pages_delta=$((swapout_after - swapout_before))"
    print -- "peak_rss_mib=$((peak_rss_kib / 1024))"
    print -- "wired_before_mib=$((wired_before * page_size / 1024 / 1024))"
    print -- "wired_peak_mib=$((peak_wired_pages * page_size / 1024 / 1024))"
    print -- "wired_after_mib=$((wired_after * page_size / 1024 / 1024))"
    print -- "pressure_before=$pressure_before"
    print -- "pressure_min=$pressure_min"
    print -- "pressure_after=$pressure_after"
    print -- "process_contamination=${process_contamination:-none}"
} >"$prefix.summary"

cat -- "$prefix.summary"
tail -n 1 "$prefix.csv" 2>/dev/null || true
grep -E 'adaptive cache|cached expert count|low-RAM|pinned non-routed|streaming expert cache budget=|streaming expert buffer mlock|task memory footprint|page faults|page reclaims' "$prefix.stderr" || true
cat -- "$prefix.logits.sha256"
cat -- "$prefix.logits.content.sha256"
cat -- "$prefix.evidence.sha256"
cat -- "$prefix.evidence.content.sha256"

exit "$final_rc"
