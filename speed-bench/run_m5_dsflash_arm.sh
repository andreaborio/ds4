#!/bin/zsh

# Bounded M5 Pro / 64 GiB DeepSeek-Flash SSD-streaming arm.  Every model run
# forces SSD residency and is killed before sustained swap, low memory
# pressure, or a wired-memory level close to the host user-wire limit.

set -u

if (( $# < 2 || $# > 3 )); then
    print -u2 -- "usage: $0 LABEL auto|auto_pin|exactN|exactN_pin [GEN_TOKENS]"
    exit 2
fi

label=$1
mode=$2
gen_tokens=${3:-64}
ctx_start=${DS4_M5_CTX_START:-128}
ctx_max=${DS4_M5_CTX_MAX:-$ctx_start}
ctx_alloc=${DS4_M5_CTX_ALLOC:-32768}
step_mul=${DS4_M5_STEP_MUL:-1}
root=${0:A:h:h}
bin=${DS4_M5_BIN:-$root/build/metal-arm64/bin/ds4-bench}
model=${DS4_M5_MODEL:-/Users/chinaski/Desktop/ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf}
prompt=${DS4_M5_PROMPT:-$root/tests/long_context_security_prompt.txt}
prefix=${DS4_M5_PREFIX:-${TMPDIR:-/tmp}/ds4-m5-${label}}
preload=${DS4_M5_PRELOAD_EXPERTS:-4096}
max_seconds=${DS4_M5_MAX_SECONDS:-240}
min_free_percent=${DS4_M5_MIN_FREE_PERCENT:-20}
max_swapout_pages=${DS4_M5_MAX_SWAPOUT_PAGES:-0}
max_wired_gib=${DS4_M5_MAX_WIRED_GIB:-46}

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

case $gen_tokens in
    ''|*[!0-9]*) print -u2 -- "GEN_TOKENS must be a positive integer"; exit 2 ;;
esac
(( gen_tokens > 0 )) || { print -u2 -- "GEN_TOKENS must be positive"; exit 2; }

[[ -x $bin ]] || { print -u2 -- "missing executable: $bin"; exit 2; }
[[ -f $model ]] || { print -u2 -- "missing model: $model"; exit 2; }
[[ -f $prompt ]] || { print -u2 -- "missing prompt: $prompt"; exit 2; }
pmset -g batt | head -n 1 | grep -q "AC Power" || {
    print -u2 -- "M5 benchmark requires AC power"
    exit 2
}

for process_name in ds4 ds4-bench ds4-server llama-server; do
    if pgrep -x "$process_name" >/dev/null 2>&1; then
        print -u2 -- "refusing to run beside $process_name"
        exit 2
    fi
done

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

cache_args=()
if [[ $cache != auto ]]; then
    cache_args=(--ssd-streaming-cache-experts "$cache")
fi

if [[ -e $prefix.logits || -e $prefix.summary || -e $prefix.csv ]]; then
    print -u2 -- "refusing to overwrite existing result prefix: $prefix"
    exit 2
fi
mkdir -p -- "$prefix.logits"
vm_stat >"$prefix.vm.before"
sysctl -n vm.swapusage >"$prefix.swap.before"

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
start_epoch=$(date +%s)
abort_reason=
pid=

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

print -- "START label=$label mode=$mode cache=$cache preload=$preload gen=$gen_tokens ctx_start=$ctx_start ctx_max=$ctx_max step_mul=$step_mul ctx_alloc=$ctx_alloc"
"$bin" \
    --metal --ssd-streaming \
    "${cache_args[@]}" \
    --ssd-streaming-preload-experts "$preload" \
    -m "$model" \
    --prompt-file "$prompt" \
    --ctx-start "$ctx_start" --ctx-max "$ctx_max" --step-mul "$step_mul" \
    --ctx-alloc "$ctx_alloc" \
    --gen-tokens "$gen_tokens" \
    --dump-frontier-logits-dir "$prefix.logits" \
    --csv "$prefix.csv" \
    >"$prefix.stdout" 2>"$prefix.stderr" &
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
        (( wired_now > peak_wired_pages )) && peak_wired_pages=$wired_now
        wired_bytes=$((wired_now * page_size))
        rss_now=$(ps -o rss= -p "$pid" 2>/dev/null | awk '{print $1 + 0}')
        if is_uint "$rss_now" && (( rss_now > peak_rss_kib )); then
            peak_rss_kib=$rss_now
        fi

        if (( swapout_delta > max_swapout_pages )); then
            abort_reason="swapout_over_${max_swapout_pages}_pages"
        elif (( free_now < min_free_percent )); then
            abort_reason="free_memory_below_${min_free_percent}_percent"
        elif (( wired_bytes > max_wired_bytes )); then
            abort_reason="wired_memory_over_${max_wired_gib}_GiB"
        elif (( elapsed > max_seconds )); then
            abort_reason="timeout_over_${max_seconds}_seconds"
        fi
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
    wait "$pid"
    rc=$?
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

final_rc=$rc
if [[ -n $abort_reason ]]; then
    final_rc=125
elif (( rc == 0 )) && [[ -n $result_error ]]; then
    final_rc=126
fi

{
    print -- "label=$label"
    print -- "mode=$mode"
    print -- "cache=$cache"
    print -- "preload=$preload"
    print -- "gen_tokens=$gen_tokens"
    print -- "ctx_start=$ctx_start"
    print -- "ctx_max=$ctx_max"
    print -- "step_mul=$step_mul"
    print -- "ctx_alloc=$ctx_alloc"
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
    print -- "pressure_after=$pressure_after"
} >"$prefix.summary"

cat -- "$prefix.summary"
tail -n 1 "$prefix.csv" 2>/dev/null || true
grep -E 'adaptive cache|cached expert count|low-RAM|pinned non-routed|streaming expert cache budget=|streaming expert buffer mlock|task memory footprint|page faults|page reclaims' "$prefix.stderr" || true
cat -- "$prefix.logits.sha256"

exit "$final_rc"
