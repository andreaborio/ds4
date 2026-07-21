#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
BIN_DIR=${DS4_BIN_DIR:-$ROOT}
PROGRAMS="hebrus hebrus-server hebrus-agent hebrus-bench hebrus-eval ds4 ds4-server ds4-agent ds4-bench ds4-eval"
OPTIONS="--role --layers --listen --coordinator --dist-prefill-chunk --dist-prefill-window --dist-activation-bits --dist-replay-check --debug"

fail() {
    echo "retired-distributed-flags: FAIL: $*" >&2
    exit 1
}

run_and_capture() {
    binary=$1
    shift
    set +e
    OUTPUT=$("$binary" "$@" 2>&1)
    STATUS=$?
    set -e
}

output_contains() {
    case "$OUTPUT" in
        *"$1"*) return 0 ;;
        *) return 1 ;;
    esac
}

diagnostic_name() {
    case "$1" in
        hebrus) echo ds4 ;;
        hebrus-server) echo ds4-server ;;
        hebrus-agent) echo ds4-agent ;;
        hebrus-bench) echo ds4-bench ;;
        hebrus-eval) echo ds4-eval ;;
        *) echo "$1" ;;
    esac
}

for program in $PROGRAMS; do
    binary="$BIN_DIR/$program"
    [ -x "$binary" ] || fail "missing executable $binary"

    run_and_capture "$binary" --help
    [ "$STATUS" -eq 0 ] ||
        fail "$program --help exited $STATUS instead of 0"
    for option in $OPTIONS; do
        case "$OUTPUT" in
            *"$option"*)
                fail "$program --help advertises retired option $option"
                ;;
        esac
    done

    run_and_capture "$binary" --help distributed
    [ "$STATUS" -eq 0 ] ||
        fail "$program --help distributed exited $STATUS instead of 0"
    output_contains "Distributed Inference (Retired)" ||
        fail "$program --help distributed omitted the retirement heading"
    output_contains "Distributed inference is outside the supported ExpertMajor v2 runtime and its command-line options have been retired." ||
        fail "$program --help distributed omitted the retirement policy"
    output_contains "There is no supported distributed startup command." ||
        fail "$program --help distributed omitted the no-startup tombstone"
    for option in $OPTIONS; do
        case "$OUTPUT" in
            *"$option"*)
                fail "$program --help distributed advertises retired option $option"
                ;;
        esac
    done

    for option in $OPTIONS; do
        expected="$(diagnostic_name "$program"): distributed option $option was retired; distributed inference is not supported"

        run_and_capture "$binary" "$option"
        [ "$STATUS" -eq 2 ] ||
            fail "$program $option exited $STATUS instead of 2"
        [ "$OUTPUT" = "$expected" ] ||
            fail "$program $option did not emit the canonical retirement error"

        run_and_capture "$binary" "$option=probe"
        [ "$STATUS" -eq 2 ] ||
            fail "$program $option=probe exited $STATUS instead of 2"
        [ "$OUTPUT" = "$expected" ] ||
            fail "$program $option=probe did not normalize to $option"

        near_miss="${option}x=probe"
        run_and_capture "$binary" "$near_miss"
        [ "$STATUS" -eq 2 ] ||
            fail "$program $near_miss exited $STATUS instead of 2"
        case "$OUTPUT" in
            *"distributed option"*)
                fail "$program incorrectly classified near-miss $near_miss"
                ;;
        esac
    done
done

echo "retired-distributed-flags: PASS"
