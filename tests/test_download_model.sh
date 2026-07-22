#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
SCRIPT="$ROOT/download_model.sh"
QWEN_CONTRACT="$ROOT/docs/contracts/qwen-release.json"
TMP_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/ds4-download-model.XXXXXX")
trap 'rm -rf "$TMP_ROOT"' EXIT HUP INT TERM
HOME="$TMP_ROOT/home"
export HOME
unset HF_TOKEN
mkdir -p "$HOME"

QWEN_EXPECTED="$TMP_ROOT/qwen-release-contract.sh"
python3 - "$QWEN_CONTRACT" >"$QWEN_EXPECTED" <<'PY'
import json
import shlex
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    contract = json.load(handle)
published = contract["publishedArtifact"]
negative = contract["negativeArtifact"]
values = {
    "QWEN_STATUS": published["status"],
    "QWEN_REPOSITORY": contract["repository"],
    "QWEN_FILE": published["filename"],
    "QWEN_BYTES": str(published["bytes"]),
    "QWEN_SHA256": published["sha256"],
    "QWEN_REVISION": published["revision"],
    "QWEN_RUNTIME_COMMIT": published["runtimeCommit"],
    "QWEN_NEGATIVE_FILE": negative["filename"],
    "QWEN_NEGATIVE_SHA256": negative["sha256"],
}
for name, value in values.items():
    print(f"{name}={shlex.quote(value)}")
PY
. "$QWEN_EXPECTED"

fail() {
    echo "download-model: FAIL: $*" >&2
    exit 1
}

assert_exact_line() {
    expected=$1
    grep -Fqx -- "$expected" "$SCRIPT" ||
        fail "missing production contract: $expected"
}

assert_contains() {
    file=$1
    expected=$2
    grep -Fq -- "$expected" "$file" ||
        fail "$file does not contain: $expected"
}

assert_case_assignment() {
    target=$1
    assignment=$2
    sed -n "/^    $target)$/, /^        ;;$/p" "$SCRIPT" |
        grep -Fqx -- "        $assignment" ||
        fail "$target does not contain the production assignment: $assignment"
}

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        fail "sha256sum or shasum is required"
    fi
}

run_expect_failure() {
    stdout_file=$1
    stderr_file=$2
    shift 2
    set +e
    "$@" >"$stdout_file" 2>"$stderr_file"
    status=$?
    set -e
    [ "$status" -ne 0 ] || fail "command unexpectedly succeeded: $*"
}

assert_exact_line 'RUNTIME_REVISION="ds4-v0.2.0"'
assert_exact_line 'RUNTIME_DEEPSEEK_BYTES=86720114272'
assert_exact_line 'RUNTIME_DEEPSEEK_SHA256="8378080263eb9224f7228d72e2afa4ac3cf74a116023fdec2c596ff228a33e3f"'
assert_exact_line 'RUNTIME_GLM_BYTES=262147193504'
assert_exact_line 'RUNTIME_GLM_SHA256="7f5017e3076e706c78f2a5322b035a9e2f6519c65ff5b6be8b2d91aeff61505d"'
assert_exact_line "RUNTIME_QWEN_STATUS=\"$QWEN_STATUS\""
assert_exact_line "RUNTIME_QWEN_REPO=\"$QWEN_REPOSITORY\""
assert_exact_line "RUNTIME_QWEN_FILE=\"$QWEN_FILE\""
assert_exact_line "RUNTIME_QWEN_BYTES=$QWEN_BYTES"
assert_exact_line "RUNTIME_QWEN_SHA256=\"$QWEN_SHA256\""
assert_exact_line "RUNTIME_QWEN_REVISION=\"$QWEN_REVISION\""
assert_exact_line "RUNTIME_QWEN_MIN_RUNTIME_COMMIT=\"$QWEN_RUNTIME_COMMIT\""
[ "$QWEN_STATUS" = published ] || fail "canonical Qwen artifact is not published"
if grep -Fq -- "$QWEN_NEGATIVE_FILE" "$SCRIPT" ||
        grep -Fq -- "$QWEN_NEGATIVE_SHA256" "$SCRIPT"; then
    fail "negative-only Qwen artifact is exposed by the downloader"
fi
assert_case_assignment deepseek-v2 'MODEL_REPO=$RUNTIME_DEEPSEEK_REPO'
assert_case_assignment deepseek-v2 'MODEL_FILE=$RUNTIME_DEEPSEEK_FILE'
assert_case_assignment deepseek-v2 'MODEL_BYTES=$RUNTIME_DEEPSEEK_BYTES'
assert_case_assignment deepseek-v2 'MODEL_SHA256=$RUNTIME_DEEPSEEK_SHA256'
assert_case_assignment glm-v2 'MODEL_REPO=$RUNTIME_GLM_REPO'
assert_case_assignment glm-v2 'MODEL_FILE=$RUNTIME_GLM_FILE'
assert_case_assignment glm-v2 'MODEL_BYTES=$RUNTIME_GLM_BYTES'
assert_case_assignment glm-v2 'MODEL_SHA256=$RUNTIME_GLM_SHA256'
assert_case_assignment qwen-v2 'MODEL_REPO=$RUNTIME_QWEN_REPO'
assert_case_assignment qwen-v2 'MODEL_FILE=$RUNTIME_QWEN_FILE'
assert_case_assignment qwen-v2 'MODEL_REVISION=$RUNTIME_QWEN_REVISION'
assert_case_assignment qwen-v2 'MODEL_BYTES=$RUNTIME_QWEN_BYTES'
assert_case_assignment qwen-v2 'MODEL_SHA256=$RUNTIME_QWEN_SHA256'
sh -n "$SCRIPT" || fail "download_model.sh has invalid shell syntax"

# Exercise the real downloader logic without a production identity override.
# Only this private temporary copy receives tiny fixture identities.
FIXTURE_FILE="$TMP_ROOT/qualified-runtime.gguf"
printf '%s' 'qualified-runtime-fixture' >"$FIXTURE_FILE"
FIXTURE_BYTES=$(wc -c <"$FIXTURE_FILE" | tr -d '[:space:]')
FIXTURE_SHA256=$(sha256_file "$FIXTURE_FILE")
TEST_SCRIPT="$TMP_ROOT/download_model.sh"
sed \
    -e "s/^RUNTIME_DEEPSEEK_BYTES=.*/RUNTIME_DEEPSEEK_BYTES=$FIXTURE_BYTES/" \
    -e "s/^RUNTIME_DEEPSEEK_SHA256=.*/RUNTIME_DEEPSEEK_SHA256=\"$FIXTURE_SHA256\"/" \
    -e "s/^RUNTIME_GLM_BYTES=.*/RUNTIME_GLM_BYTES=$FIXTURE_BYTES/" \
    -e "s/^RUNTIME_GLM_SHA256=.*/RUNTIME_GLM_SHA256=\"$FIXTURE_SHA256\"/" \
    -e "s/^RUNTIME_QWEN_BYTES=.*/RUNTIME_QWEN_BYTES=$FIXTURE_BYTES/" \
    -e "s/^RUNTIME_QWEN_SHA256=.*/RUNTIME_QWEN_SHA256=\"$FIXTURE_SHA256\"/" \
    "$SCRIPT" >"$TEST_SCRIPT"
chmod +x "$TEST_SCRIPT"

FAKE_BIN="$TMP_ROOT/bin"
FAKE_HF_LOG="$TMP_ROOT/hf.log"
mkdir -p "$FAKE_BIN"
cat >"$FAKE_BIN/hf" <<'EOF'
#!/bin/sh
set -eu

: "${FAKE_HF_LOG:?}"
if [ "${FAKE_HF_FAIL_IF_CALLED:-0}" = 1 ]; then
    echo "fake hf was called unexpectedly" >&2
    exit 99
fi

printf '%s\n' "$@" >"$FAKE_HF_LOG"
[ "$1" = download ] || exit 97
shift
repo=$1
model_file=$2
shift 2
local_dir=
while [ $# -gt 0 ]; do
    case "$1" in
        --local-dir)
            shift
            local_dir=$1
            ;;
        --repo-type|--revision|--token)
            shift
            [ $# -gt 0 ] || exit 96
            ;;
        *) exit 95 ;;
    esac
    shift
done
[ -n "$repo" ] && [ -n "$local_dir" ] || exit 94
mkdir -p "$local_dir"
printf '%s' "${FAKE_HF_PAYLOAD:-qualified-runtime-fixture}" >"$local_dir/$model_file"
EOF
chmod +x "$FAKE_BIN/hf"

run_runtime_success() {
    target=$1
    repo=$2
    model_file=$3
    revision=$4
    out_dir="$TMP_ROOT/out-$target"
    stdout_file="$TMP_ROOT/$target.stdout"
    stderr_file="$TMP_ROOT/$target.stderr"
    : >"$FAKE_HF_LOG"

    PATH="$FAKE_BIN:$PATH" FAKE_HF_LOG="$FAKE_HF_LOG" \
        DS4_GGUF_DIR="$out_dir" "$TEST_SCRIPT" "$target" \
        >"$stdout_file" 2>"$stderr_file"
    cmp "$FIXTURE_FILE" "$out_dir/$model_file" >/dev/null ||
        fail "$target did not retain the qualified fixture bytes"

    expected_log="$TMP_ROOT/$target.expected"
    cat >"$expected_log" <<EOF
download
$repo
$model_file
--repo-type
model
--local-dir
$out_dir
--revision
$revision
EOF
    cmp "$expected_log" "$FAKE_HF_LOG" >/dev/null ||
        fail "$target did not pin the exact repository revision"
    assert_contains "$stdout_file" "Verified SHA-256:  $FIXTURE_SHA256"

    : >"$FAKE_HF_LOG"
    PATH="$FAKE_BIN:$PATH" FAKE_HF_LOG="$FAKE_HF_LOG" \
        FAKE_HF_FAIL_IF_CALLED=1 DS4_GGUF_DIR="$out_dir" \
        "$TEST_SCRIPT" "$target" >"$stdout_file" 2>"$stderr_file"
    [ ! -s "$FAKE_HF_LOG" ] || fail "$target called hf for a verified existing file"
}

run_runtime_success \
    deepseek-v2 \
    andreaborio/DeepSeek-V4-Flash-Hebrus-GGUF \
    DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-DS4-ExpertMajor-v2.gguf \
    ds4-v0.2.0
run_runtime_success \
    glm-v2 \
    andreaborio/GLM-5.2-Hebrus-GGUF \
    GLM-5.2-DS4-ExpertMajor-v2-Q2_K.gguf \
    ds4-v0.2.0
run_runtime_success \
    qwen-v2 \
    "$QWEN_REPOSITORY" \
    "$QWEN_FILE" \
    "$QWEN_REVISION"

CORRUPT_DIR="$TMP_ROOT/corrupt-existing"
CORRUPT_FILE="$CORRUPT_DIR/GLM-5.2-DS4-ExpertMajor-v2-Q2_K.gguf"
mkdir -p "$CORRUPT_DIR"
printf '%s' 'corrupt' >"$CORRUPT_FILE"
: >"$FAKE_HF_LOG"
run_expect_failure "$TMP_ROOT/corrupt.stdout" "$TMP_ROOT/corrupt.stderr" \
    env PATH="$FAKE_BIN:$PATH" FAKE_HF_LOG="$FAKE_HF_LOG" \
    FAKE_HF_FAIL_IF_CALLED=1 DS4_GGUF_DIR="$CORRUPT_DIR" \
    "$TEST_SCRIPT" glm-v2
assert_contains "$TMP_ROOT/corrupt.stderr" 'Runtime artifact byte size mismatch'
[ ! -s "$FAKE_HF_LOG" ] || fail "corrupt existing file triggered a download"

WRONG_HASH_DIR="$TMP_ROOT/wrong-hash-existing"
WRONG_HASH_FILE="$WRONG_HASH_DIR/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-DS4-ExpertMajor-v2.gguf"
mkdir -p "$WRONG_HASH_DIR"
printf '%s' 'qualified-runtime-fixturE' >"$WRONG_HASH_FILE"
[ "$(wc -c <"$WRONG_HASH_FILE" | tr -d '[:space:]')" = "$FIXTURE_BYTES" ] ||
    fail "wrong-hash fixture changed byte size"
: >"$FAKE_HF_LOG"
run_expect_failure "$TMP_ROOT/wrong-hash.stdout" "$TMP_ROOT/wrong-hash.stderr" \
    env PATH="$FAKE_BIN:$PATH" FAKE_HF_LOG="$FAKE_HF_LOG" \
    FAKE_HF_FAIL_IF_CALLED=1 DS4_GGUF_DIR="$WRONG_HASH_DIR" \
    "$TEST_SCRIPT" deepseek-v2
assert_contains "$TMP_ROOT/wrong-hash.stderr" 'Runtime artifact SHA-256 mismatch'
[ ! -s "$FAKE_HF_LOG" ] || fail "wrong-hash existing file triggered a download"

DOWNLOADED_BAD_DIR="$TMP_ROOT/downloaded-bad"
: >"$FAKE_HF_LOG"
run_expect_failure "$TMP_ROOT/downloaded-bad.stdout" "$TMP_ROOT/downloaded-bad.stderr" \
    env PATH="$FAKE_BIN:$PATH" FAKE_HF_LOG="$FAKE_HF_LOG" \
    FAKE_HF_PAYLOAD=qualified-runtime-fixturE \
    DS4_GGUF_DIR="$DOWNLOADED_BAD_DIR" "$TEST_SCRIPT" deepseek-v2
assert_contains "$TMP_ROOT/downloaded-bad.stderr" 'Runtime artifact SHA-256 mismatch'
assert_contains "$FAKE_HF_LOG" 'ds4-v0.2.0'

run_offline_success() {
    target=$1
    repo=$2
    model_file=$3
    out_dir="$TMP_ROOT/out-$target"
    stdout_file="$TMP_ROOT/$target.stdout"
    stderr_file="$TMP_ROOT/$target.stderr"
    : >"$FAKE_HF_LOG"

    PATH="$FAKE_BIN:$PATH" FAKE_HF_LOG="$FAKE_HF_LOG" \
        FAKE_HF_PAYLOAD=unverified-converter-input DS4_GGUF_DIR="$out_dir" \
        "$TEST_SCRIPT" "$target" >"$stdout_file" 2>"$stderr_file"

    expected_log="$TMP_ROOT/$target.expected"
    cat >"$expected_log" <<EOF
download
$repo
$model_file
--repo-type
model
--local-dir
$out_dir
EOF
    cmp "$expected_log" "$FAKE_HF_LOG" >/dev/null ||
        fail "$target changed its unverified converter-input mapping"
    assert_contains "$stdout_file" \
        'Offline converter input present; byte identity remains unverified.'
}

run_offline_success \
    offline-deepseek-flash-q2 \
    antirez/deepseek-v4-gguf \
    DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
run_offline_success \
    offline-deepseek-flash-q2-q4 \
    antirez/deepseek-v4-gguf \
    DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed.gguf
run_offline_success \
    offline-deepseek-flash-q4 \
    antirez/deepseek-v4-gguf \
    DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf
run_offline_success \
    offline-deepseek-pro-q2 \
    antirez/deepseek-v4-gguf \
    DeepSeek-V4-Pro-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct-imatrix.gguf

echo "download-model: PASS"
