#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
MAKE_BIN=${MAKE:-make}
BACKEND=${HEBRUS_INSTALL_BACKEND:?missing HEBRUS_INSTALL_BACKEND}
METAL_PROBE=${HEBRUS_INSTALL_METAL_PROBE-}
METAL_RESOURCE_NAMES=${HEBRUS_INSTALL_METAL_RESOURCE_NAMES-}
METAL_RESOURCE_VERSION=${HEBRUS_INSTALL_METAL_RESOURCE_VERSION:?missing HEBRUS_INSTALL_METAL_RESOURCE_VERSION}
TMP_BASE=$(mktemp -d "${TMPDIR:-/tmp}/hebrus-install.XXXXXX")
STAGE="$TMP_BASE/stage root"
PREFIX_ONE=/portable-prefix
PREFIX_TWO=/unused-prefix
BINDIR_TWO=/custom/hebrus-bin
PROGRAMS="hebrus hebrus-server hebrus-bench hebrus-eval hebrus-agent ds4 ds4-server ds4-bench ds4-eval ds4-agent"
PAIRS="hebrus:ds4 hebrus-server:ds4-server hebrus-bench:ds4-bench hebrus-eval:ds4-eval hebrus-agent:ds4-agent"
METAL_SOURCE_ENV_VARS="
DS4_METAL_SOURCE_DIR
DS4_METAL_FLASH_ATTN_SOURCE
DS4_METAL_DENSE_SOURCE
DS4_METAL_QWEN35_IQ_TABLES_SOURCE
DS4_METAL_MOE_SOURCE
DS4_METAL_DSV4_HC_SOURCE
DS4_METAL_UNARY_SOURCE
DS4_METAL_DSV4_KV_SOURCE
DS4_METAL_DSV4_ROPE_SOURCE
DS4_METAL_DSV4_MISC_SOURCE
DS4_METAL_ARGSORT_SOURCE
DS4_METAL_CPY_SOURCE
DS4_METAL_CONCAT_SOURCE
DS4_METAL_GET_ROWS_SOURCE
DS4_METAL_SUM_ROWS_SOURCE
DS4_METAL_SOFTMAX_SOURCE
DS4_METAL_REPEAT_SOURCE
DS4_METAL_GLU_SOURCE
DS4_METAL_NORM_SOURCE
DS4_METAL_BIN_SOURCE
DS4_METAL_SET_ROWS_SOURCE
DS4_METAL_QWEN35_SOURCE
"

cleanup() {
    rm -rf "$TMP_BASE"
}
trap cleanup 0 1 2 3 15

fail() {
    echo "install-layout: FAIL: $*" >&2
    exit 1
}

assert_removed() {
    bin_dir=$1
    for name in $PROGRAMS; do
        path="$bin_dir/$name"
        if [ -e "$path" ] || [ -L "$path" ]; then
            fail "uninstall left $path"
        fi
    done
}

resource_root_for_bin() {
    bin_dir=$1
    bin_parent=$(dirname -- "$bin_dir")
    printf '%s/share/hebrus/v%s\n' "$bin_parent" "$METAL_RESOURCE_VERSION"
}

assert_resource_layout() {
    bin_dir=$1
    resource_root=$(resource_root_for_bin "$bin_dir")
    metal_dir="$resource_root/metal"

    if [ "$BACKEND" != metal ]; then
        [ ! -e "$metal_dir" ] && [ ! -L "$metal_dir" ] || \
            fail "CPU install unexpectedly created $metal_dir"
        return
    fi

    [ -d "$metal_dir" ] && [ ! -L "$metal_dir" ] || \
        fail "missing regular Metal resource directory $metal_dir"
    expected=0
    for name in $METAL_RESOURCE_NAMES; do
        expected=$((expected + 1))
        path="$metal_dir/$name"
        [ -f "$path" ] && [ ! -L "$path" ] || \
            fail "missing regular Metal source $path"
        [ -r "$path" ] || fail "Metal source is not readable: $path"
        [ ! -x "$path" ] || fail "Metal source is unexpectedly executable: $path"
    done
    count=$(find "$metal_dir" -mindepth 1 -maxdepth 1 -type f -print | wc -l | tr -d ' ')
    [ "$count" = "$expected" ] || \
        fail "$metal_dir contains $count files instead of $expected"
}

assert_resource_files_removed() {
    bin_dir=$1
    metal_dir=$(resource_root_for_bin "$bin_dir")/metal
    for name in $METAL_RESOURCE_NAMES; do
        path="$metal_dir/$name"
        if [ -e "$path" ] || [ -L "$path" ]; then
            fail "uninstall left Metal source $path"
        fi
    done
}

run_installed_metal_probe() {
    bin_dir=$1
    [ "$BACKEND" = metal ] || return
    [ -n "$METAL_PROBE" ] || fail "Metal install has no source-discovery probe"
    [ -x "$METAL_PROBE" ] || fail "Metal source-discovery probe is not executable: $METAL_PROBE"

    clean_cwd="$TMP_BASE/clean cwd"
    probe="$bin_dir/.hebrus-install-metal-probe"
    mkdir -p "$clean_cwd"
    cp "$METAL_PROBE" "$probe"
    chmod 0755 "$probe"
    (
        for name in $METAL_SOURCE_ENV_VARS; do
            unset "$name"
        done
        cd "$clean_cwd"
        "$probe" --metal-source-discovery
    ) || fail "installed Metal sources were not discoverable from a clean working directory"
    rm -f "$probe"
}

assert_layout() {
    bin_dir=$1
    count=$(find "$bin_dir" -mindepth 1 -maxdepth 1 -print | wc -l | tr -d ' ')
    [ "$count" = 10 ] || fail "$bin_dir contains $count entries instead of 10"
    for pair in $PAIRS; do
        canonical=${pair%%:*}
        legacy=${pair#*:}
        [ -x "$bin_dir/$canonical" ] || fail "missing executable $canonical"
        [ ! -L "$bin_dir/$canonical" ] || fail "$canonical is a symlink"
        [ -L "$bin_dir/$legacy" ] || fail "$legacy is not a symlink"
        [ "$(readlink "$bin_dir/$legacy")" = "$canonical" ] || \
            fail "$legacy is not a relative alias to $canonical"
    done
}

mkdir -p "$STAGE"
cd "$ROOT"

"$MAKE_BIN" install DESTDIR="$STAGE" PREFIX="$PREFIX_ONE"
BIN_ONE="$STAGE$PREFIX_ONE/bin"
assert_layout "$BIN_ONE"
assert_resource_layout "$BIN_ONE"
[ ! -e "$STAGE/usr/local/bin/hebrus" ] || fail "install ignored PREFIX"
python3 tests/test_capabilities.py --bin-dir "$BIN_ONE" --backend "$BACKEND"
python3 tests/test_command_aliases.py --bin-dir "$BIN_ONE" \
    --backend "$BACKEND" --layout profile

for name in hebrus hebrus-server hebrus-bench hebrus-eval hebrus-agent; do
    capabilities=$("$BIN_ONE/$name" --capabilities=json)
    case "$capabilities" in
        *"$ROOT"*|*"$STAGE"*) fail "$name exposes a checkout or install path" ;;
    esac
done

printf '%s\n' keep > "$BIN_ONE/keep-me"
if [ "$BACKEND" = metal ]; then
    RESOURCE_ONE=$(resource_root_for_bin "$BIN_ONE")
    printf '%s\n' keep > "$RESOURCE_ONE/metal/keep-resource"
fi
"$MAKE_BIN" uninstall DESTDIR="$STAGE" PREFIX="$PREFIX_ONE"
assert_removed "$BIN_ONE"
assert_resource_files_removed "$BIN_ONE"
[ "$(cat "$BIN_ONE/keep-me")" = keep ] || fail "uninstall removed an unrelated file"
if [ "$BACKEND" = metal ]; then
    [ "$(cat "$RESOURCE_ONE/metal/keep-resource")" = keep ] || \
        fail "uninstall removed an unrelated Metal resource file"
fi
"$MAKE_BIN" uninstall DESTDIR="$STAGE" PREFIX="$PREFIX_ONE"

"$MAKE_BIN" install DESTDIR="$STAGE" PREFIX="$PREFIX_TWO" BINDIR="$BINDIR_TWO"
BIN_TWO="$STAGE$BINDIR_TWO"
assert_layout "$BIN_TWO"
assert_resource_layout "$BIN_TWO"
[ ! -e "$STAGE$PREFIX_TWO/bin/hebrus" ] || fail "install ignored BINDIR"
python3 tests/test_command_aliases.py --bin-dir "$BIN_TWO" \
    --backend "$BACKEND" --layout profile
run_installed_metal_probe "$BIN_TWO"
assert_layout "$BIN_TWO"
"$MAKE_BIN" uninstall DESTDIR="$STAGE" PREFIX="$PREFIX_TWO" BINDIR="$BINDIR_TWO"
assert_removed "$BIN_TWO"
assert_resource_files_removed "$BIN_TWO"
RESOURCE_TWO=$(resource_root_for_bin "$BIN_TWO")
[ ! -d "$RESOURCE_TWO/metal" ] || fail "uninstall left empty Metal resource directory"
[ ! -d "$RESOURCE_TWO" ] || fail "uninstall left empty versioned resource root"

echo "install-layout: PASS ($BACKEND, DESTDIR/PREFIX/BINDIR, resources, explicit uninstall)"
