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
BINDIR_THREE='/custom layout//hebrus bin//.'
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

assert_resource_dir() {
    metal_dir=$1
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

assert_resource_layout() {
    bin_dir=$1
    metal_dir=$(resource_root_for_bin "$bin_dir")/metal
    assert_resource_dir "$metal_dir"
}

assert_resource_files_removed_at() {
    metal_dir=$1
    for name in $METAL_RESOURCE_NAMES; do
        path="$metal_dir/$name"
        if [ -e "$path" ] || [ -L "$path" ]; then
            fail "uninstall left Metal source $path"
        fi
    done
}

assert_resource_files_removed() {
    bin_dir=$1
    assert_resource_files_removed_at "$(resource_root_for_bin "$bin_dir")/metal"
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

    # A partial installed set plus a complementary CWD file must not be
    # accepted as one library.  Every non-overridden source comes from one
    # complete root.
    first_resource=
    for name in $METAL_RESOURCE_NAMES; do
        first_resource=$name
        break
    done
    [ -n "$first_resource" ] || fail "Metal install has no resource names"
    installed_metal=$(resource_root_for_bin "$bin_dir")/metal
    mkdir -p "$clean_cwd/metal"
    mv "$installed_metal/$first_resource" "$clean_cwd/metal/$first_resource"
    if (
        for name in $METAL_SOURCE_ENV_VARS; do
            unset "$name"
        done
        cd "$clean_cwd"
        "$probe" --metal-source-discovery >/dev/null 2>&1
    ); then
        mv "$clean_cwd/metal/$first_resource" "$installed_metal/$first_resource"
        fail "Metal source discovery combined files from separate roots"
    fi
    mv "$clean_cwd/metal/$first_resource" "$installed_metal/$first_resource"
    rmdir "$clean_cwd/metal"
    rm -f "$probe"
}

assert_layout() {
    bin_dir=$1
    expected_entries=${2:-10}
    count=$(find "$bin_dir" -mindepth 1 -maxdepth 1 -print | wc -l | tr -d ' ')
    [ "$count" = "$expected_entries" ] || \
        fail "$bin_dir contains $count entries instead of $expected_entries"
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

"$MAKE_BIN" install DESTDIR="$STAGE" PREFIX="$PREFIX_TWO" BINDIR="$BINDIR_THREE"
BIN_THREE="$STAGE/custom layout/hebrus bin"
assert_layout "$BIN_THREE"
assert_resource_layout "$BIN_THREE"
run_installed_metal_probe "$BIN_THREE"
"$MAKE_BIN" uninstall DESTDIR="$STAGE" PREFIX="$PREFIX_TWO" BINDIR="$BINDIR_THREE"
assert_removed "$BIN_THREE"
assert_resource_files_removed "$BIN_THREE"

# Root and redundant-separator BINDIR forms are valid absolute layouts.  A
# DESTDIR root is not runnable in place with executable-relative resources,
# but its staged paths must still be correct and removable.
ROOT_STAGE="$TMP_BASE/root-stage"
mkdir -p "$ROOT_STAGE"
"$MAKE_BIN" install DESTDIR="$ROOT_STAGE" PREFIX="$PREFIX_TWO" BINDIR=/
root_entries=10
[ "$BACKEND" != metal ] || root_entries=11
assert_layout "$ROOT_STAGE" "$root_entries"
ROOT_METAL="$ROOT_STAGE/share/hebrus/v$METAL_RESOURCE_VERSION/metal"
assert_resource_dir "$ROOT_METAL"
"$MAKE_BIN" uninstall DESTDIR="$ROOT_STAGE" PREFIX="$PREFIX_TWO" BINDIR=/
assert_removed "$ROOT_STAGE"
assert_resource_files_removed_at "$ROOT_METAL"

DOT_ROOT_STAGE="$TMP_BASE/dot-root-stage"
mkdir -p "$DOT_ROOT_STAGE"
"$MAKE_BIN" install DESTDIR="$DOT_ROOT_STAGE" PREFIX="$PREFIX_TWO" BINDIR=/.
assert_layout "$DOT_ROOT_STAGE" "$root_entries"
DOT_ROOT_METAL="$DOT_ROOT_STAGE/share/hebrus/v$METAL_RESOURCE_VERSION/metal"
assert_resource_dir "$DOT_ROOT_METAL"
"$MAKE_BIN" uninstall DESTDIR="$DOT_ROOT_STAGE" PREFIX="$PREFIX_TWO" BINDIR=/.
assert_removed "$DOT_ROOT_STAGE"
assert_resource_files_removed_at "$DOT_ROOT_METAL"

TRAVERSAL_STAGE="$TMP_BASE/traversal/stage"
TRAVERSAL_ESCAPE="$TMP_BASE/traversal/escape"
mkdir -p "$TRAVERSAL_STAGE"
if "$MAKE_BIN" install DESTDIR="$TRAVERSAL_STAGE" PREFIX="$PREFIX_TWO" \
        BINDIR=/../escape >/dev/null 2>&1; then
    fail "install accepted a BINDIR containing '..'"
fi
[ ! -e "$TRAVERSAL_ESCAPE/hebrus" ] || \
    fail "install escaped DESTDIR through BINDIR"
if "$MAKE_BIN" uninstall DESTDIR="$TRAVERSAL_STAGE" PREFIX="$PREFIX_TWO" \
        BINDIR=/../escape >/dev/null 2>&1; then
    fail "uninstall accepted a BINDIR containing '..'"
fi

if [ "$BACKEND" = metal ]; then
    # Uninstall must refuse a swapped resource-directory symlink before it can
    # follow the link and remove same-named files from an unrelated target.
    VICTIM="$TMP_BASE/uninstall-victim"
    mkdir -p "$VICTIM" "$RESOURCE_TWO"
    first_resource=
    for name in $METAL_RESOURCE_NAMES; do
        first_resource=$name
        break
    done
    printf '%s\n' keep > "$VICTIM/$first_resource"
    ln -s "$VICTIM" "$RESOURCE_TWO/metal"
    if "$MAKE_BIN" uninstall DESTDIR="$STAGE" PREFIX="$PREFIX_TWO" \
            BINDIR="$BINDIR_TWO" >/dev/null 2>&1; then
        fail "uninstall accepted a symlinked Metal resource directory"
    fi
    [ "$(cat "$VICTIM/$first_resource")" = keep ] || \
        fail "uninstall followed the Metal resource symlink"
fi

echo "install-layout: PASS ($BACKEND, DESTDIR/PREFIX/BINDIR, resources, explicit uninstall)"
