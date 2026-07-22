#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
MAKE_BIN=${MAKE:-make}
BACKEND=${HEBRUS_INSTALL_BACKEND:?missing HEBRUS_INSTALL_BACKEND}
TMP_BASE=$(mktemp -d "${TMPDIR:-/tmp}/hebrus-install.XXXXXX")
STAGE="$TMP_BASE/stage root"
PREFIX_ONE=/portable-prefix
PREFIX_TWO=/unused-prefix
BINDIR_TWO=/custom/hebrus-bin
PROGRAMS="hebrus hebrus-server hebrus-bench hebrus-eval hebrus-agent ds4 ds4-server ds4-bench ds4-eval ds4-agent"
PAIRS="hebrus:ds4 hebrus-server:ds4-server hebrus-bench:ds4-bench hebrus-eval:ds4-eval hebrus-agent:ds4-agent"

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
"$MAKE_BIN" uninstall DESTDIR="$STAGE" PREFIX="$PREFIX_ONE"
assert_removed "$BIN_ONE"
[ "$(cat "$BIN_ONE/keep-me")" = keep ] || fail "uninstall removed an unrelated file"
"$MAKE_BIN" uninstall DESTDIR="$STAGE" PREFIX="$PREFIX_ONE"

"$MAKE_BIN" install DESTDIR="$STAGE" PREFIX="$PREFIX_TWO" BINDIR="$BINDIR_TWO"
BIN_TWO="$STAGE$BINDIR_TWO"
assert_layout "$BIN_TWO"
[ ! -e "$STAGE$PREFIX_TWO/bin/hebrus" ] || fail "install ignored BINDIR"
python3 tests/test_command_aliases.py --bin-dir "$BIN_TWO" \
    --backend "$BACKEND" --layout profile
"$MAKE_BIN" uninstall DESTDIR="$STAGE" PREFIX="$PREFIX_TWO" BINDIR="$BINDIR_TWO"
assert_removed "$BIN_TWO"

echo "install-layout: PASS ($BACKEND, DESTDIR/PREFIX/BINDIR, explicit uninstall)"
