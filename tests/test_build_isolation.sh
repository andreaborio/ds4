#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
ARCH=$(uname -m)
METAL_REL="build/metal-${ARCH}/bin"
CPU_REL="build/cpu-${ARCH}/bin"
PROGRAMS="ds4 ds4-server ds4-bench ds4-eval ds4-agent"
MAKE_BIN=${MAKE:-make}

fail() {
    echo "build-isolation: FAIL: $*" >&2
    exit 1
}

assert_metal_binary() {
    binary=$1
    otool -L "$binary" | grep -q '/Metal.framework/' || \
        fail "$binary is not linked with Metal.framework"
}

assert_cpu_binary() {
    binary=$1
    if otool -L "$binary" | grep -q '/Metal.framework/'; then
        fail "$binary unexpectedly links Metal.framework"
    fi
}

assert_build_info() {
    binary=$1
    backend=$2
    info=$("./$binary" --build-info)
    echo "$info" | grep -q "^backend: $backend$" || \
        fail "$binary reports the wrong compiled backend"
    echo "$info" | grep -q "^arch:    $ARCH$" || \
        fail "$binary reports the wrong compiled architecture"
    echo "$info" | grep -q '^git:     .' || \
        fail "$binary does not report build provenance"
}

cd "$ROOT"
"$MAKE_BIN" clean
"$MAKE_BIN" metal

root_state=""
for name in $PROGRAMS; do
    [ -L "$name" ] || fail "$name is not a published Metal symlink"
    [ "$(readlink "$name")" = "$METAL_REL/$name" ] || \
        fail "$name does not point at $METAL_REL/$name"
    [ -x "$METAL_REL/$name" ] || fail "missing $METAL_REL/$name"
    assert_metal_binary "$name"
    assert_build_info "$name" metal
    root_state="$root_state $name:$(stat -f '%i:%m' "$name")"
done

[ -f "build/metal-${ARCH}/obj/ds4.o" ] || fail "missing Metal core object"
[ -f "build/metal-${ARCH}/obj/ds4_metal.o" ] || fail "missing Metal backend object"

"$MAKE_BIN" cpu

[ -f "build/cpu-${ARCH}/obj/ds4.o" ] || fail "missing CPU core object"
[ ! -e "build/cpu-${ARCH}/obj/ds4_metal.o" ] || fail "CPU profile contains a Metal object"

after_cpu_state=""
for name in $PROGRAMS; do
    [ -x "$CPU_REL/$name" ] || fail "missing $CPU_REL/$name"
    assert_cpu_binary "$CPU_REL/$name"
    assert_build_info "$CPU_REL/$name" cpu
    assert_metal_binary "$name"
    after_cpu_state="$after_cpu_state $name:$(stat -f '%i:%m' "$name")"
done

[ "$root_state" = "$after_cpu_state" ] || \
    fail "make cpu modified one or more published Metal root links"

"$MAKE_BIN" metal

for name in $PROGRAMS; do
    [ "$(readlink "$name")" = "$METAL_REL/$name" ] || \
        fail "final $name does not point at the Metal profile"
    assert_metal_binary "$name"
done

echo "build-isolation: PASS (metal-${ARCH} -> cpu-${ARCH} -> metal-${ARCH})"
