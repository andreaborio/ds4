#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
ARCH=$(uname -m)
METAL_REL="build/metal-${ARCH}/bin"
CPU_REL="build/cpu-${ARCH}/bin"
PAIRS="hebrus:ds4 hebrus-server:ds4-server hebrus-bench:ds4-bench hebrus-eval:ds4-eval hebrus-agent:ds4-agent"
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
for pair in $PAIRS; do
    canonical=${pair%%:*}
    legacy=${pair#*:}
    for name in "$canonical" "$legacy"; do
        [ -L "$name" ] || fail "$name is not a published Metal symlink"
        [ "$(readlink "$name")" = "$METAL_REL/$canonical" ] || \
            fail "$name does not point at $METAL_REL/$canonical"
        assert_metal_binary "$name"
        assert_build_info "$name" metal
        root_state="$root_state $name:$(stat -f '%i:%m' "$name")"
    done
    [ -x "$METAL_REL/$canonical" ] || fail "missing $METAL_REL/$canonical"
    [ ! -L "$METAL_REL/$canonical" ] || \
        fail "$METAL_REL/$canonical is not the canonical binary"
    [ -L "$METAL_REL/$legacy" ] || fail "$METAL_REL/$legacy is not an alias"
    [ "$(readlink "$METAL_REL/$legacy")" = "$canonical" ] || \
        fail "$METAL_REL/$legacy does not point at $canonical"
done

python3 tests/test_capabilities.py --bin-dir "$METAL_REL" --backend metal
python3 tests/test_command_aliases.py --bin-dir "$METAL_REL" \
    --backend metal --layout profile
python3 tests/test_command_aliases.py --bin-dir . \
    --backend metal --layout published

[ -f "build/metal-${ARCH}/obj/ds4.o" ] || fail "missing Metal core object"
[ -f "build/metal-${ARCH}/obj/ds4_metal.o" ] || fail "missing Metal backend object"

"$MAKE_BIN" cpu

[ -f "build/cpu-${ARCH}/obj/ds4.o" ] || fail "missing CPU core object"
[ ! -e "build/cpu-${ARCH}/obj/ds4_metal.o" ] || fail "CPU profile contains a Metal object"

after_cpu_state=""
for pair in $PAIRS; do
    canonical=${pair%%:*}
    legacy=${pair#*:}
    [ -x "$CPU_REL/$canonical" ] || fail "missing $CPU_REL/$canonical"
    [ ! -L "$CPU_REL/$canonical" ] || \
        fail "$CPU_REL/$canonical is not the canonical binary"
    [ -L "$CPU_REL/$legacy" ] || fail "$CPU_REL/$legacy is not an alias"
    [ "$(readlink "$CPU_REL/$legacy")" = "$canonical" ] || \
        fail "$CPU_REL/$legacy does not point at $canonical"
    for name in "$canonical" "$legacy"; do
        assert_cpu_binary "$CPU_REL/$name"
        assert_build_info "$CPU_REL/$name" cpu
        assert_metal_binary "$name"
        after_cpu_state="$after_cpu_state $name:$(stat -f '%i:%m' "$name")"
    done
done

python3 tests/test_capabilities.py --bin-dir "$CPU_REL" --backend cpu
python3 tests/test_command_aliases.py --bin-dir "$CPU_REL" \
    --backend cpu --layout profile

[ "$root_state" = "$after_cpu_state" ] || \
    fail "make cpu modified one or more published Metal root links"

"$MAKE_BIN" metal

for pair in $PAIRS; do
    canonical=${pair%%:*}
    legacy=${pair#*:}
    for name in "$canonical" "$legacy"; do
        [ "$(readlink "$name")" = "$METAL_REL/$canonical" ] || \
            fail "final $name does not point at $METAL_REL/$canonical"
        assert_metal_binary "$name"
    done
done

echo "build-isolation: PASS (metal-${ARCH} -> cpu-${ARCH} -> metal-${ARCH})"
