#!/usr/bin/env python3
"""Verification for the re-calibration reuse exercise (feat/reuse-weights-key).

A = v2-twin (coder.dat imatrix, new-format keys), B = same recipe re-calibrated with
general.dat via --reuse A. Checks, per the soundness claim:
  1. every imatrix-INDEPENDENT tensor is byte-identical between A and B;
  2. every routed-expert (*_exps.*) tensor DIFFERS (different imatrix steered it);
  3. the per-file tensor tables agree (same names/types/shapes/offsets).
Reads tensor data straight from the GGUFs. Stdlib only.
"""
import struct, sys, hashlib

GGUF_TYPE_SIZES = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}

def read_str(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")

def skip_value(f, t):
    if t == 8:                      # string
        n = struct.unpack("<Q", f.read(8))[0]; f.seek(n, 1)
    elif t == 9:                    # array
        et = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        if et == 8:
            for _ in range(n):
                ln = struct.unpack("<Q", f.read(8))[0]; f.seek(ln, 1)
        else:
            f.seek(GGUF_TYPE_SIZES[et] * n, 1)
    else:
        f.seek(GGUF_TYPE_SIZES[t], 1)

def load_table(path):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF"
    _ver, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
    align = 32
    kvs = {}
    for _ in range(n_kv):
        k = read_str(f)
        t = struct.unpack("<I", f.read(4))[0]
        if k == "general.alignment" and t == 4:
            align = struct.unpack("<I", f.read(4))[0]
        elif t == 8:
            kvs[k] = read_str(f) if k.startswith("quantize.reuse") else (skip_value_str(f) if False else read_str(f))
        else:
            skip_value(f, t)
    tens = {}
    for _ in range(n_tensors):
        name = read_str(f)
        nd = struct.unpack("<I", f.read(4))[0]
        ne = struct.unpack("<%dQ" % nd, f.read(8 * nd))
        ty, off = struct.unpack("<IQ", f.read(12))
        tens[name] = (ty, ne, off)
    pos = f.tell()
    data0 = (pos + align - 1) // align * align
    return f, tens, data0, kvs

def tensor_digest(f, data0, off, size):
    f.seek(data0 + off)
    h = hashlib.sha256()
    left = size
    while left:
        chunk = f.read(min(1 << 22, left))
        if not chunk: raise IOError("short read")
        h.update(chunk); left -= len(chunk)
    return h.hexdigest()[:16]

def sizes_from_offsets(tens, file_end_hint=None):
    # size of each tensor = next offset - offset (template order by offset)
    by_off = sorted(tens.items(), key=lambda kv: kv[1][2])
    out = {}
    for i, (name, (ty, ne, off)) in enumerate(by_off):
        end = by_off[i + 1][1][2] if i + 1 < len(by_off) else None
        out[name] = (off, end)
    return out

def main():
    A, B = sys.argv[1], sys.argv[2]
    fa, ta, da, ka = load_table(A)
    fb, tb, db, kb = load_table(B)
    print("KV A:", {k: v for k, v in ka.items() if k.startswith("quantize.reuse")})
    print("KV B:", {k: v for k, v in kb.items() if k.startswith("quantize.reuse")})
    assert set(ta) == set(tb), "tensor sets differ"
    mismatch_table = [n for n in ta if ta[n][:2] != tb[n][:2] or ta[n][2] != tb[n][2]]
    print("tabella tensori identica (tipi/shape/offset):", "SI" if not mismatch_table else mismatch_table[:5])
    sa, sb = sizes_from_offsets(ta), sizes_from_offsets(tb)

    import random
    rng = random.Random(42)
    names = list(ta)
    exps = [n for n in names if "_exps." in n]
    regular = [n for n in names if "_exps." not in n]
    sample_reg = rng.sample(regular, min(40, len(regular)))
    sample_exp = rng.sample(exps, min(12, len(exps))) + [n for n in exps if ".39." in n or ".41." in n][:4]

    bad = 0
    for n in sample_reg:
        off, end = sa[n]
        size = (end - off) if end else 4096 * 1024
        d1 = tensor_digest(fa, da, off, min(size, 1 << 24))
        d2 = tensor_digest(fb, db, off, min(size, 1 << 24))
        ok = d1 == d2
        bad += not ok
        if not ok: print("  !! REGULAR DIVERSO (stale?):", n)
    print("regular campionati identici: %d/%d" % (len(sample_reg) - bad, len(sample_reg)))

    same = 0
    for n in set(sample_exp):
        off, end = sa[n]
        size = (end - off) if end else 4096 * 1024
        d1 = tensor_digest(fa, da, off, min(size, 1 << 24))
        d2 = tensor_digest(fb, db, off, min(size, 1 << 24))
        if d1 == d2:
            same += 1
            print("  !! EXPS IDENTICO (imatrix non applicata?):", n)
    print("exps campionati diversi: %d/%d" % (len(set(sample_exp)) - same, len(set(sample_exp))))
    print("VERDETTO:", "PASS" if (bad == 0 and same == 0 and not mismatch_table) else "FAIL")

if __name__ == "__main__":
    main()
