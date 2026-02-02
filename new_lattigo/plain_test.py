#!/usr/bin/env python3
# plain_test.py
#
# Plaintext executor for your HEVM program, for structural validation.
# - Reads .cst (float64 vectors) and .hevm (u16 op stream)
# - Executes ops on plaintext slot vectors (numpy float64)
#
# Fix:
# - DO NOT skip (body_len - 40) bytes after ConfigBody.
#   In your file, ConfigBodyLength=80 is a metadata field, not necessarily
#   indicating extra bytes physically present. Your hevm_reader.go reads only
#   5x u64 config (40B) and succeeds, so we follow that.
#
# Usage:
#   python3 plain_test.py --cst ./src/_hecate_ResNet.cst --hevm ./src/ResNet.40._hecate_ResNet.hevm \
#       --slots 16384 --x0 0.1,0.2,0.3,0.4

import argparse
import struct
from pathlib import Path
import numpy as np


# ----------------------------- CST reader -----------------------------
# format:
#   int64 nvec
#   repeat nvec:
#       int64 veclen
#       float64[veclen]
def read_cst(path: Path):
    with path.open("rb") as f:
        nvec = struct.unpack("<q", f.read(8))[0]
        if nvec < 0:
            raise ValueError(f"bad cst nvec={nvec}")
        out = []
        for i in range(nvec):
            veclen = struct.unpack("<q", f.read(8))[0]
            if veclen < 0:
                raise ValueError(f"bad cst veclen[{i}]={veclen}")
            b = f.read(8 * veclen)
            if len(b) != 8 * veclen:
                raise EOFError(f"cst EOF at vec[{i}] want={8*veclen} got={len(b)}")
            out.append(np.frombuffer(b, dtype="<f8").copy())
        return out


# ----------------------------- HEVM reader -----------------------------
# header:
#   u32 magic (0x4845564D)
#   u32 header_size (24)
#   u64 arg_len
#   u64 res_len
#
# config (physically present in your file, per hevm_reader.go):
#   u64 body_len          (metadata; do NOT use to skip bytes)
#   u64 num_ops
#   u64 num_ctxt_buf
#   u64 num_ptxt_buf
#   u64 init_level
#
# then arrays:
#   argScale[arg_len] u64
#   argLevel[arg_len] u64
#   resScale[res_len] u64
#   resLevel[res_len] u64
#   resDst[res_len] u64
#
# then ops: num_ops * 8 bytes; each op is 4x u16 LE: opcode,dst,lhs,rhs
def read_hevm(path: Path):
    b = path.read_bytes()
    off = 0

    def need(n: int, what: str):
        if off + n > len(b):
            raise EOFError(f"EOF while reading {what}: need {n} bytes, have {len(b)-off}")

    def u32(what="u32"):
        nonlocal off
        need(4, what)
        v = struct.unpack_from("<I", b, off)[0]
        off += 4
        return v

    def u64(what="u64"):
        nonlocal off
        need(8, what)
        v = struct.unpack_from("<Q", b, off)[0]
        off += 8
        return v

    magic = u32("magic")
    header_size = u32("header_size")
    arg_len = u64("arg_len")
    res_len = u64("res_len")

    if magic != 0x4845564D:
        raise ValueError(f"bad magic=0x{magic:08x}")

    body_len = u64("body_len")
    num_ops = u64("num_ops")
    num_ct = u64("num_ctxt_buf")
    num_pt = u64("num_ptxt_buf")
    init_level = u64("init_level")

    # IMPORTANT:
    # Do NOT skip (body_len - 40). Your hevm_reader.go doesn't skip and succeeds.
    # body_len is treated as metadata only.

    def read_u64_arr(n, name):
        nonlocal off
        n = int(n)
        if n == 0:
            return np.zeros((0,), dtype=np.uint64)
        need(8 * n, name)
        arr = np.frombuffer(b[off:off + 8 * n], dtype="<u8").copy()
        off += 8 * n
        return arr

    argScale = read_u64_arr(arg_len, "argScale")
    argLevel = read_u64_arr(arg_len, "argLevel")
    resScale = read_u64_arr(res_len, "resScale")
    resLevel = read_u64_arr(res_len, "resLevel")
    resDst = read_u64_arr(res_len, "resDst")

    ops_bytes = int(num_ops) * 8
    need(ops_bytes, "ops")
    ops = np.frombuffer(b[off:off + ops_bytes], dtype="<u2").reshape((-1, 4)).copy()

    return {
        "magic": magic,
        "header_size": header_size,
        "arg_len": int(arg_len),
        "res_len": int(res_len),
        "body_len": int(body_len),
        "num_ops": int(num_ops),
        "num_ct": int(num_ct),
        "num_pt": int(num_pt),
        "init_level": int(init_level),
        "argScale": argScale,
        "argLevel": argLevel,
        "resScale": resScale,
        "resLevel": resLevel,
        "resDst": resDst,
        "ops": ops,
    }


# ----------------------------- helpers -----------------------------
def parse_x0(s: str):
    if s is None or s == "":
        return np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return np.array([float(p) for p in parts], dtype=np.float64)


def sign_extend_u16_to_i16(u: int) -> int:
    # Equivalent to (int16_t)u in C, without numpy overflow warnings.
    return u if u < 0x8000 else u - 0x10000


def roll(vec: np.ndarray, k: int):
    if k == 0:
        return vec
    return np.roll(vec, k)


def broadcast_to_slots(v: np.ndarray, slots: int):
    v = np.asarray(v, dtype=np.float64)
    if v.size == slots:
        return v.astype(np.float64, copy=False)
    if v.size == 0:
        return np.zeros(slots, dtype=np.float64)
    out = np.empty(slots, dtype=np.float64)
    out[:] = v[np.arange(slots) % v.size]
    return out


# ----------------------------- opcode map -----------------------------
OP_ENCODE = 0
OP_ROTATEC = 1
OP_NEGATEC = 2
OP_RESCALEC = 3
OP_MODSWC = 4
OP_UPSCALEC = 5
OP_ADDCC = 6
OP_ADDCP = 7
OP_MULCC = 8
OP_MULCP = 9
OP_BOOT = 10
OP_NOP = 0xFFFF


def run_plain(hevm, consts, slots: int, x0: np.ndarray, verbose: bool):
    ctN = max(int(hevm["num_ct"]), hevm["arg_len"] + hevm["res_len"], 1)
    ptN = max(int(hevm["num_pt"]), 1)

    C = [np.zeros(slots, dtype=np.float64) for _ in range(ctN)]
    P = [np.zeros(slots, dtype=np.float64) for _ in range(ptN)]

    ops = hevm["ops"]

    # preprocess: fill plaintext buffers from OP_ENCODE
    for (opcode, dst, lhs, rhs) in ops:
        if int(opcode) != OP_ENCODE:
            continue
        d = int(dst)
        li = int(lhs)
        if not (0 <= d < len(P)):
            continue

        if li == 0xFFFF:
            vec = np.array([1.0], dtype=np.float64)
        elif 0 <= li < len(consts):
            vec = consts[li]
        else:
            vec = np.array([0.0], dtype=np.float64)

        P[d] = broadcast_to_slots(vec, slots)

    # arg0
    if hevm["arg_len"] > 0:
        C[0] = broadcast_to_slots(x0, slots)

    # run program
    for idx, (opcode_u16, dst_u16, lhs_u16, rhs_u16) in enumerate(ops):
        opcode = int(opcode_u16)
        if opcode == OP_ENCODE or opcode == OP_NOP:
            continue

        dst = int(dst_u16)
        lhs = int(lhs_u16)
        rhs = int(rhs_u16)

        if opcode == OP_ROTATEC:
            k = sign_extend_u16_to_i16(rhs)
            if 0 <= dst < len(C) and 0 <= lhs < len(C):
                C[dst] = roll(C[lhs], k)

        elif opcode == OP_NEGATEC:
            if 0 <= dst < len(C) and 0 <= lhs < len(C):
                C[dst] = -C[lhs]

        elif opcode in (OP_RESCALEC, OP_MODSWC, OP_UPSCALEC, OP_BOOT):
            # ignored in plaintext sim
            if 0 <= dst < len(C) and 0 <= lhs < len(C):
                C[dst] = C[lhs].copy()

        elif opcode == OP_ADDCC:
            if 0 <= dst < len(C) and 0 <= lhs < len(C) and 0 <= rhs < len(C):
                C[dst] = C[lhs] + C[rhs]

        elif opcode == OP_ADDCP:
            if 0 <= dst < len(C) and 0 <= lhs < len(C) and 0 <= rhs < len(P):
                C[dst] = C[lhs] + P[rhs]

        elif opcode == OP_MULCC:
            if 0 <= dst < len(C) and 0 <= lhs < len(C) and 0 <= rhs < len(C):
                C[dst] = C[lhs] * C[rhs]

        elif opcode == OP_MULCP:
            if 0 <= dst < len(C) and 0 <= lhs < len(C) and 0 <= rhs < len(P):
                C[dst] = C[lhs] * P[rhs]

        if verbose and (idx % 5000 == 0):
            print(f"[PLAIN] step {idx}/{len(ops)} opcode={opcode}")

    # collect results using resDst mapping
    res = []
    for i in range(hevm["res_len"]):
        dst_idx = int(hevm["resDst"][i]) if i < len(hevm["resDst"]) else i
        if 0 <= dst_idx < len(C):
            res.append(C[dst_idx])
        else:
            res.append(np.zeros(slots, dtype=np.float64))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cst", required=True)
    ap.add_argument("--hevm", required=True)
    ap.add_argument("--slots", type=int, default=16384)
    ap.add_argument("--x0", default="0.1,0.2,0.3,0.4")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cst_path = Path(args.cst)
    hevm_path = Path(args.hevm)
    if not cst_path.is_file():
        raise FileNotFoundError(cst_path)
    if not hevm_path.is_file():
        raise FileNotFoundError(hevm_path)

    consts = read_cst(cst_path)
    hevm = read_hevm(hevm_path)

    print(f"[CST] vectors={len(consts)}")
    if consts:
        print(f"[CST] vec0_len={len(consts[0])}")
    print(f"[HEVM] magic=0x{hevm['magic']:08X} header_size={hevm['header_size']} arg={hevm['arg_len']} res={hevm['res_len']}")
    print(f"[HEVM] body_len={hevm['body_len']} ops={hevm['num_ops']} ctxt_buf={hevm['num_ct']} ptxt_buf={hevm['num_pt']} init_level={hevm['init_level']}")
    if hevm["arg_len"] > 0:
        print(f"[HEVM] arg0: scale={int(hevm['argScale'][0])} level={int(hevm['argLevel'][0])}")
    if hevm["res_len"] > 0:
        print(f"[HEVM] res0: scale={int(hevm['resScale'][0])} level={int(hevm['resLevel'][0])} dst={int(hevm['resDst'][0])}")

    x0 = parse_x0(args.x0)
    res = run_plain(hevm, consts, args.slots, x0, args.verbose)

    if res:
        print("[PLAIN] res0 head:", res[0][:16])
    else:
        print("[PLAIN] no results")


if __name__ == "__main__":
    main()
