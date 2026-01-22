# test_hevm_boot.py (standalone ops/boot) — no hardcoded MAX_LEVEL; query from lib
import ctypes
import os
import sys
import time
import math

LIB = os.path.join(os.path.dirname(__file__), "libLATTIGO_HEVM.so")
if not os.path.exists(LIB):
    print("missing:", LIB)
    print("run: ./build_so.sh")
    sys.exit(1)

lib = ctypes.CDLL(LIB)

# VM lifecycle
lib.initFullVM.restype = ctypes.c_size_t
lib.initFullVM.argtypes = [ctypes.c_char_p, ctypes.c_bool]
lib.freeVM.restype = None
lib.freeVM.argtypes = [ctypes.c_size_t]

# Params info
lib.Slots.restype = ctypes.c_int
lib.Slots.argtypes = [ctypes.c_size_t]
lib.LogN.restype = ctypes.c_int
lib.LogN.argtypes = [ctypes.c_size_t]
lib.MaxLevel.restype = ctypes.c_int
lib.MaxLevel.argtypes = [ctypes.c_size_t]
lib.LogDefaultScale.restype = ctypes.c_int
lib.LogDefaultScale.argtypes = [ctypes.c_size_t]

# Cipher meta
lib.CTLevel.restype = ctypes.c_int
lib.CTLevel.argtypes = [ctypes.c_size_t, ctypes.c_int]
lib.CTLog2Scale.restype = ctypes.c_int
lib.CTLog2Scale.argtypes = [ctypes.c_size_t, ctypes.c_int]

# Boot
lib.BootEnable.restype = ctypes.c_int
lib.BootEnable.argtypes = [ctypes.c_size_t]
lib.BootstrapTo.restype = ctypes.c_int
lib.BootstrapTo.argtypes = [ctypes.c_size_t, ctypes.c_int, ctypes.c_int]

# Encrypt/Decrypt
lib.EncryptTo.restype = None
lib.EncryptTo.argtypes = [
    ctypes.c_size_t, ctypes.c_int,
    ctypes.POINTER(ctypes.c_double), ctypes.c_int,
    ctypes.c_int, ctypes.c_int
]
lib.DecryptFrom.restype = None
lib.DecryptFrom.argtypes = [
    ctypes.c_size_t, ctypes.c_int,
    ctypes.POINTER(ctypes.c_double), ctypes.c_int
]

# Standalone ops
lib.OpAddCC.restype = None
lib.OpAddCC.argtypes = [ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.OpMulCC.restype = None
lib.OpMulCC.argtypes = [ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.OpRescale.restype = None
lib.OpRescale.argtypes = [ctypes.c_size_t, ctypes.c_int, ctypes.c_int]
lib.OpRotate.restype = None
lib.OpRotate.argtypes = [ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int]


def arr_d(xs):
    a = (ctypes.c_double * len(xs))()
    for i, x in enumerate(xs):
        a[i] = float(x)
    return a

def decrypt(vm, idx, n=8):
    out = (ctypes.c_double * n)()
    lib.DecryptFrom(vm, idx, out, n)
    return [out[i] for i in range(n)]

def ct_meta(vm, idx):
    lvl = lib.CTLevel(vm, idx)
    ls = lib.CTLog2Scale(vm, idx)
    return lvl, ls

def l2_err(a, b):
    s = 0.0
    for x, y in zip(a, b):
        d = x - y
        s += d * d
    return math.sqrt(s / max(1, len(a)))

def assert_close(got, exp, eps, name):
    for i, (a, b) in enumerate(zip(got, exp)):
        if abs(a - b) > eps:
            raise AssertionError(f"{name} idx={i} got={a} exp={b} diff={a-b} eps={eps}")

def rep_to_n(x, n):
    return [x[i % len(x)] for i in range(n)]


def main():
    vm = lib.initFullVM(b"", False)
    if not vm:
        raise RuntimeError("initFullVM failed")

    logN = lib.LogN(vm)
    slots = lib.Slots(vm)
    maxLevel = lib.MaxLevel(vm)
    logS = lib.LogDefaultScale(vm)
    print(f"[LATTIGO_HEVM] PARAM=N16QP1767H32768H32 logN={logN} slots={slots} maxLevel={maxLevel} logS={logS}")

    # STEP0 encrypt
    x = [0.5, -1.25, 2.0, 3.5]
    x_rep8 = rep_to_n(x, 8)
    lib.EncryptTo(vm, 0, arr_d(x), len(x), maxLevel, logS)
    d0 = decrypt(vm, 0, 8)
    print(f"[STEP0] ct0 meta: level={ct_meta(vm,0)[0]} log2(scale)={ct_meta(vm,0)[1]}")
    print(f"[STEP0] l2_err(dec8, x_rep8) = {l2_err(d0, x_rep8):.3e}")
    assert_close(d0[:4], x, 2e-2, "decrypt(x)")

    # STEP1 boot enable + bootstrap
    t0 = time.time()
    ok = lib.BootEnable(vm)
    t1 = time.time()
    if ok != 1:
        lib.freeVM(vm)
        raise RuntimeError("BootEnable failed")
    print(f"[BOOT] enable ok=1 time={t1-t0:.3f}s")

    t0 = time.time()
    ok = lib.BootstrapTo(vm, 10, 0)
    t1 = time.time()
    if ok != 1:
        lib.freeVM(vm)
        raise RuntimeError("BootstrapTo(dst=10, src=0) failed")
    d10 = decrypt(vm, 10, 8)
    print(f"[STEP1] ct10 meta: level={ct_meta(vm,10)[0]} log2(scale)={ct_meta(vm,10)[1]}")
    print(f"[STEP1] l2_err(dec8, x_rep8) = {l2_err(d10, x_rep8):.3e}")

    # STEP2 consume levels via mul-by-enc(1)
    lib.EncryptTo(vm, 1, arr_d([1.0]), 1, maxLevel, logS)

    cur = 10
    tmp_mul = 11
    tmp_rs = 12
    step = 0
    while True:
        lvl = lib.CTLevel(vm, cur)
        if lvl <= 0:
            break
        step += 1
        lib.OpMulCC(vm, tmp_mul, cur, 1)
        lib.OpRescale(vm, tmp_rs, tmp_mul)
        d_rs = decrypt(vm, tmp_rs, 8)
        err = l2_err(d_rs, x_rep8)
        print(f"  [STEP2.{step}] cur->mul level={ct_meta(vm,tmp_mul)[0]} ->rs level={ct_meta(vm,tmp_rs)[0]} err={err:.3e}")
        cur = tmp_rs

    print(f"[STEP2] reached level={lib.CTLevel(vm, cur)} (should be 0)")

    # STEP3 bootstrap again
    t0 = time.time()
    ok = lib.BootstrapTo(vm, 13, cur)
    t1 = time.time()
    if ok != 1:
        lib.freeVM(vm)
        raise RuntimeError("BootstrapTo(dst=13, src=cur) failed")
    d13 = decrypt(vm, 13, 8)
    print(f"[STEP3] Bootstrap ok=1 time={t1-t0:.3f}s meta={ct_meta(vm,13)} err={l2_err(d13, x_rep8):.3e}")
    assert_close(d13, x_rep8, 3e-1, "boot(x) after level-consume")

    # END HERE (no STEP4)
    lib.freeVM(vm)
    print("HEVM BOOT TEST PASSED (STOP AT STEP3)")


if __name__ == "__main__":
    main()
