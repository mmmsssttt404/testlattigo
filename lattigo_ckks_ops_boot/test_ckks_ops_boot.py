# test_ckks_ops_boot.py
import ctypes
import os
import sys
import time
import math

LIB = os.path.join(os.path.dirname(__file__), "libLATTIGO_CKKS_OPS.so")
if not os.path.exists(LIB):
    print("missing:", LIB)
    print("run: ./build_so.sh")
    sys.exit(1)

lib = ctypes.CDLL(LIB)

# ---------------- basic VM API ----------------
lib.CKKS_CreateVM.restype = ctypes.c_void_p
lib.CKKS_CreateVM.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]

lib.CKKS_FreeVM.restype = None
lib.CKKS_FreeVM.argtypes = [ctypes.c_void_p]

lib.CKKS_Slots.restype = ctypes.c_int
lib.CKKS_Slots.argtypes = [ctypes.c_void_p]

lib.CKKS_LogN.restype = ctypes.c_int
lib.CKKS_LogN.argtypes = [ctypes.c_void_p]

lib.CKKS_MaxLevel.restype = ctypes.c_int
lib.CKKS_MaxLevel.argtypes = [ctypes.c_void_p]

lib.CKKS_LogDefaultScale.restype = ctypes.c_int
lib.CKKS_LogDefaultScale.argtypes = [ctypes.c_void_p]

# ---------------- ciphertext metadata API (NEW) ----------------
# You must export these in Go (see ckks_ops.go below)
lib.CKKS_CTLevel.restype = ctypes.c_int
lib.CKKS_CTLevel.argtypes = [ctypes.c_void_p, ctypes.c_int]

lib.CKKS_CTLog2Scale.restype = ctypes.c_int
lib.CKKS_CTLog2Scale.argtypes = [ctypes.c_void_p, ctypes.c_int]

# ---------------- data IO ----------------
lib.CKKS_EncryptTo.restype = None
lib.CKKS_EncryptTo.argtypes = [
    ctypes.c_void_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_double), ctypes.c_int,
    ctypes.c_int, ctypes.c_int
]

lib.CKKS_DecryptFrom.restype = None
lib.CKKS_DecryptFrom.argtypes = [
    ctypes.c_void_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_double), ctypes.c_int
]

# ---------------- ops ----------------
lib.CKKS_OpMulCC.restype = None
lib.CKKS_OpMulCC.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]

lib.CKKS_OpRescale.restype = None
lib.CKKS_OpRescale.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

# ---------------- boot ----------------
lib.CKKS_BootEnable.restype = ctypes.c_int
lib.CKKS_BootEnable.argtypes = [ctypes.c_void_p]

lib.CKKS_BootstrapTo.restype = ctypes.c_int
lib.CKKS_BootstrapTo.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]


def arr_d(xs):
    a = (ctypes.c_double * len(xs))()
    for i, x in enumerate(xs):
        a[i] = float(x)
    return a


def decrypt(vm, idx, n=8):
    out = (ctypes.c_double * n)()
    lib.CKKS_DecryptFrom(vm, idx, out, n)
    return [out[i] for i in range(n)]


def ct_meta(vm, idx):
    lvl = lib.CKKS_CTLevel(vm, idx)
    lgs = lib.CKKS_CTLog2Scale(vm, idx)
    return lvl, lgs


def l2_err(got, exp):
    s = 0.0
    for a, b in zip(got, exp):
        d = a - b
        s += d * d
    return math.sqrt(s / max(1, len(got)))


def assert_close(got, exp, eps=3e-1, name=""):
    for i, (a, b) in enumerate(zip(got, exp)):
        if abs(a - b) > eps:
            raise AssertionError(f"{name} idx={i} got={a} exp={b} diff={a-b} eps={eps}")


def main():
    # CreateVM params will be overridden by BootEnable (to your default set).
    vm = lib.CKKS_CreateVM(15, 18, 40, 32)
    if not vm:
        raise RuntimeError("CreateVM failed")

    t0 = time.time()
    ok = lib.CKKS_BootEnable(vm)
    t1 = time.time()
    if ok != 1:
        lib.CKKS_FreeVM(vm)
        raise RuntimeError("CKKS_BootEnable failed")
    print(f"[BOOT] enable ok=1 time={t1-t0:.3f}s")

    slots = lib.CKKS_Slots(vm)
    logN = lib.CKKS_LogN(vm)
    maxLevel = lib.CKKS_MaxLevel(vm)
    logS = lib.CKKS_LogDefaultScale(vm)
    print(f"[CKKS] logN={logN} slots={slots} maxLevel={maxLevel} logS={logS}")

    # x replicated across slots by EncryptTo
    x = [0.5, -1.25, 2.0, 3.5]
    x_rep8 = [x[i % len(x)] for i in range(8)]
    x2_rep8 = [(x[i % len(x)] * x[i % len(x)]) for i in range(8)]

    # STEP0: encrypt ct0 at top level
    print(f"[STEP0] EncryptTo(dst=0, level={maxLevel}, logS={logS})")
    lib.CKKS_EncryptTo(vm, 0, arr_d(x), len(x), maxLevel, logS)
    lvl0, sc0 = ct_meta(vm, 0)
    d0 = decrypt(vm, 0, 8)
    print(f"[STEP0] ct0 meta: level={lvl0} log2(scale)={sc0}")
    print("[STEP0] dec8(ct0) =", d0)
    print(f"[STEP0] l2_err(dec8, x_rep8) = {l2_err(d0, x_rep8):.3e}")

    # STEP1: boot ct0 -> ct10
    t0 = time.time()
    ok = lib.CKKS_BootstrapTo(vm, 10, 0)
    t1 = time.time()
    if ok != 1:
        lib.CKKS_FreeVM(vm)
        raise RuntimeError("CKKS_BootstrapTo(dst=10, src=0) failed")
    lvl10, sc10 = ct_meta(vm, 10)
    d10 = decrypt(vm, 10, 8)
    print(f"[STEP1] BootstrapTo(dst=10, src=0) ok=1 time={t1-t0:.3f}s")
    print(f"[STEP1] ct10 meta: level={lvl10} log2(scale)={sc10}")
    print("[STEP1] dec8(ct10=boot(ct0)) =", d10)
    print(f"[STEP1] l2_err(dec8, x_rep8) = {l2_err(d10, x_rep8):.3e}")

    # STEP2: mul+rescale: ct11=ct10^2, ct12=rescale(ct11)
    print("[STEP2] MulRelin: ct11 = ct10 * ct10")
    lib.CKKS_OpMulCC(vm, 11, 10, 10)
    lvl11, sc11 = ct_meta(vm, 11)
    d11 = decrypt(vm, 11, 8)
    print(f"[STEP2] ct11 meta: level={lvl11} log2(scale)={sc11}")
    print("[STEP2] dec8(ct11=ct10^2) =", d11)
    print(f"[STEP2] l2_err(dec8, x2_rep8) = {l2_err(d11, x2_rep8):.3e}")

    print("[STEP2] Rescale: ct12 = rescale(ct11)")
    lib.CKKS_OpRescale(vm, 12, 11)
    lvl12, sc12 = ct_meta(vm, 12)
    d12 = decrypt(vm, 12, 8)
    print(f"[STEP2] ct12 meta: level={lvl12} log2(scale)={sc12}")
    print("[STEP2] dec8(ct12=rescale(ct11)) =", d12)
    print(f"[STEP2] l2_err(dec8, x2_rep8) = {l2_err(d12, x2_rep8):.3e}")

    # STEP3: boot ct12 -> ct13
    t0 = time.time()
    ok = lib.CKKS_BootstrapTo(vm, 13, 12)
    t1 = time.time()
    if ok != 1:
        lib.CKKS_FreeVM(vm)
        raise RuntimeError("CKKS_BootstrapTo(dst=13, src=12) failed")
    lvl13, sc13 = ct_meta(vm, 13)
    d13 = decrypt(vm, 13, 8)
    print(f"[STEP3] BootstrapTo(dst=13, src=12) ok=1 time={t1-t0:.3f}s")
    print(f"[STEP3] ct13 meta: level={lvl13} log2(scale)={sc13}")
    print("[STEP3] dec8(ct13=boot(ct12)) =", d13)
    print(f"[STEP3] l2_err(dec8, x2_rep8) = {l2_err(d13, x2_rep8):.3e}")

    # loose correctness check (boot adds approx error)
    assert_close(d13, x2_rep8, 3e-1, "boot(x^2)")

    lib.CKKS_FreeVM(vm)
    print("BOOT TEST PASSED")


if __name__ == "__main__":
    main()
