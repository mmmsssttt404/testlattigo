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

# ---- signatures ----
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

lib.CKKS_CTLevel.restype = ctypes.c_int
lib.CKKS_CTLevel.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.CKKS_CTLog2Scale.restype = ctypes.c_int
lib.CKKS_CTLog2Scale.argtypes = [ctypes.c_void_p, ctypes.c_int]

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

lib.CKKS_OpMulCP_Const.restype = None
lib.CKKS_OpMulCP_Const.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_double]
lib.CKKS_OpRescale.restype = None
lib.CKKS_OpRescale.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

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
    ls = lib.CKKS_CTLog2Scale(vm, idx)
    return lvl, ls


def l2_err(a, b):
    # sqrt(mean((a-b)^2))
    s = 0.0
    for x, y in zip(a, b):
        d = x - y
        s += d * d
    return math.sqrt(s / max(1, len(a)))


def assert_close(got, exp, eps, name):
    for i, (a, b) in enumerate(zip(got, exp)):
        if abs(a - b) > eps:
            raise AssertionError(f"{name} idx={i} got={a} exp={b} diff={a-b} eps={eps}")


def main():
    # CreateVM 只是占位，BootEnable 会切到 N16QP1546H192H32 并重建 keys/params
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

    logN = lib.CKKS_LogN(vm)
    slots = lib.CKKS_Slots(vm)
    maxLevel = lib.CKKS_MaxLevel(vm)
    logS = lib.CKKS_LogDefaultScale(vm)
    print(f"[CKKS] PARAM=N16QP1546H192H32 logN={logN} slots={slots} maxLevel={maxLevel} logS={logS}")

    # ---- STEP0: encrypt ----
    x = [0.5, -1.25, 2.0, 3.5]
    lvl0 = maxLevel
    lib.CKKS_EncryptTo(vm, 0, arr_d(x), len(x), lvl0, logS)
    d0 = decrypt(vm, 0, 8)
    meta0 = ct_meta(vm, 0)
    x_rep8 = [x[i % len(x)] for i in range(8)]
    print(f"[STEP0] EncryptTo(dst=0, level={lvl0}, logS={logS})")
    print(f"[STEP0] ct0 meta: level={meta0[0]} log2(scale)={meta0[1]}")
    print(f"[STEP0] dec8(ct0) = {d0}")
    print(f"[STEP0] l2_err(dec8, x_rep8) = {l2_err(d0, x_rep8):.3e}")
    assert_close(d0[:4], x, 2e-2, "decrypt(x)")

    # ---- STEP1: boot once (clean refresh) ----
    t0 = time.time()
    ok = lib.CKKS_BootstrapTo(vm, 10, 0)
    t1 = time.time()
    if ok != 1:
        lib.CKKS_FreeVM(vm)
        raise RuntimeError("BootstrapTo(dst=10, src=0) failed")
    d10 = decrypt(vm, 10, 8)
    meta10 = ct_meta(vm, 10)
    print(f"[STEP1] BootstrapTo(dst=10, src=0) ok=1 time={t1-t0:.3f}s")
    print(f"[STEP1] ct10 meta: level={meta10[0]} log2(scale)={meta10[1]}")
    print(f"[STEP1] dec8(ct10=boot(ct0)) = {d10}")
    print(f"[STEP1] l2_err(dec8, x_rep8) = {l2_err(d10, x_rep8):.3e}")

    # ---- STEP2: 用 Mul(×1.0)+Rescale 把 level 吃到 0 ----
    # 这里用 x * 1 的方式，理想情况下值不变，但会经历 scale^2 -> rescale 回到默认 scale
    cur = 10
    tmp_mul = 11
    tmp_rs = 12

    print("[STEP2] Consume levels by repeating: ct = Rescale( ct * 1.0 ) until level==0")
    step = 0
    while True:
        lvl = lib.CKKS_CTLevel(vm, cur)
        if lvl <= 0:
            break

        step += 1
        lib.CKKS_OpMulCP_Const(vm, tmp_mul, cur, ctypes.c_double(1.0))
        m_mul = ct_meta(vm, tmp_mul)

        lib.CKKS_OpRescale(vm, tmp_rs, tmp_mul)
        m_rs = ct_meta(vm, tmp_rs)

        d_rs = decrypt(vm, tmp_rs, 8)
        err = l2_err(d_rs, x_rep8)

        print(f"  [STEP2.{step}] after MulConst(1.0): ct{tmp_mul} level={m_mul[0]} log2(scale)={m_mul[1]}")
        print(f"  [STEP2.{step}] after Rescale     : ct{tmp_rs}  level={m_rs[0]} log2(scale)={m_rs[1]}")
        print(f"  [STEP2.{step}] dec8(ct{tmp_rs}) = {d_rs}")
        print(f"  [STEP2.{step}] l2_err(dec8, x_rep8) = {err:.3e}")

        # move
        cur = tmp_rs

    print(f"[STEP2] reached level={lib.CKKS_CTLevel(vm, cur)} (should be 0)")

    # ---- STEP3: boot again from level0 ----
    t0 = time.time()
    ok = lib.CKKS_BootstrapTo(vm, 13, cur)
    t1 = time.time()
    if ok != 1:
        lib.CKKS_FreeVM(vm)
        raise RuntimeError("BootstrapTo(dst=13, src=cur) failed")

    d13 = decrypt(vm, 13, 8)
    meta13 = ct_meta(vm, 13)
    print(f"[STEP3] BootstrapTo(dst=13, src={cur}) ok=1 time={t1-t0:.3f}s")
    print(f"[STEP3] ct13 meta: level={meta13[0]} log2(scale)={meta13[1]}")
    print(f"[STEP3] dec8(ct13=boot(ct{cur})) = {d13}")
    print(f"[STEP3] l2_err(dec8, x_rep8) = {l2_err(d13, x_rep8):.3e}")

    # 容忍 boot 引入的近似误差
    assert_close(d13, x_rep8, 3e-1, "boot(x) after level-consume")

    lib.CKKS_FreeVM(vm)
    print("BOOT TEST PASSED")


if __name__ == "__main__":
    main()
