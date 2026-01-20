import ctypes
import math
import os
import sys

LIB = os.path.join(os.path.dirname(__file__), "libLATTIGO_CKKS_OPS.so")
if not os.path.exists(LIB):
    print("missing:", LIB)
    print("run: ./build_so.sh")
    sys.exit(1)

lib = ctypes.CDLL(LIB)

# --- signatures ---
lib.CKKS_CreateVM.restype = ctypes.c_void_p
lib.CKKS_CreateVM.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]

lib.CKKS_FreeVM.restype = None
lib.CKKS_FreeVM.argtypes = [ctypes.c_void_p]

lib.CKKS_Slots.restype = ctypes.c_int
lib.CKKS_Slots.argtypes = [ctypes.c_void_p]

lib.CKKS_GenRotKey.restype = None
lib.CKKS_GenRotKey.argtypes = [ctypes.c_void_p, ctypes.c_int]

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

lib.CKKS_OpAddCC.restype = None
lib.CKKS_OpAddCC.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]

lib.CKKS_OpAddCP_Const.restype = None
lib.CKKS_OpAddCP_Const.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_double]

lib.CKKS_OpMulCC.restype = None
lib.CKKS_OpMulCC.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]

lib.CKKS_OpMulCP_Const.restype = None
lib.CKKS_OpMulCP_Const.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_double]

lib.CKKS_OpRotate.restype = None
lib.CKKS_OpRotate.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]

lib.CKKS_OpNegate.restype = None
lib.CKKS_OpNegate.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

lib.CKKS_OpRescale.restype = None
lib.CKKS_OpRescale.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

lib.CKKS_OpDropLevel.restype = None
lib.CKKS_OpDropLevel.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]


def arr_d(xs):
    a = (ctypes.c_double * len(xs))()
    for i, x in enumerate(xs):
        a[i] = float(x)
    return a


def decrypt(vm, idx, n=16):
    out = (ctypes.c_double * n)()
    lib.CKKS_DecryptFrom(vm, idx, out, n)
    return [out[i] for i in range(n)]


def assert_close(got, exp, eps=1e-2, name=""):
    for i, (a, b) in enumerate(zip(got, exp)):
        if abs(a - b) > eps:
            raise AssertionError(f"{name} idx={i} got={a} exp={b} diff={a-b}")


def main():
    # logN=14 => slots=8192, levels=8, scale=2^40, ct buffer=16
    vm = lib.CKKS_CreateVM(14, 8, 40, 16)
    if not vm:
        raise RuntimeError("CreateVM failed")

    slots = lib.CKKS_Slots(vm)
    print("slots =", slots)

    x = [0.5, -1.25, 2.0, 3.5]
    y = [1.0, 2.0, -0.5, 4.0]

    # Encrypt x -> ct0, y -> ct1 (level=-1 means "default max" in our wrapper, but we pass explicit here)
    lvl = 7
    logS = 40
    lib.CKKS_EncryptTo(vm, 0, arr_d(x), len(x), lvl, logS)
    lib.CKKS_EncryptTo(vm, 1, arr_d(y), len(y), lvl, logS)

    # Decrypt sanity
    dx = decrypt(vm, 0, 8)
    dy = decrypt(vm, 1, 8)
    print("dx =", dx[:8])
    print("dy =", dy[:8])
    assert_close(dx[:4], x, 2e-2, "decrypt(x)")
    assert_close(dy[:4], y, 2e-2, "decrypt(y)")

    # addcc: ct2 = ct0 + ct1
    lib.CKKS_OpAddCC(vm, 2, 0, 1)
    d2 = decrypt(vm, 2, 8)
    exp2 = [(x[i % len(x)] + y[i % len(y)]) for i in range(8)]
    assert_close(d2, exp2, 2e-2, "addcc")

    # addcp const: ct3 = ct0 + 3.0
    lib.CKKS_OpAddCP_Const(vm, 3, 0, ctypes.c_double(3.0))
    d3 = decrypt(vm, 3, 8)
    exp3 = [(x[i % len(x)] + 3.0) for i in range(8)]
    assert_close(d3, exp3, 2e-2, "addcp_const")

    # mulcp const: ct4 = ct0 * 2.5
    lib.CKKS_OpMulCP_Const(vm, 4, 0, ctypes.c_double(2.5))
    d4 = decrypt(vm, 4, 8)
    exp4 = [(x[i % len(x)] * 2.5) for i in range(8)]
    assert_close(d4, exp4, 5e-2, "mulcp_const")

    # mulcc: ct5 = ct0 * ct1 (relinearized)
    lib.CKKS_OpMulCC(vm, 5, 0, 1)
    d5 = decrypt(vm, 5, 8)
    exp5 = [(x[i % len(x)] * y[i % len(y)]) for i in range(8)]
    assert_close(d5, exp5, 1e-1, "mulcc")

    # negate: ct6 = -ct0
    lib.CKKS_OpNegate(vm, 6, 0)
    d6 = decrypt(vm, 6, 8)
    exp6 = [-(x[i % len(x)]) for i in range(8)]
    assert_close(d6, exp6, 2e-2, "negate")

    # rotate: generate rot key then rotate
    k = 1
    lib.CKKS_GenRotKey(vm, k)
    lib.CKKS_OpRotate(vm, 7, 0, k)
    d7 = decrypt(vm, 7, 8)
    # CKKS rotate by +1 means cyclic shift left by 1 in packed slots (Lattigo convention)
    # We only validate on the first few slots based on how we replicated x.
    # The replicated slot pattern is [0.5, -1.25, 2.0, 3.5, 0.5, -1.25, ...]
    base = [x[i % len(x)] for i in range(16)]
    rot = base[1:] + base[:1]
    assert_close(d7[:8], rot[:8], 5e-2, "rotate(+1)")

    # rescale: do mul then rescale
    lib.CKKS_OpMulCC(vm, 8, 0, 1)
    lib.CKKS_OpRescale(vm, 9, 8)
    d9 = decrypt(vm, 9, 8)
    # after rescale values still near x*y
    assert_close(d9, exp5, 2e-1, "mulcc+rescale")

    lib.CKKS_FreeVM(vm)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
