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

# ---- signatures (match hevm_stub.go exports) ----
lib.initClientVM.restype = ctypes.c_size_t
lib.initClientVM.argtypes = [ctypes.c_char_p]
lib.freeVM.restype = None
lib.freeVM.argtypes = [ctypes.c_size_t]

lib.create_context.restype = ctypes.c_size_t
lib.create_context.argtypes = [ctypes.c_char_p]
lib.freeContext.restype = None
lib.freeContext.argtypes = [ctypes.c_size_t]

lib.loadClient.restype = None
lib.loadClient.argtypes = [ctypes.c_size_t, ctypes.c_size_t]

# encrypt(vm, idx, data*, n)
lib.encrypt.restype = None
lib.encrypt.argtypes = [ctypes.c_size_t, ctypes.c_longlong, ctypes.POINTER(ctypes.c_double), ctypes.c_int]

# decrypt(vm, idx, out*)
lib.decrypt.restype = None
lib.decrypt.argtypes = [ctypes.c_size_t, ctypes.c_longlong, ctypes.POINTER(ctypes.c_double)]


def arr_d(xs):
    a = (ctypes.c_double * len(xs))()
    for i, x in enumerate(xs):
        a[i] = float(x)
    return a


def decrypt(vm, idx, n=8):
    # IMPORTANT: Go side assumes out buffer is at least 1<<14 doubles
    OUTN = 1 << 14
    out = (ctypes.c_double * OUTN)()
    lib.decrypt(vm, idx, out)
    return [out[i] for i in range(n)]


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


def main():
    # ---- STEP(-1): init context + loadClient (HEVM needs this to build params/keys) ----
    ctx = lib.create_context(b"")
    if not ctx:
        raise RuntimeError("create_context failed")

    vm = lib.initClientVM(b"")
    if not vm:
        lib.freeContext(ctx)
        raise RuntimeError("initClientVM failed")

    lib.loadClient(vm, ctx)

    # ---- STEP0: encrypt ----
    x = [0.5, -1.25, 2.0, 3.5]
    lib.encrypt(vm, 0, arr_d(x), len(x))
    d0 = decrypt(vm, 0, 8)
    x_rep8 = [x[i % len(x)] for i in range(8)]

    print(f"[STEP0] dec8(ct0) = {d0}")
    print(f"[STEP0] l2_err(dec8, x_rep8) = {l2_err(d0, x_rep8):.3e}")
    assert_close(d0[:4], x, 2e-2, "decrypt(x)")

    # NOTE: HEVM stub does not export boot/mul/rescale ops like CKKS_OPS,
    # so this test only validates that encrypt/decrypt path is alive.
    lib.freeVM(vm)
    lib.freeContext(ctx)
    print("HEVM BASIC TEST PASSED")


if __name__ == "__main__":
    main()
