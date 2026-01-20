# test_hevm_ops.py
import ctypes
import os
import struct
import tempfile
import math

SO = "./libLATTIGO_HEVM.so"
OUTN = 1 << 14

# Opcodes (must match hevm_stub.go)
OP_ENCODE   = 0
OP_ROTATEC  = 1
OP_NEGATEC  = 2
OP_RESCALEC = 3
OP_MODSWC   = 4
OP_UPSCALEC = 5
OP_ADDCC    = 6
OP_ADDCP    = 7
OP_MULCC    = 8
OP_MULCP    = 9
OP_BOOT     = 10

# Go stub fixed params:
# LogN=15, LogQ len = 14 => MaxLevel = 13 (0..13)
MAX_LEVEL = 13
DEFAULT_LOG2SCALE = 40


def pack_encode_rhs(level: int, log2scale: int) -> int:
    # rhs packs: level = rhs>>10, scale=rhs & 0x3FF
    if not (0 <= level <= 0x3F):
        raise ValueError("level out of range")
    if not (0 <= log2scale <= 0x3FF):
        raise ValueError("log2scale out of range")
    return ((level & 0x3F) << 10) | (log2scale & 0x3FF)


def write_constants(path, const_vectors):
    with open(path, "wb") as f:
        f.write(struct.pack("<q", len(const_vectors)))
        for vec in const_vectors:
            f.write(struct.pack("<q", len(vec)))
            for x in vec:
                f.write(struct.pack("<d", float(x)))


def write_hevm(path, *, arg_len, res_len,
              num_operations, num_ctxt_buffer, num_ptxt_buffer,
              arg_scale, arg_level, res_scale, res_level, res_dst,
              ops,
              init_level=0, reserved=0):
    magic = 0x4845564D
    hevm_header_size = 24
    config_body_length = 48

    assert len(arg_scale) == arg_len
    assert len(arg_level) == arg_len
    assert len(res_scale) == res_len
    assert len(res_level) == res_len
    assert len(res_dst) == res_len
    assert len(ops) == num_operations

    with open(path, "wb") as f:
        f.write(struct.pack("<IIQQ", magic, hevm_header_size, arg_len, res_len))
        f.write(struct.pack("<QQQQQQ",
                            config_body_length,
                            num_operations,
                            num_ctxt_buffer,
                            num_ptxt_buffer,
                            init_level,
                            reserved))
        for arr in (arg_scale, arg_level, res_scale, res_level, res_dst):
            for x in arr:
                f.write(struct.pack("<Q", int(x)))

        for (opcode, dst, lhs, rhs) in ops:
            f.write(struct.pack("<HHHH",
                                opcode & 0xFFFF, dst & 0xFFFF,
                                lhs & 0xFFFF, rhs & 0xFFFF))


def load_lib():
    lib = ctypes.CDLL(SO)

    # handle-based ABI
    lib.initFullVM.argtypes = [ctypes.c_char_p, ctypes.c_bool]
    lib.initFullVM.restype = ctypes.c_size_t

    lib.load.argtypes = [ctypes.c_size_t, ctypes.c_char_p, ctypes.c_char_p]
    lib.preprocess.argtypes = [ctypes.c_size_t]
    lib.run.argtypes = [ctypes.c_size_t]

    lib.encrypt.argtypes = [ctypes.c_size_t, ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    lib.decrypt_result.argtypes = [ctypes.c_size_t, ctypes.c_int64, ctypes.POINTER(ctypes.c_double)]

    if hasattr(lib, "freeVM"):
        lib.freeVM.argtypes = [ctypes.c_size_t]

    return lib


def assert_close(a, b, eps=1e-7):
    if math.isfinite(a) and math.isfinite(b) and abs(a - b) <= eps:
        return
    raise AssertionError(f"not close: {a} vs {b} (eps={eps})")


def run_program(lib, const_vectors, hevm_ops, *, res_dst0, input_vec,
                arg_log2scale=DEFAULT_LOG2SCALE,
                arg_level=MAX_LEVEL):
    with tempfile.TemporaryDirectory() as d:
        const_path = os.path.join(d, "constants.bin")
        hevm_path = os.path.join(d, "hevm.bin")
        write_constants(const_path, const_vectors)

        arg_len = 1
        res_len = 1

        num_ctxt_buffer = 8
        num_ptxt_buffer = 4

        arg_scale = [int(arg_log2scale)]
        arg_level_arr = [int(arg_level)]
        res_scale = [int(arg_log2scale)]
        res_level = [int(arg_level)]
        res_dst = [int(res_dst0)]

        write_hevm(
            hevm_path,
            arg_len=arg_len, res_len=res_len,
            num_operations=len(hevm_ops),
            num_ctxt_buffer=num_ctxt_buffer,
            num_ptxt_buffer=num_ptxt_buffer,
            arg_scale=arg_scale, arg_level=arg_level_arr,
            res_scale=res_scale, res_level=res_level,
            res_dst=res_dst,
            ops=hevm_ops
        )

        h = lib.initFullVM(b".", False)
        lib.load(h, const_path.encode(), hevm_path.encode())

        x = (ctypes.c_double * len(input_vec))(*[float(v) for v in input_vec])
        lib.encrypt(h, 0, x, len(input_vec))

        lib.preprocess(h)
        lib.run(h)

        outbuf = (ctypes.c_double * OUTN)()
        lib.decrypt_result(h, 0, outbuf)

        out8 = [float(outbuf[i]) for i in range(8)]

        if hasattr(lib, "freeVM"):
            lib.freeVM(h)

        return out8


def test_mulcp_then_rotate():
    """
    Mul by 2 then Rotate(+1).

    IMPORTANT:
    Your Go stub uses Lattigo evaluator.Rotate(in, k, out).
    In your current build, k=+1 behaves as LEFT rotate by 1:
        [2,4,6,8] -> [4,6,8,2]
    """
    consts = [[2.0]]
    rhs_encode = pack_encode_rhs(level=MAX_LEVEL, log2scale=DEFAULT_LOG2SCALE)

    ops = [
        (OP_ENCODE,   0, 0, rhs_encode),        # plains[0] = 2 at (MAX_LEVEL, 2^40)
        (OP_MULCP,    1, 0, 0),                 # ct1 = ct0 * plain0
        (OP_ROTATEC,  2, 1, (1 & 0xFFFF)),      # ct2 = rotate(ct1, +1)
    ]

    out8 = run_program(
        load_lib(),
        const_vectors=consts,
        hevm_ops=ops,
        res_dst0=2,
        input_vec=[1.0, 2.0, 3.0, 4.0],
        arg_level=MAX_LEVEL,
        arg_log2scale=DEFAULT_LOG2SCALE,
    )

    # LEFT rotate by 1: [2,4,6,8] -> [4,6,8,2]
    assert_close(out8[0], 4.0)
    assert_close(out8[1], 6.0)
    assert_close(out8[2], 8.0)
    assert_close(out8[3], 2.0)

    print("[PASS] test_mulcp_then_rotate, out[0:8] =", out8)


def test_addcc_then_negate():
    consts = [[]]
    ops = [
        (OP_ADDCC,   1, 0, 0),
        (OP_NEGATEC, 2, 1, 0),
    ]

    out8 = run_program(
        load_lib(),
        const_vectors=consts,
        hevm_ops=ops,
        res_dst0=2,
        input_vec=[1.0, 2.0, 3.0, 4.0],
        arg_level=MAX_LEVEL,
        arg_log2scale=DEFAULT_LOG2SCALE,
    )

    assert_close(out8[0], -2.0)
    assert_close(out8[1], -4.0)
    assert_close(out8[2], -6.0)
    assert_close(out8[3], -8.0)

    print("[PASS] test_addcc_then_negate, out[0:8] =", out8)


def test_addcp_with_constant():
    consts = [[10.0]]
    rhs_encode = pack_encode_rhs(level=MAX_LEVEL, log2scale=DEFAULT_LOG2SCALE)

    ops = [
        (OP_ENCODE, 0, 0, rhs_encode),  # plains[0] = 10 at (MAX_LEVEL, 2^40)
        (OP_ADDCP,  1, 0, 0),           # ct1 = ct0 + plain0
    ]

    out8 = run_program(
        load_lib(),
        const_vectors=consts,
        hevm_ops=ops,
        res_dst0=1,
        input_vec=[1.0, 2.0, 3.0, 4.0],
        arg_level=MAX_LEVEL,
        arg_log2scale=DEFAULT_LOG2SCALE,
    )

    assert_close(out8[0], 11.0)
    assert_close(out8[1], 12.0)
    assert_close(out8[2], 13.0)
    assert_close(out8[3], 14.0)

    print("[PASS] test_addcp_with_constant, out[0:8] =", out8)


def main():
    test_mulcp_then_rotate()
    test_addcc_then_negate()
    test_addcp_with_constant()
    print("ALL PASS")


if __name__ == "__main__":
    main()
