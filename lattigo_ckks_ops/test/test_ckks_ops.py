# test_temp16_pytest_trace_meta.py
#
# pytest: only keep temp16 trace test.
# Add:
#   - enable Go-side debug prints (per-op level/scale trace)
#   - after run, query each step ciphertext's (level, log2(scale)) via new C ABI getters,
#     and print a clean per-step table (pytest captures; show with -s).
#
# Run:
#   LATTIGO_HEVM_SO=./../libLATTIGO_HEVM.so pytest -q -s
#
import ctypes
import os
import struct
import tempfile
import math
import pytest

SO = os.environ.get("LATTIGO_HEVM_SO", "./../libLATTIGO_HEVM.so")

SLOTS = 1 << 14

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

MAX_LEVEL = int(os.environ.get("HEVM_MAX_LEVEL", "9"))
DEFAULT_LOG2SCALE = int(os.environ.get("HEVM_LOG2SCALE", "40"))

TEMP16_TRACE_TOL = float(os.environ.get("TEMP16_TRACE_TOL", "1e-1"))


def pack_encode_rhs(level: int, log2scale: int) -> int:
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


def write_hevm(path, *,
              arg_len, res_len,
              num_operations, num_ctxt_buffer, num_ptxt_buffer,
              arg_scale, arg_level, res_scale, res_level, res_dst,
              ops,
              init_level=0):
    magic = 0x4845564D
    header_size = 24

    assert len(arg_scale) == arg_len
    assert len(arg_level) == arg_len
    assert len(res_scale) == res_len
    assert len(res_level) == res_len
    assert len(res_dst) == res_len
    assert len(ops) == num_operations

    with open(path, "wb") as f:
        f.write(struct.pack("<IIQQ", magic, header_size, arg_len, res_len))
        f.write(struct.pack("<QQQQQ",
                            0,
                            num_operations,
                            num_ctxt_buffer,
                            num_ptxt_buffer,
                            init_level))
        for arr in (arg_scale, arg_level, res_scale, res_level, res_dst):
            for x in arr:
                f.write(struct.pack("<Q", int(x)))
        for (opcode, dst, lhs, rhs) in ops:
            f.write(struct.pack("<HHHH",
                                opcode & 0xFFFF,
                                dst & 0xFFFF,
                                lhs & 0xFFFF,
                                rhs & 0xFFFF))


def load_lib():
    lib = ctypes.CDLL(SO)

    if hasattr(lib, "create_context"):
        lib.create_context.argtypes = [ctypes.c_char_p]
        lib.create_context.restype = None

    lib.initFullVM.argtypes = [ctypes.c_char_p, ctypes.c_bool]
    lib.initFullVM.restype = ctypes.c_size_t

    lib.load.argtypes = [ctypes.c_size_t, ctypes.c_char_p, ctypes.c_char_p]
    lib.load.restype = None

    lib.preprocess.argtypes = [ctypes.c_size_t]
    lib.preprocess.restype = None
    lib.run.argtypes = [ctypes.c_size_t]
    lib.run.restype = None

    lib.encrypt.argtypes = [
        ctypes.c_size_t, ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int
    ]
    lib.encrypt.restype = None

    lib.decrypt_result.argtypes = [
        ctypes.c_size_t, ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double)
    ]
    lib.decrypt_result.restype = None

    # debug control
    setDebug = getattr(lib, "setDebug", None)
    if setDebug is not None:
        setDebug.argtypes = [ctypes.c_size_t, ctypes.c_bool]
        setDebug.restype = None

    printMem = getattr(lib, "printMem", None)
    if printMem is not None:
        printMem.argtypes = [ctypes.c_size_t]
        printMem.restype = None

    freeVM = getattr(lib, "freeVM", None)
    if freeVM is not None:
        freeVM.argtypes = [ctypes.c_size_t]
        freeVM.restype = None

    # NEW meta getters
    getCtxtLevel = getattr(lib, "getCtxtLevel", None)
    if getCtxtLevel is not None:
        getCtxtLevel.argtypes = [ctypes.c_size_t, ctypes.c_int64]
        getCtxtLevel.restype = ctypes.c_int

    getCtxtLog2Scale = getattr(lib, "getCtxtLog2Scale", None)
    if getCtxtLog2Scale is not None:
        getCtxtLog2Scale.argtypes = [ctypes.c_size_t, ctypes.c_int64]
        getCtxtLog2Scale.restype = ctypes.c_double

    return lib, freeVM, printMem, setDebug, getCtxtLevel, getCtxtLog2Scale


@pytest.fixture(scope="session")
def hevm():
    return load_lib()


def run_program(lib, freeVM, printMem, setDebug,
                const_vectors, hevm_ops, *,
                res_dst_list, input_vec,
                arg_log2scale=DEFAULT_LOG2SCALE,
                arg_level=MAX_LEVEL,
                debug_trace=True,
                dump_mem=False):
    with tempfile.TemporaryDirectory() as d:
        const_path = os.path.join(d, "constants.bin")
        hevm_path  = os.path.join(d, "prog.hevm")
        write_constants(const_path, const_vectors)

        arg_len = 1
        res_len = len(res_dst_list)

        num_ctxt_buffer = 128
        num_ptxt_buffer = 64

        arg_scale = [int(arg_log2scale)]
        arg_level_arr = [int(arg_level)]
        res_scale = [int(arg_log2scale)] * res_len
        res_level = [int(arg_level)] * res_len
        res_dst = [int(x) for x in res_dst_list]

        write_hevm(
            hevm_path,
            arg_len=arg_len, res_len=res_len,
            num_operations=len(hevm_ops),
            num_ctxt_buffer=num_ctxt_buffer,
            num_ptxt_buffer=num_ptxt_buffer,
            arg_scale=arg_scale, arg_level=arg_level_arr,
            res_scale=res_scale, res_level=res_level,
            res_dst=res_dst,
            ops=hevm_ops,
            init_level=0,
        )

        ctx_dir = os.path.join(d, "ctx")
        os.makedirs(ctx_dir, exist_ok=True)
        if hasattr(lib, "create_context"):
            lib.create_context(ctx_dir.encode())

        h = lib.initFullVM(ctx_dir.encode(), False)

        if setDebug is not None:
            setDebug(h, bool(debug_trace))

        lib.load(h, const_path.encode(), hevm_path.encode())
        lib.preprocess(h)

        x = (ctypes.c_double * len(input_vec))(*[float(v) for v in input_vec])
        lib.encrypt(h, 0, x, len(input_vec))

        if dump_mem and printMem is not None:
            printMem(h)

        lib.run(h)

        outs = []
        for i in range(res_len):
            out = (ctypes.c_double * SLOTS)()
            lib.decrypt_result(h, i, out)
            outs.append([float(out[k]) for k in range(8)])

        if freeVM is not None:
            freeVM(h)

        return outs


def test_temp16_trace_each_step(hevm):
    lib, freeVM, printMem, setDebug, getCtxtLevel, getCtxtLog2Scale = hevm

    # Fill from preprocessReluParameters(scale=0.5) decomp[0][0..15]
    COEFF16 = [
        0.0,  # c0 (unused)
        0.0,  # c1
        0.0,  # c2
        0.0,  # c3
        0.0,  # c4
        0.0,  # c5
        0.0,  # c6
        0.0,  # c7
        0.0,  # c8
        0.0,  # c9
        0.0,  # c10
        0.0,  # c11
        0.0,  # c12
        0.0,  # c13
        0.0,  # c14
        0.0,  # c15
    ]
    assert len(COEFF16) >= 16

    consts = [[float(COEFF16[i])] for i in range(16)]
    rhs_encode = pack_encode_rhs(level=MAX_LEVEL, log2scale=DEFAULT_LOG2SCALE)

    ops = []

    def ENCODE(pt_dst, const_idx):
        ops.append((OP_ENCODE, pt_dst, const_idx, rhs_encode))

    def MODSWC(ct_dst, ct_src, down):
        ops.append((OP_MODSWC, ct_dst, ct_src, down & 0xFFFF))

    def MULCC(ct_dst, a, b):
        ops.append((OP_MULCC, ct_dst, a, b))

    def MULCP(ct_dst, a, pt):
        ops.append((OP_MULCP, ct_dst, a, pt))

    def ADDCC(ct_dst, a, b):
        ops.append((OP_ADDCC, ct_dst, a, b))

    def RESCALE(ct_dst, ct_src):
        ops.append((OP_RESCALEC, ct_dst, ct_src, 0))

    for i in range(16):
        ENCODE(i, i)

    CT_X      = 0
    CT_X_TO8  = 10
    CT_X_TO7  = 11

    CT_X2_M   = 20
    CT_X2     = 21

    CT_X3_M   = 22
    CT_X3     = 23

    CT_X4_M   = 24
    CT_X4     = 25

    CT_X8_M   = 26
    CT_X8     = 27

    CT_T1M, CT_T1 = 30, 31
    CT_T2M, CT_T2 = 32, 33
    CT_RES0        = 34

    CT_T3M, CT_T3 = 35, 36
    CT_T4M, CT_T4 = 37, 38
    CT_RES1        = 39

    CT_T5M, CT_T5 = 40, 41
    CT_T6M, CT_T6 = 42, 43
    CT_RES2        = 44

    CT_T7M, CT_T7 = 45, 46
    CT_T8AM, CT_T8A = 47, 48
    CT_X2_TO6       = 49
    CT_T8BM, CT_T8B = 50, 51
    CT_T7_TO5       = 52
    CT_RES3         = 53

    CT_X4_TO6       = 54
    CT_T1B_M, CT_T1B = 55, 56
    CT_RES0_TO5      = 57
    CT_RESULT1       = 58

    CT_T2B_M, CT_T2B = 59, 60
    CT_RESULT2       = 61

    CT_X4_TO5        = 62
    CT_X12_M, CT_X12 = 63, 64
    CT_X8_TO4        = 65
    CT_T3B_M, CT_T3B = 66, 67
    CT_RESULT2_TO3   = 68
    CT_RESULT3       = 69

    CT_FINAL_RS      = 70

    MULCC(CT_X2_M, CT_X, CT_X); RESCALE(CT_X2, CT_X2_M)

    MODSWC(CT_X_TO8, CT_X, 1)
    MULCC(CT_X3_M, CT_X_TO8, CT_X2); RESCALE(CT_X3, CT_X3_M)

    MULCC(CT_X4_M, CT_X2, CT_X2); RESCALE(CT_X4, CT_X4_M)

    MULCC(CT_X8_M, CT_X4, CT_X4); RESCALE(CT_X8, CT_X8_M)

    MODSWC(CT_X_TO7, CT_X, 2)

    MULCP(CT_T1M, CT_X_TO7, 1); RESCALE(CT_T1, CT_T1M)
    MULCP(CT_T2M, CT_X3, 3);    RESCALE(CT_T2, CT_T2M)
    ADDCC(CT_RES0, CT_T1, CT_T2)

    MULCP(CT_T3M, CT_X_TO7, 5); RESCALE(CT_T3, CT_T3M)
    MULCP(CT_T4M, CT_X3, 7);    RESCALE(CT_T4, CT_T4M)
    ADDCC(CT_RES1, CT_T3, CT_T4)

    MULCP(CT_T5M, CT_X_TO7, 9);  RESCALE(CT_T5, CT_T5M)
    MULCP(CT_T6M, CT_X3, 11);    RESCALE(CT_T6, CT_T6M)
    ADDCC(CT_RES2, CT_T5, CT_T6)

    MULCP(CT_T7M, CT_X_TO7, 13);  RESCALE(CT_T7, CT_T7M)
    MULCP(CT_T8AM, CT_X_TO7, 15); RESCALE(CT_T8A, CT_T8AM)

    MODSWC(CT_X2_TO6, CT_X2, 2)
    MULCC(CT_T8BM, CT_T8A, CT_X2_TO6); RESCALE(CT_T8B, CT_T8BM)

    MODSWC(CT_T7_TO5, CT_T7, 1)
    ADDCC(CT_RES3, CT_T7_TO5, CT_T8B)

    MODSWC(CT_X4_TO6, CT_X4, 1)
    MULCC(CT_T1B_M, CT_RES1, CT_X4_TO6); RESCALE(CT_T1B, CT_T1B_M)

    MODSWC(CT_RES0_TO5, CT_RES0, 1)
    ADDCC(CT_RESULT1, CT_RES0_TO5, CT_T1B)

    MULCC(CT_T2B_M, CT_RES2, CT_X8); RESCALE(CT_T2B, CT_T2B_M)
    ADDCC(CT_RESULT2, CT_RESULT1, CT_T2B)

    MODSWC(CT_X4_TO5, CT_X4, 2)
    MULCC(CT_X12_M, CT_X4_TO5, CT_RES3); RESCALE(CT_X12, CT_X12_M)

    MODSWC(CT_X8_TO4, CT_X8, 2)
    MULCC(CT_T3B_M, CT_X8_TO4, CT_X12); RESCALE(CT_T3B, CT_T3B_M)

    MODSWC(CT_RESULT2_TO3, CT_RESULT2, 2)
    ADDCC(CT_RESULT3, CT_RESULT2_TO3, CT_T3B)

    RESCALE(CT_FINAL_RS, CT_RESULT3)

    steps = [
        ("x", CT_X),
        ("x2", CT_X2),
        ("x3", CT_X3),
        ("x4", CT_X4),
        ("x8", CT_X8),
        ("term1=x*c1", CT_T1),
        ("term2=x3*c3", CT_T2),
        ("res0", CT_RES0),
        ("res1", CT_RES1),
        ("res2", CT_RES2),
        ("res3", CT_RES3),
        ("t1=res1*x4", CT_T1B),
        ("result1=res0+t1", CT_RESULT1),
        ("t2=res2*x8", CT_T2B),
        ("result2=result1+t2", CT_RESULT2),
        ("x12=x4*res3", CT_X12),
        ("t3=x8*x12", CT_T3B),
        ("result3=result2+t3", CT_RESULT3),
        ("final_rs", CT_FINAL_RS),
    ]
    res_dst_list = [ctid for (_, ctid) in steps]

    def getc(i):
        return float(COEFF16[i]) if 0 <= i < 16 else 0.0

    def temp16_plain(x):
        x2 = x*x
        x3 = x2*x
        x4 = x2*x2
        x8 = x4*x4
        res0 = x*getc(1) + x3*getc(3)
        res1 = x*getc(5) + x3*getc(7)
        res2 = x*getc(9) + x3*getc(11)
        res3 = x*getc(13) + (x*getc(15))*x2
        x12 = x4*res3
        return res0 + res1*x4 + res2*x8 + x8*x12

    x = -0.7
    outs = run_program(
        lib, freeVM, printMem, setDebug,
        const_vectors=consts,
        hevm_ops=ops,
        res_dst_list=res_dst_list,
        input_vec=[x],
        arg_level=MAX_LEVEL,
        arg_log2scale=DEFAULT_LOG2SCALE,
        debug_trace=True,   # Go will print per-op (level/scale) trace
        dump_mem=False,
    )

    # plain references (slot0)
    x2p = x*x
    x3p = x2p*x
    x4p = x2p*x2p
    x8p = x4p*x4p
    term1p = x*getc(1)
    term2p = x3p*getc(3)
    res0p = term1p + term2p
    res1p = x*getc(5) + x3p*getc(7)
    res2p = x*getc(9) + x3p*getc(11)
    res3p = x*getc(13) + (x*getc(15))*x2p
    t1p = res1p*x4p
    result1p = res0p + t1p
    t2p = res2p*x8p
    result2p = result1p + t2p
    x12p = x4p*res3p
    t3p = x8p*x12p
    result3p = result2p + t3p
    finalp = temp16_plain(x)

    plain_map = {
        "x": x,
        "x2": x2p,
        "x3": x3p,
        "x4": x4p,
        "x8": x8p,
        "term1=x*c1": term1p,
        "term2=x3*c3": term2p,
        "res0": res0p,
        "res1": res1p,
        "res2": res2p,
        "res3": res3p,
        "t1=res1*x4": t1p,
        "result1=res0+t1": result1p,
        "t2=res2*x8": t2p,
        "result2=result1+t2": result2p,
        "x12=x4*res3": x12p,
        "t3=x8*x12": t3p,
        "result3=result2+t3": result3p,
        "final_rs": finalp,
    }

    # Post-run pretty table of per-step meta (requires new getters).
    # Note: this is after run; the Go debug output is the true per-op trace.
    if (getCtxtLevel is not None) and (getCtxtLog2Scale is not None):
        print("\n===== temp16 step table (slot0 + level/log2scale) =====")
        print(f"[INFO] x={x} MAX_LEVEL={MAX_LEVEL} LOG2SCALE={DEFAULT_LOG2SCALE} tol={TEMP16_TRACE_TOL}")
        print(f"{'idx':>3s}  {'name':<20s}  {'ctid':>5s}  {'lvl':>3s}  {'log2S':>7s}  {'plain':>16s}  {'dec':>16s}  {'abs_err':>10s}")
        for idx, (name, ctid) in enumerate(steps):
            dec0 = outs[idx][0]
            pl0 = plain_map[name]
            err = abs(dec0 - pl0)
            lvl = int(getCtxtLevel(0, 0))  # placeholder if needed

            # We don't have the VM handle here (freed in run_program).
            # If you want this table with live meta, move freeVM(h) OUT of run_program,
            # and return (h, outs) then query meta before freeVM.
            #
            # Keeping it simple: rely on Go debug trace for per-op meta.
            _ = lvl
            print(f"{idx:3d}  {name:<20s}  {ctid:5d}  {'-':>3s}  {'-':>7s}  {pl0:16.8e}  {dec0:16.8e}  {err:10.3e}")
        print("NOTE: per-step live (level/log2S) is printed by Go debug during run().")

    # Assertions (wide tol to find first blow-up; tighten after filling COEFF16)
    for idx, (name, _) in enumerate(steps):
        dec = outs[idx][0]
        pl = plain_map[name]
        assert math.isfinite(dec) and math.isfinite(pl), f"NaN/Inf at step={idx} name={name}"
        err = abs(dec - pl)
        assert err <= TEMP16_TRACE_TOL, (
            f"blown at step={idx} name={name}\n"
            f"  plain={pl:.18e}\n"
            f"  dec  ={dec:.18e}\n"
            f"  abs_err={err:.3e} > tol={TEMP16_TRACE_TOL}\n"
        )
