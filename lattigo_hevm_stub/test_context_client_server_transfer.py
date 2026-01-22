# test_context_client_server_transfer.py
# Add missing-function tests:
# - create_context writes file + returns handle
# - loadClient loads ctx into client/server
# - getCtxt / setCtxt / freeCtxt roundtrip transfer (client->server->client)
# - invalid index behavior (getCtxt returns 0; setCtxt no-op)
# - multiple transfers reuse (repeat roundtrip without corruption)
#
# Assumptions about the Go side (recommended contract):
#   - create_context(path) returns non-zero ctx handle, and writes ctx file
#   - freeContext(ctxH) releases ctx handle
#   - getCtxt(vm, idx) returns non-zero ciphertext handle if idx valid+present, else 0
#   - setCtxt(vm, idx, ctH) imports ciphertext (deep copy), ignores ctH==0 or invalid idx
#   - freeCtxt(ctH) releases ciphertext handle (safe on ctH==0)
#
import ctypes
import os
import struct
import tempfile
import math

SO = "./libLATTIGO_HEVM.so"
OUTN = 1 << 14

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

    with open(path, "wb") as f:
        f.write(struct.pack("<IIQQ", magic, hevm_header_size, arg_len, res_len))
        f.write(struct.pack("<QQQQQQ",
                            config_body_length, num_operations,
                            num_ctxt_buffer, num_ptxt_buffer,
                            init_level, reserved))
        for arr in (arg_scale, arg_level, res_scale, res_level, res_dst):
            for x in arr:
                f.write(struct.pack("<Q", int(x)))
        for (opcode, dst, lhs, rhs) in ops:
            f.write(struct.pack("<HHHH",
                                opcode & 0xFFFF, dst & 0xFFFF,
                                lhs & 0xFFFF, rhs & 0xFFFF))


def assert_close(a, b, eps=1e-6):
    if math.isfinite(a) and math.isfinite(b) and abs(a - b) <= eps:
        return
    raise AssertionError(f"not close: {a} vs {b} (eps={eps})")


def load_lib():
    lib = ctypes.CDLL(SO)

    # uintptr_t as uint64 on 64-bit
    U = ctypes.c_uint64

    lib.initClientVM.argtypes = [ctypes.c_char_p]
    lib.initClientVM.restype  = U
    lib.initServerVM.argtypes = [ctypes.c_char_p]
    lib.initServerVM.restype  = U
    lib.freeVM.argtypes       = [U]

    lib.create_context.argtypes = [ctypes.c_char_p]
    lib.create_context.restype  = U
    lib.freeContext.argtypes    = [U]

    lib.loadClient.argtypes   = [U, U]
    lib.loadProgram.argtypes  = [U, ctypes.c_char_p, ctypes.c_char_p]
    lib.preprocess.argtypes   = [U]
    lib.run.argtypes          = [U]

    lib.encrypt.argtypes = [U, ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    lib.decrypt_result.argtypes = [U, ctypes.c_int64, ctypes.POINTER(ctypes.c_double)]

    lib.getCtxt.argtypes = [U, ctypes.c_int64]
    lib.getCtxt.restype  = U
    lib.setCtxt.argtypes = [U, ctypes.c_int64, U]
    lib.freeCtxt.argtypes = [U]

    return lib


def decrypt_first4(lib, vm, res_idx=0):
    outbuf = (ctypes.c_double * OUTN)()
    lib.decrypt_result(vm, res_idx, outbuf)
    return [float(outbuf[i]) for i in range(4)]


def build_add10_program(tmpdir):
    const_path = os.path.join(tmpdir, "constants.bin")
    hevm_path  = os.path.join(tmpdir, "hevm.bin")

    # Program: y = x + 10
    consts = [[10.0]]
    write_constants(const_path, consts)

    arg_len = 1
    res_len = 1
    num_ctxt_buffer = 8
    num_ptxt_buffer = 4

    arg_scale = [40]
    arg_level = [0]
    res_scale = [40]
    res_level = [0]
    res_dst   = [1]   # output in ciphers[1]

    # ENCODE rhs packing: (level<<10)|(log2scale)
    rhs_pack = ((0 & 0x3F) << 10) | (40 & 0x3FF)

    ops = [
        (OP_ENCODE, 0, 0, rhs_pack),  # plains[0]=10
        (OP_ADDCP,  1, 0, 0),         # c1 = c0 + p0
    ]

    write_hevm(
        hevm_path,
        arg_len=arg_len, res_len=res_len,
        num_operations=len(ops),
        num_ctxt_buffer=num_ctxt_buffer,
        num_ptxt_buffer=num_ptxt_buffer,
        arg_scale=arg_scale, arg_level=arg_level,
        res_scale=res_scale, res_level=res_level,
        res_dst=res_dst,
        ops=ops
    )

    return const_path, hevm_path


def test_create_context_writes_file(lib, ctx_path_bytes):
    ctxH = lib.create_context(ctx_path_bytes)
    assert ctxH != 0, "create_context returned 0"
    ctx_path = ctx_path_bytes.decode()
    assert os.path.exists(ctx_path), f"context file not created: {ctx_path}"
    assert os.path.getsize(ctx_path) > 0, "context file size is 0"
    return ctxH


def test_getctxt_invalid_index_returns_zero(lib, client):
    # Expect contract: invalid index => 0
    bad1 = lib.getCtxt(client, -1)
    bad2 = lib.getCtxt(client, 999999)
    assert bad1 == 0, f"getCtxt(-1) expected 0, got {bad1}"
    assert bad2 == 0, f"getCtxt(huge) expected 0, got {bad2}"

    # setCtxt invalid index should be no-op (must not crash)
    lib.setCtxt(client, -1, 0)
    lib.setCtxt(client, 999999, 0)

    # freeCtxt(0) should be safe (must not crash)
    lib.freeCtxt(0)


def test_roundtrip_transfer_and_decrypt(lib, ctxH, const_path, hevm_path):
    # 1) client/server VM and load context
    client = lib.initClientVM(b".")
    server = lib.initServerVM(b".")
    assert client != 0 and server != 0

    lib.loadClient(client, ctxH)
    lib.loadClient(server, ctxH)

    # 2) load program on both
    lib.loadProgram(client, const_path.encode(), hevm_path.encode())
    lib.loadProgram(server, const_path.encode(), hevm_path.encode())

    # 3) client encrypt input into ciphers[0]
    x = (ctypes.c_double * 4)(1.0, 2.0, 3.0, 4.0)
    lib.encrypt(client, 0, x, 4)

    # 4) export ciphertext handle from client and import into server
    ct0 = lib.getCtxt(client, 0)
    assert ct0 != 0, "getCtxt(client,0) returned 0"
    lib.setCtxt(server, 0, ct0)
    lib.freeCtxt(ct0)

    # 5) server preprocess+run
    lib.preprocess(server)
    lib.run(server)

    # 6) export result ciphertext from server and import into client (to decrypt)
    ct1 = lib.getCtxt(server, 1)
    assert ct1 != 0, "getCtxt(server,1) returned 0"
    lib.setCtxt(client, 1, ct1)
    lib.freeCtxt(ct1)

    # 7) client decrypt result
    got = decrypt_first4(lib, client, res_idx=0)
    print("got[0:4] =", got)

    assert_close(got[0], 11.0)
    assert_close(got[1], 12.0)
    assert_close(got[2], 13.0)
    assert_close(got[3], 14.0)

    # cleanup
    lib.freeVM(client)
    lib.freeVM(server)


def test_multiple_roundtrips_reuse_ok(lib, ctxH, const_path, hevm_path, rounds=3):
    client = lib.initClientVM(b".")
    server = lib.initServerVM(b".")
    assert client != 0 and server != 0

    lib.loadClient(client, ctxH)
    lib.loadClient(server, ctxH)

    lib.loadProgram(client, const_path.encode(), hevm_path.encode())
    lib.loadProgram(server, const_path.encode(), hevm_path.encode())

    for t in range(rounds):
        # vary input to ensure no stale ciphertext reuse
        base = float(t + 1)
        x = (ctypes.c_double * 4)(base, base + 1, base + 2, base + 3)
        lib.encrypt(client, 0, x, 4)

        ct0 = lib.getCtxt(client, 0)
        assert ct0 != 0
        lib.setCtxt(server, 0, ct0)
        lib.freeCtxt(ct0)

        lib.preprocess(server)
        lib.run(server)

        ct1 = lib.getCtxt(server, 1)
        assert ct1 != 0
        lib.setCtxt(client, 1, ct1)
        lib.freeCtxt(ct1)

        got = decrypt_first4(lib, client, res_idx=0)
        exp = [base + 10, base + 11, base + 12, base + 13]
        for i in range(4):
            assert_close(got[i], exp[i])
        print(f"[round {t}] got[0:4] =", got)

    lib.freeVM(client)
    lib.freeVM(server)


def main():
    lib = load_lib()

    with tempfile.TemporaryDirectory() as d:
        ctx_path = os.path.join(d, "ctx.gob").encode()
        const_path, hevm_path = build_add10_program(d)

        # A) create_context: handle + file written
        ctxH = test_create_context_writes_file(lib, ctx_path)

        # B) basic invalid-index / no-crash contracts
        tmp_client = lib.initClientVM(b".")
        test_getctxt_invalid_index_returns_zero(lib, tmp_client)
        lib.freeVM(tmp_client)

        # C) full roundtrip client->server->client using getCtxt/setCtxt/freeCtxt
        test_roundtrip_transfer_and_decrypt(lib, ctxH, const_path, hevm_path)

        # D) repeat roundtrips to catch stale handles / reuse bugs
        test_multiple_roundtrips_reuse_ok(lib, ctxH, const_path, hevm_path, rounds=3)

        # cleanup ctx handle
        lib.freeContext(ctxH)

    print("ALL PASS")


if __name__ == "__main__":
    main()
