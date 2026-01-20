# test_hevm_lattigo.py
import ctypes
import os
import struct
import tempfile

SO = "./libLATTIGO_HEVM.so"

# ---- build a minimal constants.bin (SEAL format) ----
def write_constants(path):
    # int64 len = 1
    # int64 veclen = 1
    # double[1] = 0.25
    with open(path, "wb") as f:
        f.write(struct.pack("<q", 1))
        f.write(struct.pack("<q", 1))
        f.write(struct.pack("<d", 0.25))

# ---- build a minimal hevm.bin per your HEVMHeader.h layout ----
def write_hevm(path):
    magic = 0x4845564D
    hevm_header_size = 24  # 8 + sizeof(ConfigHeader(16))
    arg_len = 1
    res_len = 1

    # ConfigBody: 6*u64
    config_body_length = 48
    num_operations = 0
    num_ctxt_buffer = 8   # 给大一点，避免你 toy runtime 里 resDst 指到较大 idx 时越界
    num_ptxt_buffer = 4
    init_level = 0
    reserved = 0

    # arrays (len=arg_len/res_len)
    arg_scale = [40]
    arg_level = [0]
    res_scale = [40]
    res_level = [0]

    # 关键：你的 Go runtime decrypt_result 会按 res_dst[resIdx] 取 ciphertext id
    # 让 res 指向 arg0(cipher id = 0)，即可最小测试 decrypt_result == input
    res_dst = [0]

    with open(path, "wb") as f:
        # HEVMHeader
        f.write(struct.pack("<IIQQ", magic, hevm_header_size, arg_len, res_len))
        # ConfigBody (6*u64)
        f.write(struct.pack("<QQQQQQ",
                            config_body_length,
                            num_operations,
                            num_ctxt_buffer,
                            num_ptxt_buffer,
                            init_level,
                            reserved))
        # 5 arrays
        for arr in (arg_scale, arg_level, res_scale, res_level, res_dst):
            for x in arr:
                f.write(struct.pack("<Q", x))
        # ops: none (num_operations=0)

def main():
    lib = ctypes.CDLL(SO)

    # --- signatures ---
    lib.initFullVM.argtypes = [ctypes.c_char_p, ctypes.c_bool]
    lib.initFullVM.restype = ctypes.c_void_p

    lib.load.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
    lib.preprocess.argtypes = [ctypes.c_void_p]
    lib.run.argtypes = [ctypes.c_void_p]

    lib.getArgLen.argtypes = [ctypes.c_void_p]
    lib.getArgLen.restype = ctypes.c_int64
    lib.getResLen.argtypes = [ctypes.c_void_p]
    lib.getResLen.restype = ctypes.c_int64

    lib.encrypt.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    lib.decrypt_result.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.POINTER(ctypes.c_double)]

    with tempfile.TemporaryDirectory() as d:
        const_path = os.path.join(d, "constants.bin")
        hevm_path = os.path.join(d, "hevm.bin")
        write_constants(const_path)
        write_hevm(hevm_path)

        vm = lib.initFullVM(b".", False)
        lib.load(vm, const_path.encode(), hevm_path.encode())

        print("ArgLen =", lib.getArgLen(vm))
        print("ResLen =", lib.getResLen(vm))

        # encrypt arg0 = [0.25]
        x = (ctypes.c_double * 1)(0.25)
        lib.encrypt(vm, 0, x, 1)

        lib.preprocess(vm)
        lib.run(vm)

        # 关键修复：decrypt_result 会写 1<<14 个 double，必须分配足够大的 buffer
        OUTN = 1 << 14
        outbuf = (ctypes.c_double * OUTN)()
        lib.decrypt_result(vm, 0, outbuf)

        print("decrypt_result[0] =", outbuf[0])
        # 如果想验证它确实写满了（toy runtime 会复制槽位）
        # print("first 8 =", list(outbuf[:8]))

if __name__ == "__main__":
    main()
