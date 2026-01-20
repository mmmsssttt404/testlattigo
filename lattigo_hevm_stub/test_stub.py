import ctypes
import numpy as np
from pathlib import Path

so = ctypes.CDLL(str(Path("./libLATTIGO_HEVM.so").absolute()))

so.initFullVM.argtypes = [ctypes.c_char_p, ctypes.c_bool]
so.initFullVM.restype = ctypes.c_void_p

so.load.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
so.preprocess.argtypes = [ctypes.c_void_p]
so.run.argtypes = [ctypes.c_void_p]

so.getArgLen.argtypes = [ctypes.c_void_p]
so.getArgLen.restype = ctypes.c_int64
so.getResLen.argtypes = [ctypes.c_void_p]
so.getResLen.restype = ctypes.c_int64

so.encrypt.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
so.decrypt_result.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.POINTER(ctypes.c_double)]

vm = so.initFullVM(b"/tmp/hevm_stub", False)
so.load(vm, b"/tmp/x.cst", b"/tmp/y.hevm")   # stub 不检查文件存在
so.preprocess(vm)

arglen = so.getArgLen(vm)
reslen = so.getResLen(vm)
print("arglen=", arglen, "reslen=", reslen)

inp = np.array([1.0, 2.0, 3.0], dtype=np.float64)
so.encrypt(vm, 0, inp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), inp.size)

out = np.zeros(1<<14, dtype=np.float64)
so.run(vm)
so.decrypt_result(vm, 0, out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

print("out[0:4] =", out[:4])
