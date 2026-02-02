#!/usr/bin/env python3
# test_lattigo_hevm_so_flat.py
#
# Folder layout (ALL in the same directory):
#   libLATTIGO_HEVM.so
#   _hecate_ResNet.cst
#   ResNet.40._hecate_ResNet.hevm
#   (optional) profiled_LATTIGO_CPU.json
#
# Usage:
#   python3 test_lattigo_hevm_so_flat.py --compile_opt 40 --stem ResNet
#   python3 test_lattigo_hevm_so_flat.py --hevm ResNet.40._hecate_ResNet.hevm --cst _hecate_ResNet.cst
#
# It will:
# - create context dir (~/.hevm/lattigo) if missing via create_context(dir)
# - initFullVM(dir, False)
# - load(vm, cst, hevm)
# - preprocess(vm)
# - encrypt arg0 with a small dummy vector (if arglen>0)
# - run(vm)
# - decrypt_result(res0) and print first few numbers (if reslen>0)

import argparse
import ctypes
from pathlib import Path
import numpy as np


def bind_api(lw: ctypes.CDLL):
    # Init VM functions
    lw.initFullVM.argtypes = [ctypes.c_char_p, ctypes.c_bool]
    lw.initFullVM.restype = ctypes.c_void_p
    lw.initClientVM.argtypes = [ctypes.c_char_p]
    lw.initClientVM.restype = ctypes.c_void_p
    lw.initServerVM.argtypes = [ctypes.c_char_p]
    lw.initServerVM.restype = ctypes.c_void_p

    # Context
    lw.create_context.argtypes = [ctypes.c_char_p]

    # Loader
    lw.load.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
    lw.loadClient.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lw.getArgLen.argtypes = [ctypes.c_void_p]
    lw.getArgLen.restype = ctypes.c_int64
    lw.getResLen.argtypes = [ctypes.c_void_p]
    lw.getResLen.restype = ctypes.c_int64

    # Encrypt/Decrypt
    lw.encrypt.argtypes = [ctypes.c_void_p, ctypes.c_int64,
                           ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    lw.decrypt.argtypes = [ctypes.c_void_p, ctypes.c_int64,
                           ctypes.POINTER(ctypes.c_double)]
    lw.decrypt_result.argtypes = [ctypes.c_void_p, ctypes.c_int64,
                                  ctypes.POINTER(ctypes.c_double)]

    # Helpers (optional)
    if hasattr(lw, "getResIdx"):
        lw.getResIdx.argtypes = [ctypes.c_void_p, ctypes.c_int64]
        lw.getResIdx.restype = ctypes.c_int64
    if hasattr(lw, "getCtxt"):
        lw.getCtxt.argtypes = [ctypes.c_void_p, ctypes.c_int64]
        lw.getCtxt.restype = ctypes.c_void_p

    # Runner
    lw.preprocess.argtypes = [ctypes.c_void_p]
    lw.run.argtypes = [ctypes.c_void_p]

    # Debug / GPU / mem
    if hasattr(lw, "setDebug"):
        lw.setDebug.argtypes = [ctypes.c_void_p, ctypes.c_bool]
    if hasattr(lw, "setToGPU"):
        lw.setToGPU.argtypes = [ctypes.c_void_p, ctypes.c_bool]
    if hasattr(lw, "printMem"):
        lw.printMem.argtypes = [ctypes.c_void_p]

    # Optional slots query
    if hasattr(lw, "Slots"):
        lw.Slots.argtypes = [ctypes.c_void_p]
        lw.Slots.restype = ctypes.c_int

    # Optional freeVM
    if hasattr(lw, "freeVM"):
        lw.freeVM.argtypes = [ctypes.c_void_p]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="ResNet", help="Benchmark stem, used to infer filenames")
    ap.add_argument("--compile_opt", default="40", help="Waterline/opt number used in hevm filename, e.g. 40")
    ap.add_argument("--cst", default=None, help="Constants file path (default inferred from --stem)")
    ap.add_argument("--hevm", default=None, help="HEVM file path (default inferred from --stem/--compile_opt)")
    ap.add_argument("--ctx", default=None, help="Context dir (default: ~/.hevm/lattigo)")
    ap.add_argument("--cpu", action="store_true", help="Force CPU (initFullVM(..., False))")
    ap.add_argument("--gpu", action="store_true", help="Force GPU (initFullVM(..., True))")
    ap.add_argument("--no-run", action="store_true", help="Skip run()")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    so_path = here / "libLATTIGO_HEVM.so"
    if not so_path.is_file():
        raise FileNotFoundError(f"Missing {so_path}")

    cst_path = Path(args.cst) if args.cst else (here / f"_hecate_{args.stem}.cst")
    hevm_path = Path(args.hevm) if args.hevm else (here / f"{args.stem}.{args.compile_opt}._hecate_{args.stem}.hevm")

    if not cst_path.is_file():
        raise FileNotFoundError(f"Missing constants: {cst_path}")
    if not hevm_path.is_file():
        raise FileNotFoundError(f"Missing hevm: {hevm_path}")

    ctx_dir = Path(args.ctx) if args.ctx else (Path.home() / ".hevm" / "lattigo")
    ctx_dir.mkdir(parents=True, exist_ok=True)

    # Load .so from this folder
    lw = ctypes.CDLL(str(so_path))
    bind_api(lw)

    # If ctx dir looks empty, call create_context
    # (simple heuristic: if no files inside, generate)
    if not any(ctx_dir.iterdir()):
        print(f"[TEST] ctx empty -> create_context({ctx_dir})")
        lw.create_context(str(ctx_dir).encode("utf-8"))

    # initFullVM
    use_gpu = False
    if args.gpu:
        use_gpu = True
    if args.cpu:
        use_gpu = False

    print(f"[TEST] initFullVM({ctx_dir}, useGPU={use_gpu})")
    vm = lw.initFullVM(str(ctx_dir).encode("utf-8"), bool(use_gpu))
    if not vm:
        raise RuntimeError("initFullVM returned NULL")

    # load + preprocess
    print(f"[TEST] load(cst={cst_path.name}, hevm={hevm_path.name})")
    lw.load(vm, str(cst_path).encode("utf-8"), str(hevm_path).encode("utf-8"))

    print("[TEST] preprocess()")
    lw.preprocess(vm)

    arglen = int(lw.getArgLen(vm))
    reslen = int(lw.getResLen(vm))
    print(f"[TEST] arglen={arglen}, reslen={reslen}")

    # encrypt arg0 with dummy data if args exist
    if arglen > 0:
        x = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        lw.encrypt(vm, 0, x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), x.size)
        print("[TEST] encrypt(arg0) ok")
    else:
        print("[TEST] arglen=0 skip encrypt")

    # run
    if not args.no_run:
        print("[TEST] run()")
        lw.run(vm)
        if hasattr(lw, "printMem"):
            lw.printMem(vm)
    else:
        print("[TEST] --no-run skip run")

    # decrypt result 0
    if reslen > 0:
        slots = 1 << 14
        if hasattr(lw, "Slots"):
            try:
                slots = int(lw.Slots(vm))
            except Exception:
                pass

        out = np.zeros(slots, dtype=np.float64)
        lw.decrypt_result(vm, 0, out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        print("[TEST] decrypt_result(res0) head:", out[:16])
    else:
        print("[TEST] reslen=0 skip decrypt_result")

    # free
    if hasattr(lw, "freeVM"):
        lw.freeVM(vm)

    print("[TEST] OK")


if __name__ == "__main__":
    main()
