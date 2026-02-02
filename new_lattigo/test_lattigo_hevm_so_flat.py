#!/usr/bin/env python3
# test_lattigo_hevm_so_flat.py
#
# Expect all artifacts in the SAME directory as this script:
#   ./libLATTIGO_HEVM.so
#   ./_hecate_ResNet.cst
#   ./ResNet.40._hecate_ResNet.hevm
#   (optional) ./profiled_LATTIGO_CPU.json
#
# It will:
# - ensure ctx dir exists (default: ~/.hevm/lattigo)
# - create_context(ctx) if ctx looks empty
# - set env LATTIGO_CTX_DIR=ctx (for auto-rotkey extension inside .so)
# - initFullVM(ctx, useGPU)
# - load(vm, cst, hevm)
# - preprocess(vm)
# - encrypt arg0 with dummy vector (if arglen>0)
# - run(vm)
# - decrypt_result(res0) and print head
#
# Usage:
#   python3 test_lattigo_hevm_so_flat.py --stem ResNet --compile_opt 40 --cpu
#   python3 test_lattigo_hevm_so_flat.py --cst _hecate_ResNet.cst --hevm ResNet.40._hecate_ResNet.hevm
#   python3 test_lattigo_hevm_so_flat.py --ctx ./ctx_local   (use a local ctx dir)

import argparse
import ctypes
import os
from pathlib import Path

import numpy as np


def bind_api(lw: ctypes.CDLL):
    # Context
    lw.create_context.argtypes = [ctypes.c_char_p]
    lw.create_context.restype = None

    # Init VMs
    lw.initFullVM.argtypes = [ctypes.c_char_p, ctypes.c_bool]
    lw.initFullVM.restype = ctypes.c_void_p

    lw.initClientVM.argtypes = [ctypes.c_char_p]
    lw.initClientVM.restype = ctypes.c_void_p

    lw.initServerVM.argtypes = [ctypes.c_char_p]
    lw.initServerVM.restype = ctypes.c_void_p

    # Load / preprocess / run
    lw.load.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
    lw.load.restype = None

    if hasattr(lw, "preprocess"):
        lw.preprocess.argtypes = [ctypes.c_void_p]
        lw.preprocess.restype = None

    if hasattr(lw, "run"):
        lw.run.argtypes = [ctypes.c_void_p]
        lw.run.restype = None

    # Arg/Res sizes
    lw.getArgLen.argtypes = [ctypes.c_void_p]
    lw.getArgLen.restype = ctypes.c_int64

    lw.getResLen.argtypes = [ctypes.c_void_p]
    lw.getResLen.restype = ctypes.c_int64

    # Encrypt/Decrypt
    lw.encrypt.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
    ]
    lw.encrypt.restype = None

    # Some .so may not export decrypt(); decrypt_result is used by your runner anyway.
    if hasattr(lw, "decrypt"):
        lw.decrypt.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_double),
        ]
        lw.decrypt.restype = None

    lw.decrypt_result.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
    ]
    lw.decrypt_result.restype = None

    # Optional helpers
    if hasattr(lw, "Slots"):
        lw.Slots.argtypes = [ctypes.c_void_p]
        lw.Slots.restype = ctypes.c_int

    if hasattr(lw, "printMem"):
        lw.printMem.argtypes = [ctypes.c_void_p]
        lw.printMem.restype = None

    if hasattr(lw, "freeVM"):
        lw.freeVM.argtypes = [ctypes.c_void_p]
        lw.freeVM.restype = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="ResNet")
    ap.add_argument("--compile_opt", default="40")
    ap.add_argument("--cst", default=None, help="default: ./_hecate_{stem}.cst")
    ap.add_argument("--hevm", default=None, help="default: ./{stem}.{opt}._hecate_{stem}.hevm")
    ap.add_argument("--ctx", default=None, help="default: ~/.hevm/lattigo")
    ap.add_argument("--cpu", action="store_true", help="initFullVM(..., False)")
    ap.add_argument("--gpu", action="store_true", help="initFullVM(..., True)")
    ap.add_argument("--no-preprocess", action="store_true")
    ap.add_argument("--no-run", action="store_true")
    ap.add_argument("--dummy-len", type=int, default=4, help="dummy input length for encrypt(arg0)")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent

    so_path = here / "libLATTIGO_HEVM.so"
    if not so_path.is_file():
        raise FileNotFoundError(f"Missing {so_path}")

    cst_path = Path(args.cst) if args.cst else (here / f"_hecate_{args.stem}.cst")
    hevm_path = Path(args.hevm) if args.hevm else (here / f"{args.stem}.{args.compile_opt}._hecate_{args.stem}.hevm")

    if not cst_path.is_file():
        raise FileNotFoundError(f"Missing constants: {cst_path} (expected in {here})")
    if not hevm_path.is_file():
        raise FileNotFoundError(f"Missing hevm: {hevm_path} (expected in {here})")

    # ctx dir
    ctx_dir = Path(args.ctx) if args.ctx else (Path.home() / ".hevm" / "lattigo")
    ctx_dir.mkdir(parents=True, exist_ok=True)

    # IMPORTANT for your .so: load() may auto-generate missing rot keys and write gal.lattigo.
    # The stub uses env LATTIGO_CTX_DIR to know where to write.
    os.environ["LATTIGO_CTX_DIR"] = str(ctx_dir)

    # Load .so
    lw = ctypes.CDLL(str(so_path))
    bind_api(lw)

    # If ctx is empty-ish, create it
    is_empty = True
    try:
        is_empty = not any(ctx_dir.iterdir())
    except Exception:
        is_empty = True

    if is_empty:
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

    # load
    print(f"[TEST] load(cst={cst_path.name}, hevm={hevm_path.name})")
    lw.load(vm, str(cst_path).encode("utf-8"), str(hevm_path).encode("utf-8"))

    # preprocess
    if not args.no_preprocess and hasattr(lw, "preprocess"):
        print("[TEST] preprocess()")
        lw.preprocess(vm)
    else:
        print("[TEST] skip preprocess")

    arglen = int(lw.getArgLen(vm))
    reslen = int(lw.getResLen(vm))
    print(f"[TEST] arglen={arglen}, reslen={reslen}")

    # encrypt arg0 with dummy data
    if arglen > 0:
        dlen = max(1, int(args.dummy_len))
        x = np.linspace(0.1, 0.1 * dlen, dlen, dtype=np.float64)
        lw.encrypt(vm, 0, x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), x.size)
        print(f"[TEST] encrypt(arg0) ok, dummy_len={dlen}")
    else:
        print("[TEST] arglen=0 -> skip encrypt")

    # run
    if not args.no_run and hasattr(lw, "run"):
        print("[TEST] run()")
        lw.run(vm)
        if hasattr(lw, "printMem"):
            lw.printMem(vm)
    else:
        print("[TEST] skip run")

    # decrypt result 0
    if reslen > 0:
        # default N=15 => slots=2^(N-1)=16384
        slots = 16384
        if hasattr(lw, "Slots"):
            try:
                slots = int(lw.Slots(vm))
            except Exception:
                pass

        out = np.zeros(slots, dtype=np.float64)
        lw.decrypt_result(vm, 0, out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        print("[TEST] decrypt_result(res0) head:", out[:16])
    else:
        print("[TEST] reslen=0 -> skip decrypt_result")

    # free
    if hasattr(lw, "freeVM"):
        lw.freeVM(vm)

    print("[TEST] OK")


if __name__ == "__main__":
    main()
