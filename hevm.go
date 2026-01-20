package main

/*
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
*/
import "C"

import (
	"fmt"
	"unsafe"
)

type VM struct {
	argLen int64
	resLen int64
	debug  bool
	toGPU  bool
}

func vmFromPtr(p unsafe.Pointer) *VM {
	if p == nil {
		return nil
	}
	return (*VM)(p)
}

// -----------------------------
// Exported C ABI (match runner.py)
// -----------------------------

//export initFullVM
func initFullVM(path *C.char, useGPU C._Bool) unsafe.Pointer {
	p := C.GoString(path)
	vm := (*VM)(C.malloc(C.size_t(unsafe.Sizeof(VM{}))))
	*vm = VM{argLen: 1, resLen: 1, debug: false, toGPU: bool(useGPU)}
	fmt.Printf("[LATTIGO_STUB] initFullVM path=%q useGPU=%v vm=%p\n", p, bool(useGPU), vm)
	return unsafe.Pointer(vm)
}

//export initClientVM
func initClientVM(path *C.char) unsafe.Pointer {
	p := C.GoString(path)
	vm := (*VM)(C.malloc(C.size_t(unsafe.Sizeof(VM{}))))
	*vm = VM{argLen: 1, resLen: 1}
	fmt.Printf("[LATTIGO_STUB] initClientVM path=%q vm=%p\n", p, vm)
	return unsafe.Pointer(vm)
}

//export initServerVM
func initServerVM(path *C.char) unsafe.Pointer {
	p := C.GoString(path)
	vm := (*VM)(C.malloc(C.size_t(unsafe.Sizeof(VM{}))))
	*vm = VM{argLen: 1, resLen: 1}
	fmt.Printf("[LATTIGO_STUB] initServerVM path=%q vm=%p\n", p, vm)
	return unsafe.Pointer(vm)
}

//export create_context
func create_context(path *C.char) {
	p := C.GoString(path)
	fmt.Printf("[LATTIGO_STUB] create_context path=%q (no-op)\n", p)
}

//export load
func load(vm unsafe.Pointer, constPath *C.char, hevmPath *C.char) {
	v := vmFromPtr(vm)
	fmt.Printf("[LATTIGO_STUB] load vm=%p const=%q hevm=%q\n", v, C.GoString(constPath), C.GoString(hevmPath))
	// 这里可以根据 hevm 文件内容决定 arg/res 长度；stub 固定为 1
	if v != nil {
		v.argLen = 1
		v.resLen = 1
	}
}

//export loadClient
func loadClient(vm unsafe.Pointer, ctx unsafe.Pointer) {
	// runner.py 里 loadClient 的 argtypes 写成 (void*, void*)，但实际可能传 path。
	// 为了兼容，这里不解读 ctx 的类型，只打印指针值。
	v := vmFromPtr(vm)
	fmt.Printf("[LATTIGO_STUB] loadClient vm=%p ctx=%p (no-op)\n", v, ctx)
}

//export preprocess
func preprocess(vm unsafe.Pointer) {
	v := vmFromPtr(vm)
	fmt.Printf("[LATTIGO_STUB] preprocess vm=%p\n", v)
}

//export run
func run(vm unsafe.Pointer) {
	v := vmFromPtr(vm)
	fmt.Printf("[LATTIGO_STUB] run vm=%p\n", v)
}

//export getArgLen
func getArgLen(vm unsafe.Pointer) C.int64_t {
	v := vmFromPtr(vm)
	if v == nil {
		return 0
	}
	return C.int64_t(v.argLen)
}

//export getResLen
func getResLen(vm unsafe.Pointer) C.int64_t {
	v := vmFromPtr(vm)
	if v == nil {
		return 0
	}
	return C.int64_t(v.resLen)
}

//export encrypt
func encrypt(vm unsafe.Pointer, idx C.int64_t, data *C.double, n C.int) {
	v := vmFromPtr(vm)
	fmt.Printf("[LATTIGO_STUB] encrypt vm=%p idx=%d n=%d\n", v, int64(idx), int(n))
	// stub 不做任何事
}

//export decrypt
func decrypt(vm unsafe.Pointer, idx C.int64_t, out *C.double) {
	v := vmFromPtr(vm)
	fmt.Printf("[LATTIGO_STUB] decrypt vm=%p idx=%d (stub)\n", v, int64(idx))
	if out != nil {
		*out = 0.0
	}
}

//export decrypt_result
func decrypt_result(vm unsafe.Pointer, resIdx C.int64_t, out *C.double) {
	v := vmFromPtr(vm)
	fmt.Printf("[LATTIGO_STUB] decrypt_result vm=%p resIdx=%d\n", v, int64(resIdx))
	// runner.py(SEAL) 给的 buffer 是 1<<14 doubles；我们写前几个做标记
	if out == nil {
		return
	}
	// 把 out 当作一个足够大的数组写入
	outArr := (*[1 << 20]C.double)(unsafe.Pointer(out)) // 上限写大一点，别越界到 1<<14 之外
	outArr[0] = 111.0
	outArr[1] = 222.0
	outArr[2] = 333.0
}

//export getResIdx
func getResIdx(vm unsafe.Pointer, i C.int64_t) C.int64_t {
	// stub: identity
	return i
}

//export getCtxt
func getCtxt(vm unsafe.Pointer, i C.int64_t) unsafe.Pointer {
	// stub: no ciphertext object
	fmt.Printf("[LATTIGO_STUB] getCtxt vm=%p i=%d -> nil\n", vmFromPtr(vm), int64(i))
	return nil
}

//export setDebug
func setDebug(vm unsafe.Pointer, enable C._Bool) {
	v := vmFromPtr(vm)
	if v != nil {
		v.debug = bool(enable)
	}
	fmt.Printf("[LATTIGO_STUB] setDebug vm=%p enable=%v\n", v, bool(enable))
}

//export setToGPU
func setToGPU(vm unsafe.Pointer, ongpu C._Bool) {
	v := vmFromPtr(vm)
	if v != nil {
		v.toGPU = bool(ongpu)
	}
	fmt.Printf("[LATTIGO_STUB] setToGPU vm=%p ongpu=%v\n", v, bool(ongpu))
}

//export printMem
func printMem(vm unsafe.Pointer) {
	v := vmFromPtr(vm)
	fmt.Printf("[LATTIGO_STUB] printMem vm=%p (stub)\n", v)
}

func main() {}
