// ckks_ops.go (完整文件：在你现有版本上补齐“密文 level/scale 查询”导出)
// 注意：下面只展示与你当前接口一致的一个完整可编译版本骨架。
// 你已有的 BootEnable / BootstrapTo 逻辑保留；这里重点新增：CKKS_CTLevel / CKKS_CTLog2Scale。
// 若你文件里已存在同名函数，请以此为准合并。

package main

/*
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
*/
import "C"

import (
	"fmt"
	"math"
	"runtime/cgo"
	"unsafe"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"

	// bootstrapping
	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
)

type VM struct {
	params ckks.Parameters
	slots  int

	encoder   *ckks.Encoder
	evaluator *ckks.Evaluator

	encryptor *rlwe.Encryptor
	decryptor *rlwe.Decryptor

	sk  *rlwe.SecretKey
	pk  *rlwe.PublicKey
	rlk *rlwe.RelinearizationKey
	gks *rlwe.GaloisKey

	ct []rlwe.Ciphertext

	// --- boot state ---
	bootEnabled bool
	bootParamID int // optional: just for prints
	btpParams   bootstrapping.Parameters
	btpKeys     *bootstrapping.EvaluationKeys
	btpEval     *bootstrapping.Evaluator
}

func getVM(h C.uintptr_t) *VM {
	if h == 0 {
		return nil
	}
	hd := cgo.Handle(h)
	v, ok := hd.Value().(*VM)
	if !ok {
		return nil
	}
	return v
}

func (v *VM) ensureCT(i int) {
	if i < 0 {
		return
	}
	if len(v.ct) <= i {
		newCT := make([]rlwe.Ciphertext, i+1)
		copy(newCT, v.ct)
		v.ct = newCT
	}
	if v.ct[i].Value == nil {
		tmp := ckks.NewCiphertext(v.params, 1, v.params.MaxLevel())
		v.ct[i] = *tmp
	}
}

func (v *VM) ctAt(i int) *rlwe.Ciphertext {
	if i < 0 || i >= len(v.ct) {
		return nil
	}
	if v.ct[i].Value == nil {
		v.ensureCT(i)
	}
	return &v.ct[i]
}

//export CKKS_CreateVM
func CKKS_CreateVM(logN C.int, levels C.int, logDefaultScale C.int, ctCapacity C.int) C.uintptr_t {
	ln := int(logN)
	lv := int(levels)
	lds := int(logDefaultScale)
	capN := int(ctCapacity)

	if ln < 10 {
		ln = 10
	}
	if lv < 3 {
		lv = 3
	}
	if lds < 20 {
		lds = 20
	}
	if capN < 8 {
		capN = 8
	}

	logQ := make([]int, lv)
	for i := range logQ {
		logQ[i] = 60
	}
	logP := []int{60, 60}

	pl := ckks.ParametersLiteral{
		LogN:            ln,
		LogQ:            logQ,
		LogP:            logP,
		LogDefaultScale: lds,
	}

	params, err := ckks.NewParametersFromLiteral(pl)
	if err != nil {
		fmt.Printf("[CKKS][ERR] NewParametersFromLiteral: %v\n", err)
		return 0
	}

	vm := &VM{}
	vm.params = params
	vm.slots = 1 << (ln - 1)

	const encPrec = uint(53)
	vm.encoder = ckks.NewEncoder(params, encPrec)

	kgen := ckks.NewKeyGenerator(params)
	sk, pk := kgen.GenKeyPairNew()
	vm.sk, vm.pk = sk, pk
	vm.rlk = kgen.GenRelinearizationKeyNew(sk)

	vm.encryptor = rlwe.NewEncryptor(params, pk)
	vm.decryptor = rlwe.NewDecryptor(params, sk)

	evk := rlwe.NewMemEvaluationKeySet(vm.rlk)
	vm.evaluator = ckks.NewEvaluator(params, evk)

	vm.ct = make([]rlwe.Ciphertext, capN)
	for i := 0; i < capN; i++ {
		tmp := ckks.NewCiphertext(params, 1, params.MaxLevel())
		vm.ct[i] = *tmp
	}

	h := cgo.NewHandle(vm)
	fmt.Printf("[CKKS] CreateVM handle=%d logN=%d slots=%d levels=%d logScale=%d ctCap=%d\n",
		uintptr(h), ln, vm.slots, lv, lds, capN)
	return C.uintptr_t(h)
}

//export CKKS_FreeVM
func CKKS_FreeVM(h C.uintptr_t) {
	if h == 0 {
		return
	}
	cgo.Handle(h).Delete()
}

//export CKKS_Slots
func CKKS_Slots(h C.uintptr_t) C.int {
	v := getVM(h)
	if v == nil {
		return 0
	}
	return C.int(v.slots)
}

//export CKKS_LogN
func CKKS_LogN(h C.uintptr_t) C.int {
	v := getVM(h)
	if v == nil {
		return 0
	}
	return C.int(v.params.LogN())
}

//export CKKS_MaxLevel
func CKKS_MaxLevel(h C.uintptr_t) C.int {
	v := getVM(h)
	if v == nil {
		return 0
	}
	return C.int(v.params.MaxLevel())
}

//export CKKS_LogDefaultScale
func CKKS_LogDefaultScale(h C.uintptr_t) C.int {
	v := getVM(h)
	if v == nil {
		return 0
	}
	return C.int(v.params.LogDefaultScale())
}

// -------- NEW: ciphertext metadata exports --------

//export CKKS_CTLevel
func CKKS_CTLevel(h C.uintptr_t, idx C.int) C.int {
	v := getVM(h)
	if v == nil {
		return -1
	}
	ct := v.ctAt(int(idx))
	if ct == nil || ct.Value == nil {
		return -1
	}
	return C.int(ct.Level())
}

//export CKKS_CTLog2Scale
func CKKS_CTLog2Scale(h C.uintptr_t, idx C.int) C.int {
	v := getVM(h)
	if v == nil {
		return 0
	}
	ct := v.ctAt(int(idx))
	if ct == nil || ct.Value == nil {
		return 0
	}
	// use float64 approximation for logging
	return C.int(math.Round(math.Log2(ct.Scale.Float64())))
}

//export CKKS_EncryptTo
func CKKS_EncryptTo(h C.uintptr_t, dst C.int, data *C.double, n C.int, level C.int, log2Scale C.int) {
	v := getVM(h)
	if v == nil || data == nil {
		return
	}

	d := int(dst)
	ln := int(n)
	lvl := int(level)
	ls := int(log2Scale)

	v.ensureCT(d)

	if lvl < 0 || lvl > v.params.MaxLevel() {
		lvl = v.params.MaxLevel()
	}
	if ls <= 0 {
		ls = int(v.params.LogDefaultScale())
	}

	src := (*[1 << 30]C.double)(unsafe.Pointer(data))[:ln:ln]
	vec := make([]complex128, v.slots)
	for i := 0; i < v.slots; i++ {
		vec[i] = complex(float64(src[i%ln]), 0)
	}

	pt := ckks.NewPlaintext(v.params, lvl)
	pt.Scale = rlwe.NewScale(math.Exp2(float64(ls)))
	v.encoder.Encode(vec, pt)

	ct := ckks.NewCiphertext(v.params, 1, lvl)
	v.encryptor.Encrypt(pt, ct)

	v.ct[d] = *ct
}

//export CKKS_DecryptFrom
func CKKS_DecryptFrom(h C.uintptr_t, src C.int, out *C.double, outN C.int) {
	v := getVM(h)
	if v == nil || out == nil {
		return
	}
	s := int(src)
	ct := v.ctAt(s)
	if ct == nil {
		return
	}

	pt := ckks.NewPlaintext(v.params, ct.Level())
	v.decryptor.Decrypt(ct, pt)

	vec := make([]complex128, v.slots)
	v.encoder.Decode(pt, vec)

	n := int(outN)
	if n <= 0 {
		return
	}
	if n > v.slots {
		n = v.slots
	}

	outArr := (*[1 << 30]C.double)(unsafe.Pointer(out))[:n:n]
	for i := 0; i < n; i++ {
		outArr[i] = C.double(real(vec[i]))
	}
}

//export CKKS_OpMulCC
func CKKS_OpMulCC(h C.uintptr_t, dst, a, b C.int) {
	v := getVM(h)
	if v == nil {
		return
	}
	v.ensureCT(int(dst))
	v.evaluator.MulRelin(v.ctAt(int(a)), v.ctAt(int(b)), v.ctAt(int(dst)))
}

//export CKKS_OpRescale
func CKKS_OpRescale(h C.uintptr_t, dst, a C.int) {
	v := getVM(h)
	if v == nil {
		return
	}
	v.ensureCT(int(dst))
	v.evaluator.Rescale(v.ctAt(int(a)), v.ctAt(int(dst)))
}

// ---------------- boot API (你已有的实现；这里只给一个可工作形态) ----------------

//export CKKS_BootEnable
func CKKS_BootEnable(h C.uintptr_t) C.int {
	v := getVM(h)
	if v == nil {
		return 0
	}
	if v.bootEnabled {
		return 1
	}

	// 强制使用你指定的默认参数：N16QP1546H192H32
	// 这段 literal 必须与 lattigo circuits/ckks/bootstrapping 的默认参数一致（你已经贴了那段）
	residual := ckks.ParametersLiteral{
		LogN:            16,
		LogQ:            []int{60, 40, 40, 40, 40, 40, 40, 40, 40, 40},
		LogP:            []int{61, 61, 61, 61, 61},
		Xs:              ring.Ternary{H: 192},
		LogDefaultScale: 40,
	}
	params, err := ckks.NewParametersFromLiteral(residual)
	if err != nil {
		fmt.Printf("[CKKS][BOOT][ERR] NewParametersFromLiteral(residual): %v\n", err)
		return 0
	}

	// build boot params from literal defaults (ParametersLiteral{} means default boot settings for this residual set)
	btpParams, err := bootstrapping.NewParametersFromLiteral(params, bootstrapping.ParametersLiteral{})
	if err != nil {
		fmt.Printf("[CKKS][BOOT][ERR] NewParametersFromLiteral(boot): %v\n", err)
		return 0
	}

	// regen keys under new params
	kgen := rlwe.NewKeyGenerator(btpParams.BootstrappingParameters)
	sk := kgen.GenSecretKeyNew()

	// boot keys (includes relinearization + galois + switching keys)
	btpKeys, _, err := btpParams.GenEvaluationKeys(sk)
	if err != nil {
		fmt.Printf("[CKKS][BOOT][ERR] GenEvaluationKeys: %v\n", err)
		return 0
	}
	eval, err := bootstrapping.NewEvaluator(btpParams, btpKeys)
	if err != nil {
		fmt.Printf("[CKKS][BOOT][ERR] NewEvaluator: %v\n", err)
		return 0
	}

	// rebuild VM state
	v.params = params
	v.slots = 1 << (params.LogN() - 1)
	v.sk = sk
	v.pk = kgen.GenPublicKeyNew(sk)
	v.rlk = kgen.GenRelinearizationKeyNew(sk)

	v.encoder = ckks.NewEncoder(params, 53)
	v.encryptor = rlwe.NewEncryptor(params, v.pk)
	v.decryptor = rlwe.NewDecryptor(params, v.sk)

	// evaluator needs rlk at least (for MulRelin outside boot)
	evk := rlwe.NewMemEvaluationKeySet(v.rlk)
	v.evaluator = ckks.NewEvaluator(params, evk)

	v.btpParams = btpParams
	v.btpKeys = btpKeys
	v.btpEval = eval
	v.bootEnabled = true

	// re-init ciphertext buffer to new params
	for i := range v.ct {
		tmp := ckks.NewCiphertext(params, 1, params.MaxLevel())
		v.ct[i] = *tmp
	}

	fmt.Printf("[CKKS][BOOT] enabled PARAM=N16QP1546H192H32 logN=%d slots=%d maxLevel=%d logScale=%d\n",
		params.LogN(), v.slots, params.MaxLevel(), params.LogDefaultScale())

	return 1
}

//export CKKS_BootstrapTo
func CKKS_BootstrapTo(h C.uintptr_t, dst, src C.int) C.int {
	v := getVM(h)
	if v == nil || !v.bootEnabled || v.btpEval == nil {
		return 0
	}
	v.ensureCT(int(dst))
	in := v.ctAt(int(src))
	out := v.ctAt(int(dst))
	if in == nil || out == nil {
		return 0
	}

	// IMPORTANT: Boot evaluator expects ciphertext over v.params (residual scheme params)
	// It also generally expects input at level 0 (depending on implementation).
	// The boot evaluator in Lattigo handles level normalization internally, but to be safe you can:
	// - drop to level 0 before boot if needed.
	// Here we directly call Bootstrap; your earlier logs show it works with your flow.

	ct, err := v.btpEval.Bootstrap(in)
	if err != nil {
		fmt.Printf("[CKKS][BOOT][ERR] Bootstrap: %v\n", err)
		return 0
	}
	*out = *ct
	return 1
}

func main() {}
