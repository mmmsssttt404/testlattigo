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

	gkSingle *rlwe.GaloisKey

	ct []rlwe.Ciphertext

	bootEnabled bool
	btpParams   *bootstrapping.Parameters
	bootEval    *bootstrapping.Evaluator
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

func (v *VM) resetWithParams(params ckks.Parameters, ctCapacity int) {
	v.params = params
	v.slots = params.MaxSlots()

	const encPrec = uint(53)
	v.encoder = ckks.NewEncoder(params, encPrec)

	kgen := rlwe.NewKeyGenerator(params)
	v.sk = kgen.GenSecretKeyNew()
	v.pk = kgen.GenPublicKeyNew(v.sk)
	v.rlk = kgen.GenRelinearizationKeyNew(v.sk)

	v.encryptor = rlwe.NewEncryptor(params, v.pk)
	v.decryptor = rlwe.NewDecryptor(params, v.sk)

	evk := rlwe.NewMemEvaluationKeySet(v.rlk)
	v.evaluator = ckks.NewEvaluator(params, evk)

	v.gkSingle = nil

	v.ct = make([]rlwe.Ciphertext, ctCapacity)
	for i := 0; i < ctCapacity; i++ {
		tmp := ckks.NewCiphertext(params, 1, params.MaxLevel())
		v.ct[i] = *tmp
	}
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
	vm.resetWithParams(params, capN)
	vm.bootEnabled = false
	vm.btpParams = nil
	vm.bootEval = nil

	h := cgo.NewHandle(vm)
	fmt.Printf("[CKKS] CreateVM handle=%d logN=%d slots=%d levels=%d logScale=%d ctCap=%d\n",
		uintptr(h), params.LogN(), params.MaxSlots(), params.MaxLevel()+1, params.LogDefaultScale(), capN)
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
		return -1
	}
	ct := v.ctAt(int(idx))
	if ct == nil || ct.Value == nil {
		return -1
	}
	ls := math.Log2(ct.Scale.Float64())
	return C.int(int(math.Round(ls)))
}

//export CKKS_GenRotKey
func CKKS_GenRotKey(h C.uintptr_t, rot C.int) {
	v := getVM(h)
	if v == nil {
		return
	}
	k := int(rot)
	if k == 0 {
		return
	}

	kgen := rlwe.NewKeyGenerator(v.params)
	galEl := v.params.GaloisElementForRotation(k)
	v.gkSingle = kgen.GenGaloisKeyNew(galEl, v.sk)

	evk := rlwe.NewMemEvaluationKeySet(v.rlk, v.gkSingle)
	v.evaluator = ckks.NewEvaluator(v.params, evk)
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
	ct := v.ctAt(int(src))
	if ct == nil || ct.Value == nil {
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
	ctA := v.ctAt(int(a))
	ctB := v.ctAt(int(b))
	ctD := v.ctAt(int(dst))
	if ctA == nil || ctB == nil || ctD == nil {
		return
	}
	if err := v.evaluator.MulRelin(ctA, ctB, ctD); err != nil {
		fmt.Printf("[CKKS][ERR] MulRelin: %v\n", err)
	}
}

//export CKKS_OpMulCP_Const
func CKKS_OpMulCP_Const(h C.uintptr_t, dst, a C.int, c C.double) {
	v := getVM(h)
	if v == nil {
		return
	}
	v.ensureCT(int(dst))

	in := v.ctAt(int(a))
	out := v.ctAt(int(dst))
	if in == nil || out == nil {
		return
	}

	pt := ckks.NewPlaintext(v.params, in.Level())
	pt.Scale = in.Scale

	vec := make([]complex128, v.slots)
	for i := 0; i < v.slots; i++ {
		vec[i] = complex(float64(c), 0)
	}
	v.encoder.Encode(vec, pt)

	if err := v.evaluator.Mul(in, pt, out); err != nil {
		fmt.Printf("[CKKS][ERR] Mul(ct,pt): %v\n", err)
	}
}

//export CKKS_OpRescale
func CKKS_OpRescale(h C.uintptr_t, dst, a C.int) {
	v := getVM(h)
	if v == nil {
		return
	}
	v.ensureCT(int(dst))
	if err := v.evaluator.Rescale(v.ctAt(int(a)), v.ctAt(int(dst))); err != nil {
		fmt.Printf("[CKKS][ERR] Rescale: %v\n", err)
	}
}

//export CKKS_OpDropLevel
func CKKS_OpDropLevel(h C.uintptr_t, dst, a, down C.int) {
	v := getVM(h)
	if v == nil {
		return
	}
	v.ensureCT(int(dst))
	*v.ctAt(int(dst)) = *v.ctAt(int(a))
	d := int(down)
	if d > 0 {
		// v6 DropLevel: no return value
		v.evaluator.DropLevel(v.ctAt(int(dst)), d)
	}
}

//
// -------- Boot --------
//

//export CKKS_BootEnable
func CKKS_BootEnable(h C.uintptr_t) C.int {
	v := getVM(h)
	if v == nil {
		return 0
	}

	ps := bootstrapping.N16QP1546H192H32

	// scheme params
	residualParams, err := ckks.NewParametersFromLiteral(ps.SchemeParams)
	if err != nil {
		fmt.Printf("[CKKS][BOOT][ERR] NewParametersFromLiteral(scheme): %v\n", err)
		return 0
	}

	// boot params (value type)
	btpParamsVal, err := bootstrapping.NewParametersFromLiteral(residualParams, ps.BootstrappingParams)
	if err != nil {
		fmt.Printf("[CKKS][BOOT][ERR] NewParametersFromLiteral(boot): %v\n", err)
		return 0
	}

	// IMPORTANT: use the SAME parameter domain for all ops:
	// Use BootstrappingParameters as your sole params.
	vmParams := btpParamsVal.BootstrappingParameters

	ctCap := len(v.ct)
	if ctCap <= 0 {
		ctCap = 32
	}

	// reset VM
	v.resetWithParams(vmParams, ctCap)

	// generate boot keys under the SAME sk (current sk is from vmParams)
	btpKeys, _, err := btpParamsVal.GenEvaluationKeys(v.sk)
	if err != nil {
		fmt.Printf("[CKKS][BOOT][ERR] GenEvaluationKeys: %v\n", err)
		return 0
	}

	bootEval, err := bootstrapping.NewEvaluator(btpParamsVal, btpKeys)
	if err != nil {
		fmt.Printf("[CKKS][BOOT][ERR] NewEvaluator: %v\n", err)
		return 0
	}

	v.bootEnabled = true
	// store pointer (take address)
	v.btpParams = &btpParamsVal
	v.bootEval = bootEval

	fmt.Printf("[CKKS][BOOT] enabled PARAM=N16QP1546H192H32 logN=%d slots=%d maxLevel=%d logScale=%d\n",
		v.params.LogN(), v.params.MaxSlots(), v.params.MaxLevel(), v.params.LogDefaultScale())

	return 1
}

//export CKKS_BootstrapTo
func CKKS_BootstrapTo(h C.uintptr_t, dst, src C.int) C.int {
	v := getVM(h)
	if v == nil || !v.bootEnabled || v.bootEval == nil {
		return 0
	}

	in := v.ctAt(int(src))
	if in == nil || in.Value == nil {
		return 0
	}

	v.ensureCT(int(dst))
	out := v.ctAt(int(dst))
	if out == nil {
		return 0
	}

	// Many default boot parameter sets expect level 0 input.
	tmp := *in
	if tmp.Level() > 0 {
		// DropLevel to 0 (no return value in v6)
		v.evaluator.DropLevel(&tmp, tmp.Level())
	}

	ctOut, err := v.bootEval.Bootstrap(&tmp)
	if err != nil {
		fmt.Printf("[CKKS][BOOT][ERR] Bootstrap: %v\n", err)
		return 0
	}

	*out = *ctOut
	return 1
}

func main() {}
