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

	// Explicit precision to avoid encoder internal buffer issues.
	const encPrec = uint(53)
	vm.encoder = ckks.NewEncoder(params, encPrec)

	kgen := ckks.NewKeyGenerator(params)
	sk, pk := kgen.GenKeyPairNew()
	vm.sk, vm.pk = sk, pk
	vm.rlk = kgen.GenRelinearizationKeyNew(sk)

	vm.encryptor = ckks.NewEncryptor(params, pk)
	vm.decryptor = ckks.NewDecryptor(params, sk)

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

	kgen := ckks.NewKeyGenerator(v.params)
	galEl := v.params.GaloisElementForRotation(k)

	v.gks = kgen.GenGaloisKeyNew(galEl, v.sk)

	evk := rlwe.NewMemEvaluationKeySet(v.rlk, v.gks)
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
	s := int(src)
	if s < 0 || s >= len(v.ct) {
		return
	}

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

//
// -------- Ops --------
//

//export CKKS_OpAddCC
func CKKS_OpAddCC(h C.uintptr_t, dst, a, b C.int) {
	// Match SEAL VM semantics:
	//   ciphers[lhs].scale() = ciphers[rhs].scale();
	//   add(lhs, rhs, dst);
	v := getVM(h)
	if v == nil {
		return
	}

	da, db, dd := int(a), int(b), int(dst)
	v.ensureCT(dd)

	ctA := v.ctAt(da)
	ctB := v.ctAt(db)
	ctD := v.ctAt(dd)
	if ctA == nil || ctB == nil || ctD == nil {
		return
	}

	// SEAL-style metadata alignment (lhs scale := rhs scale)
	ctA.Scale = ctB.Scale

	v.evaluator.Add(ctA, ctB, ctD)
}

//export CKKS_OpAddCP_Const
func CKKS_OpAddCP_Const(h C.uintptr_t, dst, a C.int, c C.double) {
	// Match SEAL VM semantics in spirit:
	//   ciphers[lhs].scale() = plains[rhs].scale();
	//   add_plain(lhs, plain, dst);
	//
	// Here "plain" is a constant vector; we choose a plaintext scale and then
	// force ciphertext scale to that plaintext scale before adding.
	v := getVM(h)
	if v == nil {
		return
	}

	da, dd := int(a), int(dst)
	v.ensureCT(dd)

	in := v.ctAt(da)
	out := v.ctAt(dd)
	if in == nil || out == nil {
		return
	}

	// Choose a plaintext scale. If you later replace this with "real plains[rhs]",
	// use that plaintext's scale. For const-op, we keep it consistent with current ciphertext.
	pt := ckks.NewPlaintext(v.params, in.Level())
	pt.Scale = in.Scale

	// SEAL-style: ciphertext scale := plaintext scale (no-op here, but makes semantics explicit)
	in.Scale = pt.Scale

	vec := make([]complex128, v.slots)
	for i := 0; i < v.slots; i++ {
		vec[i] = complex(float64(c), 0)
	}
	v.encoder.Encode(vec, pt)

	v.evaluator.Add(in, pt, out)
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

//export CKKS_OpMulCP_Const
func CKKS_OpMulCP_Const(h C.uintptr_t, dst, a C.int, c C.double) {
	v := getVM(h)
	if v == nil {
		return
	}
	v.ensureCT(int(dst))

	in := v.ctAt(int(a))
	if in == nil {
		return
	}

	pt := ckks.NewPlaintext(v.params, in.Level())
	pt.Scale = in.Scale

	vec := make([]complex128, v.slots)
	for i := 0; i < v.slots; i++ {
		vec[i] = complex(float64(c), 0)
	}
	v.encoder.Encode(vec, pt)

	v.evaluator.Mul(in, pt, v.ctAt(int(dst)))
}

//export CKKS_OpRotate
func CKKS_OpRotate(h C.uintptr_t, dst, a, k C.int) {
	v := getVM(h)
	if v == nil {
		return
	}
	v.ensureCT(int(dst))
	v.evaluator.Rotate(v.ctAt(int(a)), int(k), v.ctAt(int(dst)))
}

//export CKKS_OpNegate
func CKKS_OpNegate(h C.uintptr_t, dst, a C.int) {
	v := getVM(h)
	if v == nil {
		return
	}
	v.ensureCT(int(dst))
	// No Neg in your v6.1.1; use multiply by constant -1.
	v.evaluator.Mul(v.ctAt(int(a)), -1.0, v.ctAt(int(dst)))
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
		// v6.1.1 DropLevel is in-place.
		v.evaluator.DropLevel(v.ctAt(int(dst)), d)
	}
}

func main() {}
