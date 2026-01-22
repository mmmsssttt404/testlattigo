// hevm_stub.go (Lattigo v6.1.1)
// HEVM runtime with:
// - SEAL-like semantics (notably: scale metadata alignment on ADDCC/ADDCP)
// - client/server context split (create_context -> loadClient -> loadProgram)
// - cgo-safe handles (runtime/cgo.Handle) for VM, Context, Ciphertext exchange
// - LAZY rotation keys (only generate the specific rotation key when OP_ROTATEC is executed)
//
// NOTE (test-only design): server VM also loads the secret key from context so it can
// lazily generate rotation keys. This is NOT a secure deployment model, but it matches
// your current testing goal: avoid generating many rotation keys upfront.

package main

/*
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

// C-equivalent structs (match HEVMHeader.h binary layout)

typedef struct {
    uint32_t magic_number;      // 0x4845564D
    uint32_t hevm_header_size;  // 8 + sizeof(ConfigHeader)
    struct {
        uint64_t arg_length;
        uint64_t res_length;
    } config_header;
} HEVMHeader;

typedef struct {
    uint64_t config_body_length;
    uint64_t num_operations;
    uint64_t num_ctxt_buffer;
    uint64_t num_ptxt_buffer;
    uint64_t init_level;
    uint64_t reserved; // some tools read 6*u64
} ConfigBody;

typedef struct {
    uint16_t opcode;
    uint16_t dst;
    uint16_t lhs;
    uint16_t rhs;
} HEVMOperation;

*/
import "C"

import (
	"encoding/binary"
	"encoding/gob"
	"fmt"
	"io"
	"math"
	"os"
	"runtime/cgo"
	"sync"
	"unsafe"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

const (
	kMagicHEVM = uint32(0x4845564D)

	// opcodes (match your SEAL runtime)
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

	// roles for loadClient
	ROLE_FULL   = 0
	ROLE_CLIENT = 1
	ROLE_SERVER = 2
)

// -------------------- Context (gob) --------------------

type Context struct {
	ParamsLit ckks.ParametersLiteral

	// Keys
	SK  rlwe.SecretKey
	PK  rlwe.PublicKey
	RLK rlwe.RelinearizationKey
}

// -------------------- VM --------------------

type VM struct {
	// role: FULL/CLIENT/SERVER
	role int

	// parsed from hevm file
	header C.HEVMHeader
	config C.ConfigBody
	ops    []C.HEVMOperation

	// constants.bin
	constants [][]float64

	// arrays from hevm header
	argScale []uint64
	argLevel []uint64
	resScale []uint64
	resLevel []uint64
	resDst   []uint64

	// ---- Lattigo objects ----
	params    ckks.Parameters
	encoder   *ckks.Encoder
	encryptor *rlwe.Encryptor
	decryptor *rlwe.Decryptor
	evaluator *ckks.Evaluator

	sk  *rlwe.SecretKey
	pk  *rlwe.PublicKey
	rlk *rlwe.RelinearizationKey

	// rotation keys (lazy): map rot -> key
	gksMu sync.Mutex
	gkMap map[int]*rlwe.GaloisKey
	gks   []*rlwe.GaloisKey // flattened for NewMemEvaluationKeySet

	// buffers
	ciphers []rlwe.Ciphertext // ciphertext buffer
	plains  []rlwe.Plaintext  // plaintext buffer

	// runtime flags
	debug bool
	toGPU bool

	slots int
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

func getCtx(h C.uintptr_t) *Context {
	if h == 0 {
		return nil
	}
	hd := cgo.Handle(h)
	cx, ok := hd.Value().(*Context)
	if !ok {
		return nil
	}
	return cx
}

func (v *VM) dbg(format string, args ...any) {
	if v != nil && v.debug {
		fmt.Printf(format, args...)
	}
}

// -------------------- binary helpers --------------------

func readU32(r io.Reader) (uint32, error) {
	var b [4]byte
	if _, err := io.ReadFull(r, b[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(b[:]), nil
}

func readU64(r io.Reader) (uint64, error) {
	var b [8]byte
	if _, err := io.ReadFull(r, b[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(b[:]), nil
}

func readOp(r io.Reader) (C.HEVMOperation, error) {
	var b [8]byte
	if _, err := io.ReadFull(r, b[:]); err != nil {
		return C.HEVMOperation{}, err
	}
	return C.HEVMOperation{
		opcode: C.uint16_t(binary.LittleEndian.Uint16(b[0:2])),
		dst:    C.uint16_t(binary.LittleEndian.Uint16(b[2:4])),
		lhs:    C.uint16_t(binary.LittleEndian.Uint16(b[4:6])),
		rhs:    C.uint16_t(binary.LittleEndian.Uint16(b[6:8])),
	}, nil
}

func readConstantsBin(path string) ([][]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var b8 [8]byte
	if _, err := io.ReadFull(f, b8[:]); err != nil {
		return nil, err
	}
	n := int64(binary.LittleEndian.Uint64(b8[:]))
	if n < 0 || n > 1_000_000 {
		return nil, fmt.Errorf("constants len invalid: %d", n)
	}

	out := make([][]float64, n)
	for i := int64(0); i < n; i++ {
		if _, err := io.ReadFull(f, b8[:]); err != nil {
			return nil, err
		}
		veclen := int64(binary.LittleEndian.Uint64(b8[:]))
		if veclen < 0 || veclen > 1_000_000 {
			return nil, fmt.Errorf("constants veclen invalid: %d", veclen)
		}
		raw := make([]byte, veclen*8)
		if _, err := io.ReadFull(f, raw); err != nil {
			return nil, err
		}
		vec := make([]float64, veclen)
		for j := int64(0); j < veclen; j++ {
			u := binary.LittleEndian.Uint64(raw[j*8 : j*8+8])
			vec[j] = math.Float64frombits(u)
		}
		out[i] = vec
	}
	return out, nil
}

// -------------------- CKKS setup --------------------

func defaultParamsLiteral() ckks.ParametersLiteral {
	logN := 15
	logQ := make([]int, 14)
	for i := range logQ {
		logQ[i] = 60
	}
	logP := []int{60, 60}
	return ckks.ParametersLiteral{
		LogN:            logN,
		LogQ:            logQ,
		LogP:            logP,
		LogDefaultScale: 40,
	}
}

// FULL VM: create params + generate keys
func (v *VM) setupCKKSFixedNewKeys() error {
	pl := defaultParamsLiteral()

	params, err := ckks.NewParametersFromLiteral(pl)
	if err != nil {
		return err
	}
	v.params = params
	v.slots = 1 << (pl.LogN - 1)

	const encPrec = uint(53)
	v.encoder = ckks.NewEncoder(params, encPrec)

	kgen := ckks.NewKeyGenerator(params)
	v.sk, v.pk = kgen.GenKeyPairNew()
	v.rlk = kgen.GenRelinearizationKeyNew(v.sk)

	v.encryptor = ckks.NewEncryptor(params, v.pk)
	v.decryptor = ckks.NewDecryptor(params, v.sk)

	// evaluator initially with RLK only; rotations are lazy
	v.resetRotationCache()
	evk := rlwe.NewMemEvaluationKeySet(v.rlk)
	v.evaluator = ckks.NewEvaluator(params, evk)

	return nil
}

// CLIENT/SERVER VM: load params + keys from context
func (v *VM) setupFromContext(cx *Context) error {
	if cx == nil {
		return fmt.Errorf("nil context")
	}
	params, err := ckks.NewParametersFromLiteral(cx.ParamsLit)
	if err != nil {
		return err
	}
	v.params = params
	v.slots = 1 << (cx.ParamsLit.LogN - 1)

	const encPrec = uint(53)
	v.encoder = ckks.NewEncoder(params, encPrec)

	// attach keys
	v.sk = &cx.SK
	v.pk = &cx.PK
	v.rlk = &cx.RLK

	// encryptor/decryptor
	v.encryptor = ckks.NewEncryptor(params, v.pk)
	v.decryptor = ckks.NewDecryptor(params, v.sk)

	// evaluator initially with RLK only; rotations are lazy
	v.resetRotationCache()
	evk := rlwe.NewMemEvaluationKeySet(v.rlk)
	v.evaluator = ckks.NewEvaluator(params, evk)

	return nil
}

func (v *VM) resetRotationCache() {
	v.gksMu.Lock()
	defer v.gksMu.Unlock()
	v.gkMap = make(map[int]*rlwe.GaloisKey)
	v.gks = nil
}

// LAZY: only create the galois key for the exact rotation used.
func (v *VM) ensureRotationKey(k int) {
	if k == 0 {
		return
	}

	// fast check
	v.gksMu.Lock()
	if v.gkMap == nil {
		v.gkMap = make(map[int]*rlwe.GaloisKey)
	}
	if _, ok := v.gkMap[k]; ok {
		v.gksMu.Unlock()
		return
	}
	v.gksMu.Unlock()

	// slow: generate outside lock
	galEl := v.params.GaloisElementForRotation(k)
	kgen := ckks.NewKeyGenerator(v.params)
	// v6.1.1: returns []*rlwe.GaloisKey
	gksOne := kgen.GenGaloisKeysNew([]uint64{galEl}, v.sk)
	if len(gksOne) == 0 || gksOne[0] == nil {
		return
	}
	gk := gksOne[0]

	// commit + rebuild evaluator keyset
	v.gksMu.Lock()
	defer v.gksMu.Unlock()
	if _, ok := v.gkMap[k]; ok {
		return
	}
	v.gkMap[k] = gk
	v.gks = append(v.gks, gk)

	evk := rlwe.NewMemEvaluationKeySet(v.rlk, v.gks...)
	v.evaluator = ckks.NewEvaluator(v.params, evk)
}

// -------------------- helpers: encode constant -> plaintext --------------------

func (v *VM) encodeToPlainAt(dst *rlwe.Plaintext, values []float64, level int, log2Scale int) {
	pt := ckks.NewPlaintext(v.params, level)
	pt.Scale = rlwe.NewScale(math.Exp2(float64(log2Scale)))

	vec := make([]complex128, v.slots)
	if len(values) > 0 {
		for i := 0; i < v.slots; i++ {
			vec[i] = complex(values[i%len(values)], 0)
		}
	}
	v.encoder.Encode(vec, pt)

	*dst = *pt
}

// -------------------- HEVM loader --------------------

func (v *VM) loadHEVM(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	magic, err := readU32(f)
	if err != nil {
		return err
	}
	hsize, err := readU32(f)
	if err != nil {
		return err
	}
	argLen, err := readU64(f)
	if err != nil {
		return err
	}
	resLen, err := readU64(f)
	if err != nil {
		return err
	}

	v.header.magic_number = C.uint32_t(magic)
	v.header.hevm_header_size = C.uint32_t(hsize)
	v.header.config_header.arg_length = C.uint64_t(argLen)
	v.header.config_header.res_length = C.uint64_t(resLen)

	if magic != kMagicHEVM {
		return fmt.Errorf("bad magic: got 0x%x expect 0x%x", magic, kMagicHEVM)
	}

	// ConfigBody: 6*u64
	cbl, err := readU64(f)
	if err != nil {
		return err
	}
	numOps, err := readU64(f)
	if err != nil {
		return err
	}
	numCtxt, err := readU64(f)
	if err != nil {
		return err
	}
	numPtxt, err := readU64(f)
	if err != nil {
		return err
	}
	initLvl, err := readU64(f)
	if err != nil {
		return err
	}
	reserved, err := readU64(f)
	if err != nil {
		return err
	}

	v.config.config_body_length = C.uint64_t(cbl)
	v.config.num_operations = C.uint64_t(numOps)
	v.config.num_ctxt_buffer = C.uint64_t(numCtxt)
	v.config.num_ptxt_buffer = C.uint64_t(numPtxt)
	v.config.init_level = C.uint64_t(initLvl)
	v.config.reserved = C.uint64_t(reserved)

	aLen := int(argLen)
	rLen := int(resLen)

	v.argScale = make([]uint64, aLen)
	v.argLevel = make([]uint64, aLen)
	v.resScale = make([]uint64, rLen)
	v.resLevel = make([]uint64, rLen)
	v.resDst = make([]uint64, rLen)

	readArr := func(dst []uint64) error {
		for i := range dst {
			x, err := readU64(f)
			if err != nil {
				return err
			}
			dst[i] = x
		}
		return nil
	}
	if err := readArr(v.argScale); err != nil {
		return err
	}
	if err := readArr(v.argLevel); err != nil {
		return err
	}
	if err := readArr(v.resScale); err != nil {
		return err
	}
	if err := readArr(v.resLevel); err != nil {
		return err
	}
	if err := readArr(v.resDst); err != nil {
		return err
	}

	v.ops = make([]C.HEVMOperation, int(numOps))
	for i := 0; i < int(numOps); i++ {
		op, err := readOp(f)
		if err != nil {
			return err
		}
		v.ops[i] = op
	}

	// allocate buffers for ct/pt (ct buffer also holds args+results as in SEAL runtime)
	ctN := int(numCtxt)
	ptN := int(numPtxt)
	if ctN < aLen+rLen {
		ctN = aLen + rLen
	}
	if ctN < 1 {
		ctN = 1
	}
	if ptN < 1 {
		ptN = 1
	}

	v.ciphers = make([]rlwe.Ciphertext, ctN)
	v.plains = make([]rlwe.Plaintext, ptN)

	maxLevel := v.params.MaxLevel()
	for i := range v.ciphers {
		v.ciphers[i] = *ckks.NewCiphertext(v.params, 1, maxLevel)
	}
	for i := range v.plains {
		v.plains[i] = *ckks.NewPlaintext(v.params, maxLevel)
	}

	return nil
}

// -------------------- Ciphertext handle exchange --------------------

type CtxtBox struct {
	CT rlwe.Ciphertext
}

func exportCiphertextHandle(ct *rlwe.Ciphertext) C.uintptr_t {
	if ct == nil {
		return 0
	}
	box := &CtxtBox{CT: *ct}
	return C.uintptr_t(cgo.NewHandle(box))
}

func importCiphertextHandle(h C.uintptr_t) *rlwe.Ciphertext {
	if h == 0 {
		return nil
	}
	hd := cgo.Handle(h)
	box, ok := hd.Value().(*CtxtBox)
	if !ok || box == nil {
		return nil
	}
	return &box.CT
}

// -------------------- Exported C ABI (HANDLE-BASED) --------------------

//export initFullVM
func initFullVM(path *C.char, useGPU C._Bool) C.uintptr_t {
	_ = C.GoString(path)
	vm := &VM{role: ROLE_FULL, debug: false, toGPU: bool(useGPU)}
	h := cgo.NewHandle(vm)
	fmt.Printf("[LATTIGO_HEVM] initFullVM useGPU=%v handle=%d\n", bool(useGPU), uintptr(h))
	return C.uintptr_t(h)
}

//export initClientVM
func initClientVM(path *C.char) C.uintptr_t {
	_ = C.GoString(path)
	vm := &VM{role: ROLE_CLIENT, debug: false}
	h := cgo.NewHandle(vm)
	fmt.Printf("[LATTIGO_HEVM] initClientVM handle=%d\n", uintptr(h))
	return C.uintptr_t(h)
}

//export initServerVM
func initServerVM(path *C.char) C.uintptr_t {
	_ = C.GoString(path)
	vm := &VM{role: ROLE_SERVER, debug: false}
	h := cgo.NewHandle(vm)
	fmt.Printf("[LATTIGO_HEVM] initServerVM handle=%d\n", uintptr(h))
	return C.uintptr_t(h)
}

//export freeVM
func freeVM(h C.uintptr_t) {
	if h == 0 {
		return
	}
	cgo.Handle(h).Delete()
}

//export create_context
func create_context(path *C.char) C.uintptr_t {
	p := C.GoString(path)

	// create context: params + keys
	pl := defaultParamsLiteral()
	params, err := ckks.NewParametersFromLiteral(pl)
	if err != nil {
		fmt.Printf("[LATTIGO_HEVM][ERR] create_context params: %v\n", err)
		return 0
	}

	kgen := ckks.NewKeyGenerator(params)
	sk, pk := kgen.GenKeyPairNew()
	rlk := kgen.GenRelinearizationKeyNew(sk)

	cx := &Context{
		ParamsLit: pl,
		SK:        *sk,
		PK:        *pk,
		RLK:       *rlk,
	}

	// write gob
	if p != "" {
		if f, err := os.Create(p); err == nil {
			enc := gob.NewEncoder(f)
			if e := enc.Encode(cx); e != nil {
				fmt.Printf("[LATTIGO_HEVM][ERR] create_context encode: %v\n", e)
			}
			_ = f.Close()
		} else {
			fmt.Printf("[LATTIGO_HEVM][ERR] create_context create file: %v\n", err)
		}
	}

	h := cgo.NewHandle(cx)
	fmt.Printf("[LATTIGO_HEVM] create_context path=%q handle=%d\n", p, uintptr(h))
	return C.uintptr_t(h)
}

//export freeContext
func freeContext(h C.uintptr_t) {
	if h == 0 {
		return
	}
	cgo.Handle(h).Delete()
}

//export loadClient
func loadClient(vmH C.uintptr_t, ctxH C.uintptr_t) {
	v := getVM(vmH)
	cx := getCtx(ctxH)
	if v == nil || cx == nil {
		return
	}
	// role message (match your log style)
	fmt.Printf("[LATTIGO_HEVM] loadClient vm=%d ctx=%d role=%d (handle)\n", uintptr(vmH), uintptr(ctxH), v.role)

	if err := v.setupFromContext(cx); err != nil {
		fmt.Printf("[LATTIGO_HEVM][ERR] loadClient setup: %v\n", err)
		return
	}
}

//export load
func load(h C.uintptr_t, constPath *C.char, hevmPath *C.char) {
	v := getVM(h)
	if v == nil {
		return
	}

	cpath := C.GoString(constPath)
	hpath := C.GoString(hevmPath)
	fmt.Printf("[LATTIGO_HEVM] load handle=%d const=%q hevm=%q\n", uintptr(h), cpath, hpath)

	consts, err := readConstantsBin(cpath)
	if err != nil {
		fmt.Printf("[LATTIGO_HEVM][ERR] read constants: %v\n", err)
		return
	}
	v.constants = consts

	// Full VM creates keys here
	if err := v.setupCKKSFixedNewKeys(); err != nil {
		fmt.Printf("[LATTIGO_HEVM][ERR] setup CKKS: %v\n", err)
		return
	}

	if err := v.loadHEVM(hpath); err != nil {
		fmt.Printf("[LATTIGO_HEVM][ERR] read hevm: %v\n", err)
		return
	}
}

//export loadProgram
func loadProgram(h C.uintptr_t, constPath *C.char, hevmPath *C.char) {
	v := getVM(h)
	if v == nil {
		return
	}
	cpath := C.GoString(constPath)
	hpath := C.GoString(hevmPath)
	fmt.Printf("[LATTIGO_HEVM] loadProgram handle=%d const=%q hevm=%q\n", uintptr(h), cpath, hpath)

	consts, err := readConstantsBin(cpath)
	if err != nil {
		fmt.Printf("[LATTIGO_HEVM][ERR] read constants: %v\n", err)
		return
	}
	v.constants = consts

	// expect params/keys already present (from loadClient)
	if (v.encoder == nil) || (v.evaluator == nil) || (v.params.LogN() == 0) {
		fmt.Printf("[LATTIGO_HEVM][ERR] loadProgram: call loadClient() first\n")
		return
	}

	if err := v.loadHEVM(hpath); err != nil {
		fmt.Printf("[LATTIGO_HEVM][ERR] read hevm: %v\n", err)
		return
	}

	// ensure rotation cache starts empty for this program
	v.resetRotationCache()
	evk := rlwe.NewMemEvaluationKeySet(v.rlk)
	v.evaluator = ckks.NewEvaluator(v.params, evk)
}

//export preprocess
func preprocess(h C.uintptr_t) {
	v := getVM(h)
	if v == nil {
		return
	}
	fmt.Printf("[LATTIGO_HEVM] preprocess handle=%d\n", uintptr(h))

	// For opcode==ENCODE, encode constants into plains[dst] at given (level, scale)
	for _, op := range v.ops {
		if int(op.opcode) != OP_ENCODE {
			continue
		}

		dst := int(op.dst)
		lhs := int(op.lhs)
		level := int(uint16(op.rhs) >> 10)
		log2Scale := int(uint16(op.rhs) & 0x3FF)

		if dst < 0 || dst >= len(v.plains) {
			continue
		}

		var src []float64
		if lhs == 65535 {
			src = []float64{1.0}
		} else if lhs >= 0 && lhs < len(v.constants) {
			src = v.constants[lhs]
		} else {
			src = nil
		}

		v.encodeToPlainAt(&v.plains[dst], src, level, log2Scale)
	}
}

//export run
func run(h C.uintptr_t) {
	v := getVM(h)
	if v == nil {
		return
	}
	fmt.Printf("[LATTIGO_HEVM] run handle=%d\n", uintptr(h))

	ctAt := func(i int) *rlwe.Ciphertext {
		if i < 0 || i >= len(v.ciphers) {
			return nil
		}
		return &v.ciphers[i]
	}
	ptAt := func(i int) *rlwe.Plaintext {
		if i < 0 || i >= len(v.plains) {
			return nil
		}
		return &v.plains[i]
	}

	for _, op := range v.ops {
		switch int(op.opcode) {
		case OP_ENCODE:
			continue

		case OP_ROTATEC: {
			dst := int(op.dst)
			src := int(op.lhs)
			k := int(int16(op.rhs))
			in, out := ctAt(src), ctAt(dst)
			if in == nil || out == nil {
				continue
			}
			// LAZY: only generate this rotation key when it is actually used
			v.ensureRotationKey(k)
			v.evaluator.Rotate(in, k, out)
		}

		case OP_NEGATEC: {
			dst := int(op.dst)
			src := int(op.lhs)
			in, out := ctAt(src), ctAt(dst)
			if in == nil || out == nil {
				continue
			}
			v.evaluator.Mul(in, -1.0, out)
		}

		case OP_RESCALEC: {
			dst := int(op.dst)
			src := int(op.lhs)
			in, out := ctAt(src), ctAt(dst)
			if in == nil || out == nil {
				continue
			}
			v.evaluator.Rescale(in, out)
		}

		case OP_MODSWC: {
			dst := int(op.dst)
			src := int(op.lhs)
			down := int(op.rhs)
			in, out := ctAt(src), ctAt(dst)
			if in == nil || out == nil {
				continue
			}
			*out = *in
			if down > 0 {
				v.evaluator.DropLevel(out, down) // in-place
			}
		}

		case OP_UPSCALEC:
			continue

		case OP_ADDCC: {
			dst := int(op.dst)
			l := int(op.lhs)
			r := int(op.rhs)
			in0, in1, out := ctAt(l), ctAt(r), ctAt(dst)
			if in0 == nil || in1 == nil || out == nil {
				continue
			}
			// SEAL-style metadata alignment
			in0.Scale = in1.Scale
			v.evaluator.Add(in0, in1, out)
		}

		case OP_ADDCP: {
			dst := int(op.dst)
			l := int(op.lhs)
			p := int(op.rhs)
			in0, pt, out := ctAt(l), ptAt(p), ctAt(dst)
			if in0 == nil || pt == nil || out == nil {
				continue
			}
			// SEAL-style metadata alignment
			in0.Scale = pt.Scale
			v.evaluator.Add(in0, pt, out)
		}

		case OP_MULCC: {
			dst := int(op.dst)
			l := int(op.lhs)
			r := int(op.rhs)
			in0, in1, out := ctAt(l), ctAt(r), ctAt(dst)
			if in0 == nil || in1 == nil || out == nil {
				continue
			}
			v.evaluator.MulRelin(in0, in1, out)
		}

		case OP_MULCP: {
			dst := int(op.dst)
			l := int(op.lhs)
			p := int(op.rhs)
			in0, pt, out := ctAt(l), ptAt(p), ctAt(dst)
			if in0 == nil || pt == nil || out == nil {
				continue
			}
			v.evaluator.Mul(in0, pt, out)
		}

		case OP_BOOT:
			continue

		default:
			continue
		}
	}
}

//export getArgLen
func getArgLen(h C.uintptr_t) C.int64_t {
	v := getVM(h)
	if v == nil {
		return 0
	}
	return C.int64_t(v.header.config_header.arg_length)
}

//export getResLen
func getResLen(h C.uintptr_t) C.int64_t {
	v := getVM(h)
	if v == nil {
		return 0
	}
	return C.int64_t(v.header.config_header.res_length)
}

//export encrypt
func encrypt(h C.uintptr_t, idx C.int64_t, data *C.double, n C.int) {
	v := getVM(h)
	if v == nil {
		return
	}
	i := int(idx)
	ln := int(n)
	if i < 0 || i >= len(v.ciphers) || data == nil || ln <= 0 {
		return
	}

	level := v.params.MaxLevel()
	log2Scale := int(v.params.LogDefaultScale())
	if i < len(v.argLevel) && int(v.argLevel[i]) <= v.params.MaxLevel() {
		level = int(v.argLevel[i])
	}
	if i < len(v.argScale) {
		log2Scale = int(v.argScale[i])
	}

	pt := ckks.NewPlaintext(v.params, level)
	pt.Scale = rlwe.NewScale(math.Exp2(float64(log2Scale)))

	src := (*[1 << 30]C.double)(unsafe.Pointer(data))[:ln:ln]
	vec := make([]complex128, v.slots)
	for s := 0; s < v.slots; s++ {
		vec[s] = complex(float64(src[s%ln]), 0)
	}
	v.encoder.Encode(vec, pt)

	ct := ckks.NewCiphertext(v.params, 1, level)
	v.encryptor.Encrypt(pt, ct)
	v.ciphers[i] = *ct
}

//export decrypt
func decrypt(h C.uintptr_t, idx C.int64_t, out *C.double) {
	v := getVM(h)
	if v == nil {
		return
	}
	i := int(idx)
	if out == nil || i < 0 || i >= len(v.ciphers) {
		return
	}

	pt := ckks.NewPlaintext(v.params, v.ciphers[i].Level())
	v.decryptor.Decrypt(&v.ciphers[i], pt)

	vec := make([]complex128, v.slots)
	v.encoder.Decode(pt, vec)

	outArr := (*[1 << 14]C.double)(unsafe.Pointer(out))
	lim := v.slots
	if lim > (1 << 14) {
		lim = 1 << 14
	}
	for k := 0; k < lim; k++ {
		outArr[k] = C.double(real(vec[k]))
	}
}

//export decrypt_result
func decrypt_result(h C.uintptr_t, resIdx C.int64_t, out *C.double) {
	v := getVM(h)
	if v == nil || out == nil {
		return
	}
	r := int(resIdx)
	if r < 0 || r >= len(v.resDst) {
		*out = 0
		return
	}
	ctID := int(v.resDst[r])
	decrypt(h, C.int64_t(ctID), out)
}

//export getResIdx
func getResIdx(h C.uintptr_t, i C.int64_t) C.int64_t {
	v := getVM(h)
	if v == nil {
		return i
	}
	ii := int(i)
	if ii < 0 || ii >= len(v.resDst) {
		return i
	}
	return C.int64_t(v.resDst[ii])
}

//export getCtxt
func getCtxt(h C.uintptr_t, i C.int64_t) C.uintptr_t {
	v := getVM(h)
	if v == nil {
		return 0
	}
	ii := int(i)
	if ii < 0 || ii >= len(v.ciphers) {
		return 0
	}
	return exportCiphertextHandle(&v.ciphers[ii])
}

//export setCtxt
func setCtxt(h C.uintptr_t, i C.int64_t, ctH C.uintptr_t) {
	v := getVM(h)
	if v == nil {
		return
	}
	ii := int(i)
	if ii < 0 || ii >= len(v.ciphers) {
		return
	}
	ct := importCiphertextHandle(ctH)
	if ct == nil {
		return
	}
	v.ciphers[ii] = *ct
}

//export freeCtxt
func freeCtxt(ctH C.uintptr_t) {
	if ctH == 0 {
		return
	}
	cgo.Handle(ctH).Delete()
}

//export setDebug
func setDebug(h C.uintptr_t, enable C._Bool) {
	v := getVM(h)
	if v != nil {
		v.debug = bool(enable)
	}
}

//export setToGPU
func setToGPU(h C.uintptr_t, ongpu C._Bool) {
	v := getVM(h)
	if v != nil {
		v.toGPU = bool(ongpu)
	}
}

//export printMem
func printMem(h C.uintptr_t) {
	v := getVM(h)
	if v == nil {
		return
	}
	fmt.Printf("[LATTIGO_HEVM] printMem handle=%d (ckks)\n", uintptr(h))
}

func main() {}
