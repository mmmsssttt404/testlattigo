// hevm_stub.go (Lattigo v6.1.1) — HEVM ABI aligned to HEaaN backend style
//
// Build:
//   go build -buildmode=c-shared -o libLATTIGO_HEVM.so hevm_stub.go
//
// ABI compatibility goal:
// - Exported C function NAMES match the HEaaN reference:
//     initFullVM / initClientVM / initServerVM / create_context
//     load / loadClient / encrypt / decrypt / decrypt_result / getResIdx / getCtxt
//     preprocess / run / getArgLen / getResLen / setDebug / setToGPU / printMem
// - Internal op handlers are methods (rotate/negate/rescale/modswitch/upscale/addcc/addcp/mulcc/mulcp/bootstrap)
//   and run() dispatches to them (no big switch bodies).
//
// Notes:
// - Because Go cannot hand out pointers to Go-managed memory to C, all "void* handles"
//   are implemented as C-malloc'ed pointers holding a cgo.Handle integer.
//   => freeVM/freeContext/freeCtxt must be called to avoid leaks.
// - loadClient(void* vm, void* is) in HEaaN takes an std::istream*. Here we accept:
//     *is == NULL  => no-op
//     *is != NULL  => treat is as (char*) path to a .hevm file and load ONLY the header/metadata part.
//   If you don't need header-only loading, you can ignore loadClient and just call load().
//
// - Params domain is UNIFIED: VM ops run under BootstrappingParameters domain (btp.BootstrappingParameters).
//
// - Upscale is not supported (assert-like error message).

package main

/*
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

typedef struct {
    uint32_t magic_number;      // 0x4845564D
    uint32_t hevm_header_size;  // 8 + sizeof(config_header)
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
    uint64_t reserved;
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
	"path/filepath"
	"runtime/cgo"
	"sync"
	"time"
	"unsafe"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

const (
	kMagicHEVM = uint32(0x4845564D)

	// opcodes (match HEaaN backend)
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

	// roles
	ROLE_FULL   = 0
	ROLE_CLIENT = 1
	ROLE_SERVER = 2
)

// -------------------- preset --------------------

type defaultParametersLiteral struct {
	CKKS ckks.ParametersLiteral
	BOOT bootstrapping.ParametersLiteral
}

// Official Lattigo boot preset literals (v6.1.1)
var N16QP1546H192H32 = defaultParametersLiteral{
	CKKS: bootstrapping.N16QP1546H192H32.SchemeParams,
	BOOT: bootstrapping.N16QP1546H192H32.BootstrappingParams,
}

func bootPreset() (name string, p defaultParametersLiteral) {
	return "N16QP1546H192H32", N16QP1546H192H32
}

const contextGobName = "context.lattigo.gob"

// -------------------- Context saved on disk --------------------

type Context struct {
	ParamName string
	ParamsLit ckks.ParametersLiteral
	BootLit   bootstrapping.ParametersLiteral

	// Keys are generated under BootstrappingParameters domain.
	SK  rlwe.SecretKey
	PK  rlwe.PublicKey
	RLK rlwe.RelinearizationKey
}

// -------------------- Handle helpers (C malloc + cgo.Handle) --------------------

// handlePtr layout: *(uintptr*)ptr == uintptr(cgo.Handle)
func newHandlePtr(v any) unsafe.Pointer {
	h := cgo.NewHandle(v)
	p := C.malloc(C.size_t(unsafe.Sizeof(uintptr(0))))
	*(*uintptr)(p) = uintptr(h)
	return p
}
func loadHandlePtr(p unsafe.Pointer) (cgo.Handle, bool) {
	if p == nil {
		return cgo.Handle(0), false
	}
	u := *(*uintptr)(p)
	if u == 0 {
		return cgo.Handle(0), false
	}
	return cgo.Handle(u), true
}
func freeHandlePtr(p unsafe.Pointer) {
	if p == nil {
		return
	}
	u := *(*uintptr)(p)
	if u != 0 {
		cgo.Handle(u).Delete()
	}
	C.free(p)
}

// -------------------- HEVM (Lattigo) --------------------

type LATTIGO_HEVM struct {
	// data buffers
	buffer [][]float64

	// hevm file metadata
	header C.HEVMHeader
	config C.ConfigBody
	ops    []C.HEVMOperation

	arg_scale []uint64
	arg_level []uint64
	res_scale []uint64
	res_level []uint64
	res_dst   []uint64

	// ckks objects
	paramName string

	// residual/scheme params, and boot parameters (constructed from residual + boot literal)
	residualParams ckks.Parameters
	bootLit        bootstrapping.ParametersLiteral
	bootParams     bootstrapping.Parameters

	// ONLY domain for all ops:
	params ckks.Parameters
	slots  int

	sk  *rlwe.SecretKey
	pk  *rlwe.PublicKey
	rlk *rlwe.RelinearizationKey

	encoder   *ckks.Encoder
	encryptor *rlwe.Encryptor
	decryptor *rlwe.Decryptor
	evaluator *ckks.Evaluator

	// ciphertext/plain buffers
	ciphers []rlwe.Ciphertext
	scalec  []float64 // track log2(scale) for debug / decode hint
	plains  []rlwe.Plaintext
	scalep  []float64 // log2(scale) for pt buffer
	levelp  []uint64

	// preencode path
	msgs      [][]float64
	preencode bool

	// lazy rotation keys (needs SK)
	gksMu sync.Mutex
	gkMap map[int]*rlwe.GaloisKey
	gks   []*rlwe.GaloisKey

	// bootstrapping
	bootMu     sync.Mutex
	bootInited bool
	bootErr    error
	bootEVK    *bootstrapping.EvaluationKeys
	bootEval   *bootstrapping.Evaluator
	boot_time  uint64 // microseconds
	boot_cnt   uint64

	// flags
	debug bool
	togpu bool // kept for ABI parity only
}

// -------------------- VM creation / load (disk) --------------------

func (v *LATTIGO_HEVM) setupFromPresetNewKeys() error {
	name, preset := bootPreset()
	v.paramName = name
	v.bootLit = preset.BOOT

	residual, err := ckks.NewParametersFromLiteral(preset.CKKS)
	if err != nil {
		return err
	}
	btp, err := bootstrapping.NewParametersFromLiteral(residual, preset.BOOT)
	if err != nil {
		return err
	}

	vmParams := btp.BootstrappingParameters

	v.residualParams = residual
	v.bootParams = btp
	v.params = vmParams
	v.slots = v.params.MaxSlots()

	const encPrec = uint(53)
	v.encoder = ckks.NewEncoder(v.params, encPrec)

	kgen := ckks.NewKeyGenerator(v.params)
	v.sk, v.pk = kgen.GenKeyPairNew()
	v.rlk = kgen.GenRelinearizationKeyNew(v.sk)

	v.encryptor = ckks.NewEncryptor(v.params, v.pk)
	v.decryptor = ckks.NewDecryptor(v.params, v.sk)

	v.resetRotationCache()
	evk := rlwe.NewMemEvaluationKeySet(v.rlk)
	v.evaluator = ckks.NewEvaluator(v.params, evk)

	v.ensureBuffers(64, 64)
	v.ensureMsgBuffers(64)

	// reset boot state
	v.bootInited = false
	v.bootErr = nil
	v.bootEVK = nil
	v.bootEval = nil
	v.boot_time = 0
	v.boot_cnt = 0
	return nil
}

func (v *LATTIGO_HEVM) setupFromContext(cx *Context) error {
	if cx == nil {
		return fmt.Errorf("nil context")
	}

	residual, err := ckks.NewParametersFromLiteral(cx.ParamsLit)
	if err != nil {
		return err
	}
	btp, err := bootstrapping.NewParametersFromLiteral(residual, cx.BootLit)
	if err != nil {
		return err
	}

	vmParams := btp.BootstrappingParameters

	v.paramName = cx.ParamName
	v.bootLit = cx.BootLit

	v.residualParams = residual
	v.bootParams = btp
	v.params = vmParams
	v.slots = v.params.MaxSlots()

	const encPrec = uint(53)
	v.encoder = ckks.NewEncoder(v.params, encPrec)

	// keys are under vmParams domain
	v.sk = &cx.SK
	v.pk = &cx.PK
	v.rlk = &cx.RLK

	v.encryptor = ckks.NewEncryptor(v.params, v.pk)
	v.decryptor = ckks.NewDecryptor(v.params, v.sk)

	v.resetRotationCache()
	evk := rlwe.NewMemEvaluationKeySet(v.rlk)
	v.evaluator = ckks.NewEvaluator(v.params, evk)

	v.ensureBuffers(64, 64)
	v.ensureMsgBuffers(64)

	v.bootInited = false
	v.bootErr = nil
	v.bootEVK = nil
	v.bootEval = nil
	v.boot_time = 0
	v.boot_cnt = 0
	return nil
}

func (v *LATTIGO_HEVM) contextPath(dir string) string {
	return filepath.Join(dir, contextGobName)
}

func (v *LATTIGO_HEVM) loadLattigo(dir string) error {
	p := v.contextPath(dir)
	f, err := os.Open(p)
	if err != nil {
		return err
	}
	defer f.Close()

	var cx Context
	if err := gob.NewDecoder(f).Decode(&cx); err != nil {
		return err
	}
	return v.setupFromContext(&cx)
}

// -------------------- buffers --------------------

func (v *LATTIGO_HEVM) ensureBuffers(ctN, ptN int) {
	if ctN < 1 {
		ctN = 1
	}
	if ptN < 1 {
		ptN = 1
	}

	if len(v.ciphers) < ctN {
		old := len(v.ciphers)
		v.ciphers = append(v.ciphers, make([]rlwe.Ciphertext, ctN-old)...)
		v.scalec = append(v.scalec, make([]float64, ctN-old)...)
		for i := old; i < ctN; i++ {
			v.ciphers[i] = *ckks.NewCiphertext(v.params, 1, v.params.MaxLevel())
			v.scalec[i] = float64(v.params.LogDefaultScale())
		}
	}

	if len(v.plains) < ptN {
		old := len(v.plains)
		v.plains = append(v.plains, make([]rlwe.Plaintext, ptN-old)...)
		v.scalep = append(v.scalep, make([]float64, ptN-old)...)
		v.levelp = append(v.levelp, make([]uint64, ptN-old)...)
		for i := old; i < ptN; i++ {
			v.plains[i] = *ckks.NewPlaintext(v.params, v.params.MaxLevel())
			v.scalep[i] = float64(v.params.LogDefaultScale())
			v.levelp[i] = uint64(v.params.MaxLevel())
		}
	}
}

func (v *LATTIGO_HEVM) ensureMsgBuffers(n int) {
	if n < 1 {
		n = 1
	}
	if len(v.msgs) < n {
		old := len(v.msgs)
		v.msgs = append(v.msgs, make([][]float64, n-old)...)
		for i := old; i < n; i++ {
			v.msgs[i] = make([]float64, 0)
		}
	}
}

// -------------------- rotation keys --------------------

func (v *LATTIGO_HEVM) resetRotationCache() {
	v.gksMu.Lock()
	defer v.gksMu.Unlock()
	v.gkMap = make(map[int]*rlwe.GaloisKey)
	v.gks = nil
}

func (v *LATTIGO_HEVM) clampCiphertextToMaxLevel(ct *rlwe.Ciphertext) {
	if ct == nil || v.params.LogN() == 0 {
		return
	}
	maxL := v.params.MaxLevel()
	for i := range ct.Value {
		if len(ct.Value[i].Coeffs) > maxL+1 {
			ct.Value[i].Coeffs = ct.Value[i].Coeffs[:maxL+1]
		}
	}
}

func (v *LATTIGO_HEVM) ensureRotationKey(k int) {
	if k == 0 || v.sk == nil || v.params.LogN() == 0 {
		return
	}

	v.gksMu.Lock()
	if v.gkMap == nil {
		v.gkMap = make(map[int]*rlwe.GaloisKey)
	}
	if _, ok := v.gkMap[k]; ok {
		v.gksMu.Unlock()
		return
	}
	v.gksMu.Unlock()

	galEl := v.params.GaloisElementForRotation(k)
	kgen := ckks.NewKeyGenerator(v.params)
	gksOne := kgen.GenGaloisKeysNew([]uint64{galEl}, v.sk)
	if len(gksOne) == 0 || gksOne[0] == nil {
		return
	}
	gk := gksOne[0]

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

// -------------------- bootstrapping --------------------

func (v *LATTIGO_HEVM) ensureBoot() error {
	v.bootMu.Lock()
	defer v.bootMu.Unlock()

	if v.bootInited {
		return v.bootErr
	}
	v.bootInited = true

	if v.sk == nil {
		v.bootErr = fmt.Errorf("boot: missing secret key (SK)")
		return v.bootErr
	}
	if v.residualParams.LogN() == 0 {
		v.bootErr = fmt.Errorf("boot: missing residual params")
		return v.bootErr
	}

	btp, err := bootstrapping.NewParametersFromLiteral(v.residualParams, v.bootLit)
	if err != nil {
		v.bootErr = fmt.Errorf("boot: NewParametersFromLiteral: %w", err)
		return v.bootErr
	}
	v.bootParams = btp

	evk, _, err := v.bootParams.GenEvaluationKeys(v.sk)
	if err != nil {
		v.bootErr = fmt.Errorf("boot: GenEvaluationKeys: %w", err)
		return v.bootErr
	}
	v.bootEVK = evk

	eval, err := bootstrapping.NewEvaluator(v.bootParams, evk)
	if err != nil {
		v.bootErr = fmt.Errorf("boot: NewEvaluator: %w", err)
		return v.bootErr
	}
	v.bootEval = eval

	fmt.Printf("[LATTIGO_HEVM][BOOT] enabled PARAM=%s logN=%d slots=%d maxLevel=%d logScale=%d\n",
		v.paramName, v.params.LogN(), v.slots, v.params.MaxLevel(), int(v.params.LogDefaultScale()))

	return nil
}

// -------------------- constants bin (same layout as HEaaN backend) --------------------

func (v *LATTIGO_HEVM) loadConstants(name string) error {
	f, err := os.Open(name)
	if err != nil {
		return err
	}
	defer f.Close()

	var b8 [8]byte
	if _, err := io.ReadFull(f, b8[:]); err != nil {
		return err
	}
	n := int64(binary.LittleEndian.Uint64(b8[:]))
	if n < 0 || n > 1_000_000 {
		return fmt.Errorf("constants len invalid: %d", n)
	}

	out := make([][]float64, n)
	for i := int64(0); i < n; i++ {
		if _, err := io.ReadFull(f, b8[:]); err != nil {
			return err
		}
		veclen := int64(binary.LittleEndian.Uint64(b8[:]))
		if veclen < 0 || veclen > 1_000_000 {
			return fmt.Errorf("constants veclen invalid: %d", veclen)
		}
		raw := make([]byte, veclen*8)
		if _, err := io.ReadFull(f, raw); err != nil {
			return err
		}
		vec := make([]float64, veclen)
		for j := int64(0); j < veclen; j++ {
			u := binary.LittleEndian.Uint64(raw[j*8 : j*8+8])
			vec[j] = math.Float64frombits(u)
		}
		out[i] = vec
	}
	v.buffer = out
	return nil
}

// -------------------- HEVM binary parsing --------------------

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

func (v *LATTIGO_HEVM) loadHeader(r io.Reader) error {
	magic, err := readU32(r)
	if err != nil {
		return err
	}
	hsize, err := readU32(r)
	if err != nil {
		return err
	}
	argLen, err := readU64(r)
	if err != nil {
		return err
	}
	resLen, err := readU64(r)
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

	cbl, err := readU64(r)
	if err != nil {
		return err
	}
	numOps, err := readU64(r)
	if err != nil {
		return err
	}
	numCtxt, err := readU64(r)
	if err != nil {
		return err
	}
	numPtxt, err := readU64(r)
	if err != nil {
		return err
	}
	initLvl, err := readU64(r)
	if err != nil {
		return err
	}
	reserved, err := readU64(r)
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

	v.arg_scale = make([]uint64, aLen)
	v.arg_level = make([]uint64, aLen)
	v.res_scale = make([]uint64, rLen)
	v.res_level = make([]uint64, rLen)
	v.res_dst = make([]uint64, rLen)

	readArr := func(dst []uint64) error {
		for i := range dst {
			x, err := readU64(r)
			if err != nil {
				return err
			}
			dst[i] = x
		}
		return nil
	}
	if err := readArr(v.arg_scale); err != nil {
		return err
	}
	if err := readArr(v.arg_level); err != nil {
		return err
	}
	if err := readArr(v.res_scale); err != nil {
		return err
	}
	if err := readArr(v.res_level); err != nil {
		return err
	}
	if err := readArr(v.res_dst); err != nil {
		return err
	}

	// Allocate buffers similarly to HEaaN backend:
	// ciphers >= arg_len + res_len, plus any extra num_ctxt_buffer if present.
	ctN := int(numCtxt)
	ptN := int(numPtxt)
	minCt := aLen + rLen
	if ctN < minCt {
		ctN = minCt
	}
	v.ensureBuffers(ctN, max(1, ptN))
	v.ensureMsgBuffers(max(1, ptN))

	// reset scales bookkeeping
	for i := 0; i < len(v.scalec); i++ {
		v.scalec[i] = float64(v.params.LogDefaultScale())
	}
	for i := 0; i < len(v.scalep); i++ {
		v.scalep[i] = float64(v.params.LogDefaultScale())
		v.levelp[i] = uint64(v.params.MaxLevel())
	}
	return nil
}

func (v *LATTIGO_HEVM) resetResDst() {
	// Match HEaaN behavior: res_dst[i] = i + arg_length
	aLen := int(v.header.config_header.arg_length)
	for i := 0; i < int(v.header.config_header.res_length); i++ {
		v.res_dst[i] = uint64(i + aLen)
	}
}

func (v *LATTIGO_HEVM) loadHEVM(name string) error {
	f, err := os.Open(name)
	if err != nil {
		return err
	}
	defer f.Close()

	if err := v.loadHeader(f); err != nil {
		return err
	}

	numOps := int(v.config.num_operations)
	v.ops = make([]C.HEVMOperation, numOps)
	for i := 0; i < numOps; i++ {
		op, err := readOp(f)
		if err != nil {
			return err
		}
		v.ops[i] = op
	}
	return nil
}

// -------------------- encode path --------------------

func (v *LATTIGO_HEVM) to_msg(dst int16, src []float64) {
	v.ensureMsgBuffers(int(dst) + 1)
	// store raw real values; encoding happens online.
	cp := make([]float64, len(src))
	copy(cp, src)
	v.msgs[int(dst)] = cp
}

func (v *LATTIGO_HEVM) encode_online(dst int16) {
	if len(v.plains) == 0 {
		v.ensureBuffers(len(v.ciphers), 1)
	}
	if len(v.msgs) <= int(dst) {
		return
	}
	level := int(v.levelp[int(dst)])
	log2Scale := int(v.scalep[int(dst)])

	if level > v.params.MaxLevel() {
		level = v.params.MaxLevel()
	}
	if level < 0 {
		level = v.params.MaxLevel()
	}
	if log2Scale <= 0 {
		log2Scale = int(v.params.LogDefaultScale())
	}

	src := v.msgs[int(dst)]
	pt := ckks.NewPlaintext(v.params, level)
	pt.Scale = rlwe.NewScale(math.Exp2(float64(log2Scale)))

	vec := make([]complex128, v.slots)
	if len(src) > 0 {
		for i := 0; i < v.slots; i++ {
			vec[i] = complex(src[i%len(src)], 0)
		}
	}
	v.encoder.Encode(vec, pt)
	v.plains[0] = *pt
}

func (v *LATTIGO_HEVM) encode_internal(dst *rlwe.Plaintext, src []float64, level int, log2Scale int) {
	if level > v.params.MaxLevel() {
		level = v.params.MaxLevel()
	}
	if level < 0 {
		level = v.params.MaxLevel()
	}
	if log2Scale <= 0 {
		log2Scale = int(v.params.LogDefaultScale())
	}

	pt := ckks.NewPlaintext(v.params, level)
	pt.Scale = rlwe.NewScale(math.Exp2(float64(log2Scale)))

	vec := make([]complex128, v.slots)
	if len(src) > 0 {
		for i := 0; i < v.slots; i++ {
			vec[i] = complex(src[i%len(src)], 0)
		}
	}
	v.encoder.Encode(vec, pt)
	*dst = *pt
}

func (v *LATTIGO_HEVM) preprocess() {
	identity := []float64{1.0}

	for i := range v.ops {
		op := v.ops[i]
		if int(op.opcode) != OP_ENCODE {
			continue
		}
		dst := int16(op.dst)
		lhs := int(op.lhs)
		level := int(uint16(op.rhs) >> 10)
		log2Scale := int(uint16(op.rhs) & 0x3FF)

		var src []float64
		if lhs == int(uint16(^uint16(0))) { // (unsigned short)-1 == 65535
			src = identity
		} else if lhs >= 0 && lhs < len(v.buffer) {
			src = v.buffer[lhs]
		} else {
			src = nil
		}

		v.ensureBuffers(len(v.ciphers), int(dst)+1)
		v.ensureMsgBuffers(int(dst) + 1)

		v.levelp[int(dst)] = uint64(level)
		v.scalep[int(dst)] = float64(log2Scale)

		if v.preencode {
			v.encode_internal(&v.plains[int(dst)], src, level, log2Scale)
		} else {
			v.to_msg(dst, src)
		}
	}
}

// -------------------- op handlers (match HEaaN method names) --------------------

func (v *LATTIGO_HEVM) encode(dst, src int16, level int8, scale int8) {
	// no-op: preprocess handles encoding (same spirit as HEaaN backend).
	_ = dst
	_ = src
	_ = level
	_ = scale
}

func (v *LATTIGO_HEVM) rotate(dst, src, offset int16) {
	k := int(int16(offset))
	if v.debug {
		fmt.Printf("[rotate] scalec[%d]=%.0f k=%d level=%d\n", src, v.scalec[src], k, v.ciphers[src].Level())
	}
	v.clampCiphertextToMaxLevel(&v.ciphers[src])
	v.ensureRotationKey(k)
	_ = v.evaluator.Rotate(&v.ciphers[src], k, &v.ciphers[dst])
	v.clampCiphertextToMaxLevel(&v.ciphers[dst])
	v.scalec[dst] = v.scalec[src]
}

func (v *LATTIGO_HEVM) negate(dst, src int16) {
	if v.debug {
		fmt.Printf("[negate] scalec[%d]=%.0f level=%d\n", src, v.scalec[src], v.ciphers[src].Level())
	}
	v.clampCiphertextToMaxLevel(&v.ciphers[src])
	// use Mul by scalar to avoid relying on optional Neg API
	_ = v.evaluator.Mul(&v.ciphers[src], -1.0, &v.ciphers[dst])
	v.clampCiphertextToMaxLevel(&v.ciphers[dst])
	v.scalec[dst] = v.scalec[src]
}

func (v *LATTIGO_HEVM) rescale(dst, src int16) {
	if v.debug {
		fmt.Printf("[rescale] scalec[%d]=%.0f level=%d\n", src, v.scalec[src], v.ciphers[src].Level())
	}
	v.clampCiphertextToMaxLevel(&v.ciphers[src])
	_ = v.evaluator.Rescale(&v.ciphers[src], &v.ciphers[dst])
	v.clampCiphertextToMaxLevel(&v.ciphers[dst])
	v.scalec[dst] = log2ScaleOf(&v.ciphers[dst], v.params.LogDefaultScale())
}

func (v *LATTIGO_HEVM) modswitch(dst, src, downFactor int16) {
	df := int(downFactor)
	if v.debug {
		fmt.Printf("[modswitch] scalec[%d]=%.0f down=%d level=%d\n", src, v.scalec[src], df, v.ciphers[src].Level())
	}
	v.ciphers[dst] = v.ciphers[src]
	v.clampCiphertextToMaxLevel(&v.ciphers[dst])
	if df > 0 {
		v.evaluator.DropLevel(&v.ciphers[dst], df)
	}
	v.clampCiphertextToMaxLevel(&v.ciphers[dst])
	v.scalec[dst] = log2ScaleOf(&v.ciphers[dst], v.params.LogDefaultScale())
}

func (v *LATTIGO_HEVM) upscale(dst, src, upFactor int16) {
	_ = dst
	_ = src
	_ = upFactor
	fmt.Printf("[LATTIGO_HEVM][ERR] upscale: not supported\n")
}

func (v *LATTIGO_HEVM) addcc(dst, lhs, rhs int16) {
	if v.debug {
		fmt.Printf("[addcc] scalec[%d]=%.0f scalec[%d]=%.0f\n", lhs, v.scalec[lhs], rhs, v.scalec[rhs])
	}

	a := v.ciphers[lhs].CopyNew()
	b := v.ciphers[rhs].CopyNew()
	v.clampCiphertextToMaxLevel(a)
	v.clampCiphertextToMaxLevel(b)

	minL := a.Level()
	if b.Level() < minL {
		minL = b.Level()
	}
	if a.Level() > minL {
		v.evaluator.DropLevel(a, a.Level()-minL)
	}
	if b.Level() > minL {
		v.evaluator.DropLevel(b, b.Level()-minL)
	}
	// align scale
	b.Scale = a.Scale

	tmp := ckks.NewCiphertext(v.params, 1, minL)
	tmp.Scale = a.Scale
	_ = v.evaluator.Add(a, b, tmp)

	v.clampCiphertextToMaxLevel(tmp)
	v.ciphers[dst] = *tmp
	v.scalec[dst] = log2ScaleOf(&v.ciphers[dst], v.params.LogDefaultScale())
}

func (v *LATTIGO_HEVM) addcp(dst, lhs, rhs int16) {
	if v.debug {
		fmt.Printf("[addcp] scalec[%d]=%.0f scalep[%d]=%.0f\n", lhs, v.scalec[lhs], rhs, v.scalep[rhs])
	}

	if v.preencode {
		// plains[rhs]
		pt := &v.plains[rhs]
		a := v.ciphers[lhs].CopyNew()
		v.clampCiphertextToMaxLevel(a)
		if a.Level() > pt.Level() {
			v.evaluator.DropLevel(a, a.Level()-pt.Level())
		}

		tmp := ckks.NewCiphertext(v.params, 1, a.Level())
		tmp.Scale = a.Scale
		_ = v.evaluator.Add(a, pt, tmp)

		v.clampCiphertextToMaxLevel(tmp)
		v.ciphers[dst] = *tmp
		v.scalec[dst] = log2ScaleOf(&v.ciphers[dst], v.params.LogDefaultScale())
		return
	}

	// online encode into plains[0]
	v.encode_online(rhs)
	pt := &v.plains[0]

	a := v.ciphers[lhs].CopyNew()
	v.clampCiphertextToMaxLevel(a)
	if a.Level() > pt.Level() {
		v.evaluator.DropLevel(a, a.Level()-pt.Level())
	}

	tmp := ckks.NewCiphertext(v.params, 1, a.Level())
	tmp.Scale = a.Scale
	_ = v.evaluator.Add(a, pt, tmp)

	v.clampCiphertextToMaxLevel(tmp)
	v.ciphers[dst] = *tmp
	v.scalec[dst] = log2ScaleOf(&v.ciphers[dst], v.params.LogDefaultScale())
}

func (v *LATTIGO_HEVM) mulcc(dst, lhs, rhs int16) {
	if v.debug {
		fmt.Printf("[mulcc] scalec[%d]=%.0f scalec[%d]=%.0f\n", lhs, v.scalec[lhs], rhs, v.scalec[rhs])
	}

	a := v.ciphers[lhs].CopyNew()
	b := v.ciphers[rhs].CopyNew()
	v.clampCiphertextToMaxLevel(a)
	v.clampCiphertextToMaxLevel(b)

	minL := a.Level()
	if b.Level() < minL {
		minL = b.Level()
	}
	if a.Level() > minL {
		v.evaluator.DropLevel(a, a.Level()-minL)
	}
	if b.Level() > minL {
		v.evaluator.DropLevel(b, b.Level()-minL)
	}

	tmp := ckks.NewCiphertext(v.params, 1, minL)
	_ = v.evaluator.MulRelin(a, b, tmp) // "without rescale" semantics: rescale is a separate op

	v.clampCiphertextToMaxLevel(tmp)
	v.ciphers[dst] = *tmp
	v.scalec[dst] = log2ScaleOf(&v.ciphers[dst], v.params.LogDefaultScale())
}

func (v *LATTIGO_HEVM) mulcp(dst, lhs, rhs int16) {
	if v.debug {
		fmt.Printf("[mulcp] scalec[%d]=%.0f scalep[%d]=%.0f ctLevel=%d ptLevel=%d\n",
			lhs, v.scalec[lhs], rhs, v.scalep[rhs], v.ciphers[lhs].Level(), v.levelp[rhs])
	}

	if v.preencode {
		v.clampCiphertextToMaxLevel(&v.ciphers[lhs])
		_ = v.evaluator.Mul(&v.ciphers[lhs], &v.plains[rhs], &v.ciphers[dst])
		v.clampCiphertextToMaxLevel(&v.ciphers[dst])
		v.scalec[dst] = log2ScaleOf(&v.ciphers[dst], v.params.LogDefaultScale())
		return
	}

	v.encode_online(rhs)
	v.clampCiphertextToMaxLevel(&v.ciphers[lhs])
	_ = v.evaluator.Mul(&v.ciphers[lhs], &v.plains[0], &v.ciphers[dst])
	v.clampCiphertextToMaxLevel(&v.ciphers[dst])
	v.scalec[dst] = log2ScaleOf(&v.ciphers[dst], v.params.LogDefaultScale())
}

func (v *LATTIGO_HEVM) bootstrap(dst int16, src int64, targetLevel uint64) {
	s := int(src)
	if v.debug {
		fmt.Printf("[bootstrap] scalec[%d]=%.0f level=%d target=%d\n", s, v.scalec[s], v.ciphers[s].Level(), targetLevel)
	}

	v.clampCiphertextToMaxLevel(&v.ciphers[s])

	if err := v.ensureBoot(); err != nil {
		fmt.Printf("[LATTIGO_HEVM][BOOT][ERR] %v\n", err)
		return
	}

	t0 := time.Now()
	booted, err := v.bootEval.Bootstrap(&v.ciphers[s])
	if err != nil {
		fmt.Printf("[LATTIGO_HEVM][BOOT][ERR] Bootstrap: %v\n", err)
		return
	}
	dur := time.Since(t0)
	v.boot_time += uint64(dur / time.Microsecond)
	v.boot_cnt++

	v.clampCiphertextToMaxLevel(booted)
	v.ciphers[int(dst)] = *booted

	// optional target level drop
	if int(targetLevel) >= 0 && int(targetLevel) <= v.params.MaxLevel() {
		if v.ciphers[int(dst)].Level() > int(targetLevel) {
			v.evaluator.DropLevel(&v.ciphers[int(dst)], v.ciphers[int(dst)].Level()-int(targetLevel))
		}
	}
	v.clampCiphertextToMaxLevel(&v.ciphers[int(dst)])
	v.scalec[int(dst)] = log2ScaleOf(&v.ciphers[int(dst)], v.params.LogDefaultScale())
}

// -------------------- run dispatcher --------------------

func (v *LATTIGO_HEVM) run() {
	iword := int((uint64(v.header.hevm_header_size) + uint64(v.config.config_body_length)) / 8)
	jop := 0

	for idx := range v.ops {
		op := v.ops[idx]

		if v.debug {
			fmt.Printf("\n%o %d\n", iword, jop)
			fmt.Printf("opcode [%d], dst [%d], lhs [%d], rhs [%d]\n", op.opcode, op.dst, op.lhs, op.rhs)
			iword++
			jop++
		}

		switch int(op.opcode) {
		case OP_ENCODE:
			v.encode(int16(op.dst), int16(op.lhs), int8(op.rhs>>10), int8(op.rhs&0x3FF))

		case OP_ROTATEC:
			v.rotate(int16(op.dst), int16(op.lhs), int16(op.rhs))

		case OP_NEGATEC:
			v.negate(int16(op.dst), int16(op.lhs))

		case OP_RESCALEC:
			v.rescale(int16(op.dst), int16(op.lhs))

		case OP_MODSWC:
			v.modswitch(int16(op.dst), int16(op.lhs), int16(op.rhs))

		case OP_UPSCALEC:
			v.upscale(int16(op.dst), int16(op.lhs), int16(op.rhs))

		case OP_ADDCC:
			v.addcc(int16(op.dst), int16(op.lhs), int16(op.rhs))

		case OP_ADDCP:
			v.addcp(int16(op.dst), int16(op.lhs), int16(op.rhs))

		case OP_MULCC:
			v.mulcc(int16(op.dst), int16(op.lhs), int16(op.rhs))

		case OP_MULCP:
			v.mulcp(int16(op.dst), int16(op.lhs), int16(op.rhs))

		case OP_BOOT:
			v.bootstrap(int16(op.dst), int64(op.lhs), uint64(op.rhs))

		default:
			// ignore
		}
	}
}

// -------------------- encrypt/decrypt (match HEaaN exported ABI names) --------------------

func (v *LATTIGO_HEVM) encrypt(i int64, dat *C.double, length int) {
	if dat == nil || length <= 0 {
		return
	}
	id := int(i)
	if id < 0 {
		return
	}
	v.ensureBuffers(id+1, len(v.plains))
	if v.encoder == nil || v.encryptor == nil {
		_ = v.setupFromPresetNewKeys()
	}

	level := v.params.MaxLevel()
	log2Scale := int(v.params.LogDefaultScale())
	if id < len(v.arg_level) {
		if int(v.arg_level[id]) <= v.params.MaxLevel() {
			level = int(v.arg_level[id])
		}
	}
	if id < len(v.arg_scale) {
		if v.arg_scale[id] > 0 {
			log2Scale = int(v.arg_scale[id])
		}
	}

	src := (*[1 << 30]C.double)(unsafe.Pointer(dat))[:length:length]
	vals := make([]float64, length)
	for k := 0; k < length; k++ {
		vals[k] = float64(src[k])
	}

	var pt rlwe.Plaintext
	v.encode_internal(&pt, vals, level, log2Scale)

	ct := ckks.NewCiphertext(v.params, 1, level)
	v.encryptor.Encrypt(&pt, ct)

	v.clampCiphertextToMaxLevel(ct)
	v.ciphers[id] = *ct
	v.scalec[id] = log2ScaleOf(&v.ciphers[id], v.params.LogDefaultScale())
}

func (v *LATTIGO_HEVM) decrypt(i int64, dat *C.double) {
	if dat == nil {
		return
	}
	id := int(i)
	if id < 0 || id >= len(v.ciphers) || v.decryptor == nil || v.encoder == nil {
		return
	}

	v.clampCiphertextToMaxLevel(&v.ciphers[id])

	pt := ckks.NewPlaintext(v.params, v.ciphers[id].Level())
	v.decryptor.Decrypt(&v.ciphers[id], pt)

	// decode at ciphertext scale
	pt.Scale = v.ciphers[id].Scale

	vec := make([]complex128, v.slots)
	v.encoder.Decode(pt, vec)

	out := (*[1 << 30]C.double)(unsafe.Pointer(dat))[:v.slots:v.slots]
	for k := 0; k < v.slots; k++ {
		out[k] = C.double(real(vec[k]))
	}
}

// -------------------- Ctxt handle exchange (communication) --------------------

type CtxtBox struct{ CT rlwe.Ciphertext }

func exportCiphertextHandle(ct *rlwe.Ciphertext) unsafe.Pointer {
	if ct == nil {
		return nil
	}
	box := &CtxtBox{CT: *ct} // copy
	return newHandlePtr(box)
}

func importCiphertextHandle(p unsafe.Pointer) *rlwe.Ciphertext {
	hd, ok := loadHandlePtr(p)
	if !ok {
		return nil
	}
	box, ok := hd.Value().(*CtxtBox)
	if !ok || box == nil {
		return nil
	}
	return &box.CT
}

// -------------------- exported C ABI (names match HEaaN backend) --------------------

//export initFullVM
func initFullVM(dir *C.char, device C._Bool) unsafe.Pointer {
	strdir := C.GoString(dir)
	vm := &LATTIGO_HEVM{
		debug: false,
		togpu: bool(device),
	}
	// Try loading context from dir; fallback to new keys.
	if strdir != "" {
		if err := vm.loadLattigo(strdir); err != nil {
			_ = vm.setupFromPresetNewKeys()
		}
	} else {
		_ = vm.setupFromPresetNewKeys()
	}
	fmt.Printf("[LATTIGO_HEVM] initFullVM dir=%q device=%v\n", strdir, bool(device))
	return newHandlePtr(vm)
}

//export initClientVM
func initClientVM(dir *C.char) unsafe.Pointer {
	strdir := C.GoString(dir)
	vm := &LATTIGO_HEVM{debug: false, togpu: false}
	if strdir != "" {
		if err := vm.loadLattigo(strdir); err != nil {
			_ = vm.setupFromPresetNewKeys()
		}
	} else {
		_ = vm.setupFromPresetNewKeys()
	}
	fmt.Printf("[LATTIGO_HEVM] initClientVM dir=%q\n", strdir)
	return newHandlePtr(vm)
}

//export initServerVM
func initServerVM(dir *C.char) unsafe.Pointer {
	strdir := C.GoString(dir)
	vm := &LATTIGO_HEVM{debug: false, togpu: true}
	if strdir != "" {
		if err := vm.loadLattigo(strdir); err != nil {
			_ = vm.setupFromPresetNewKeys()
		}
	} else {
		_ = vm.setupFromPresetNewKeys()
	}
	fmt.Printf("[LATTIGO_HEVM] initServerVM dir=%q\n", strdir)
	return newHandlePtr(vm)
}

//export freeVM
func freeVM(vm unsafe.Pointer) {
	freeHandlePtr(vm)
}

//export create_context
func create_context(dir *C.char) {
	strdir := C.GoString(dir)
	if strdir == "" {
		fmt.Printf("[LATTIGO_HEVM][ERR] create_context: empty dir\n")
		return
	}
	_ = os.MkdirAll(strdir, 0o755)

	name, preset := bootPreset()

	residual, err := ckks.NewParametersFromLiteral(preset.CKKS)
	if err != nil {
		fmt.Printf("[LATTIGO_HEVM][ERR] create_context residual params: %v\n", err)
		return
	}
	btp, err := bootstrapping.NewParametersFromLiteral(residual, preset.BOOT)
	if err != nil {
		fmt.Printf("[LATTIGO_HEVM][ERR] create_context boot params: %v\n", err)
		return
	}
	vmParams := btp.BootstrappingParameters

	kgen := ckks.NewKeyGenerator(vmParams)
	sk, pk := kgen.GenKeyPairNew()
	rlk := kgen.GenRelinearizationKeyNew(sk)

	cx := &Context{
		ParamName: name,
		ParamsLit: preset.CKKS,
		BootLit:   preset.BOOT,
		SK:        *sk,
		PK:        *pk,
		RLK:       *rlk,
	}

	p := filepath.Join(strdir, contextGobName)
	f, err := os.Create(p)
	if err != nil {
		fmt.Printf("[LATTIGO_HEVM][ERR] create_context create file: %v\n", err)
		return
	}
	defer f.Close()

	if err := gob.NewEncoder(f).Encode(cx); err != nil {
		fmt.Printf("[LATTIGO_HEVM][ERR] create_context encode: %v\n", err)
		return
	}

	fmt.Printf("[LATTIGO_HEVM] create_context preset=%s dir=%q saved=%q\n", name, strdir, p)
}

//export load
func load(vm unsafe.Pointer, constant *C.char, vmfile *C.char) {
	hd, ok := loadHandlePtr(vm)
	if !ok {
		return
	}
	hevm, ok := hd.Value().(*LATTIGO_HEVM)
	if !ok || hevm == nil {
		return
	}

	cpath := C.GoString(constant)
	hpath := C.GoString(vmfile)
	fmt.Printf("[LATTIGO_HEVM] load const=%q hevm=%q\n", cpath, hpath)

	if cpath != "" {
		if err := hevm.loadConstants(cpath); err != nil {
			fmt.Printf("[LATTIGO_HEVM][ERR] loadConstants: %v\n", err)
			return
		}
	}
	if hevm.encoder == nil || hevm.evaluator == nil || hevm.params.LogN() == 0 {
		_ = hevm.setupFromPresetNewKeys()
	}
	if err := hevm.loadHEVM(hpath); err != nil {
		fmt.Printf("[LATTIGO_HEVM][ERR] loadHEVM: %v\n", err)
		return
	}

	fmt.Printf("[LATTIGO_HEVM] preset=%s logN=%d slots=%d maxLevel=%d logScale=%d\n",
		hevm.paramName, hevm.params.LogN(), hevm.slots, hevm.params.MaxLevel(), int(hevm.params.LogDefaultScale()))
}

//export loadClient
func loadClient(vm unsafe.Pointer, is unsafe.Pointer) {
	// HEaaN: loads header from std::istream and resets res_dst.
	// Here: if is != NULL treat it as (char*) path to a .hevm file and load ONLY header arrays, then resetResDst.
	hd, ok := loadHandlePtr(vm)
	if !ok {
		return
	}
	hevm, ok := hd.Value().(*LATTIGO_HEVM)
	if !ok || hevm == nil {
		return
	}
	if is == nil {
		return
	}

	path := C.GoString((*C.char)(is))
	if path == "" {
		return
	}
	if hevm.encoder == nil || hevm.evaluator == nil || hevm.params.LogN() == 0 {
		_ = hevm.setupFromPresetNewKeys()
	}

	f, err := os.Open(path)
	if err != nil {
		fmt.Printf("[LATTIGO_HEVM][ERR] loadClient open: %v\n", err)
		return
	}
	defer f.Close()

	if err := hevm.loadHeader(f); err != nil {
		fmt.Printf("[LATTIGO_HEVM][ERR] loadClient loadHeader: %v\n", err)
		return
	}
	hevm.resetResDst()
	fmt.Printf("[LATTIGO_HEVM] loadClient header-only from %q (argLen=%d resLen=%d)\n",
		path, hevm.header.config_header.arg_length, hevm.header.config_header.res_length)
}

//export encrypt
func encrypt(vm unsafe.Pointer, i C.int64_t, dat *C.double, length C.int) {
	hd, ok := loadHandlePtr(vm)
	if !ok {
		return
	}
	hevm, ok := hd.Value().(*LATTIGO_HEVM)
	if !ok || hevm == nil {
		return
	}
	hevm.encrypt(int64(i), dat, int(length))
}

//export decrypt
func decrypt(vm unsafe.Pointer, i C.int64_t, dat *C.double) {
	hd, ok := loadHandlePtr(vm)
	if !ok {
		return
	}
	hevm, ok := hd.Value().(*LATTIGO_HEVM)
	if !ok || hevm == nil {
		return
	}
	hevm.decrypt(int64(i), dat)
}

//export decrypt_result
func decrypt_result(vm unsafe.Pointer, i C.int64_t, dat *C.double) {
	hd, ok := loadHandlePtr(vm)
	if !ok {
		return
	}
	hevm, ok := hd.Value().(*LATTIGO_HEVM)
	if !ok || hevm == nil {
		return
	}
	ii := int64(i)
	if ii < 0 || ii >= int64(len(hevm.res_dst)) {
		return
	}
	hevm.decrypt(int64(hevm.res_dst[ii]), dat)
}

//export getResIdx
func getResIdx(vm unsafe.Pointer, i C.int64_t) C.int64_t {
	hd, ok := loadHandlePtr(vm)
	if !ok {
		return i
	}
	hevm, ok := hd.Value().(*LATTIGO_HEVM)
	if !ok || hevm == nil {
		return i
	}
	ii := int64(i)
	if ii < 0 || ii >= int64(len(hevm.res_dst)) {
		return i
	}
	return C.int64_t(hevm.res_dst[ii])
}

//export getCtxt
func getCtxt(vm unsafe.Pointer, id C.int64_t) unsafe.Pointer {
	hd, ok := loadHandlePtr(vm)
	if !ok {
		return nil
	}
	hevm, ok := hd.Value().(*LATTIGO_HEVM)
	if !ok || hevm == nil {
		return nil
	}
	ii := int(id)
	if ii < 0 || ii >= len(hevm.ciphers) {
		return nil
	}
	return exportCiphertextHandle(&hevm.ciphers[ii])
}

//export setCtxt
func setCtxt(vm unsafe.Pointer, id C.int64_t, ct unsafe.Pointer) {
	hd, ok := loadHandlePtr(vm)
	if !ok {
		return
	}
	hevm, ok := hd.Value().(*LATTIGO_HEVM)
	if !ok || hevm == nil {
		return
	}
	ii := int(id)
	if ii < 0 || ii >= len(hevm.ciphers) {
		return
	}
	c := importCiphertextHandle(ct)
	if c == nil {
		return
	}
	hevm.ciphers[ii] = *c
	hevm.scalec[ii] = log2ScaleOf(&hevm.ciphers[ii], hevm.params.LogDefaultScale())
}

//export freeCtxt
func freeCtxt(ct unsafe.Pointer) {
	freeHandlePtr(ct)
}

//export preprocess
func preprocess(vm unsafe.Pointer) {
	hd, ok := loadHandlePtr(vm)
	if !ok {
		return
	}
	hevm, ok := hd.Value().(*LATTIGO_HEVM)
	if !ok || hevm == nil {
		return
	}
	hevm.preprocess()
}

//export run
func run(vm unsafe.Pointer) {
	hd, ok := loadHandlePtr(vm)
	if !ok {
		return
	}
	hevm, ok := hd.Value().(*LATTIGO_HEVM)
	if !ok || hevm == nil {
		return
	}
	hevm.run()
}

//export getArgLen
func getArgLen(vm unsafe.Pointer) C.int64_t {
	hd, ok := loadHandlePtr(vm)
	if !ok {
		return 0
	}
	hevm, ok := hd.Value().(*LATTIGO_HEVM)
	if !ok || hevm == nil {
		return 0
	}
	return C.int64_t(hevm.header.config_header.arg_length)
}

//export getResLen
func getResLen(vm unsafe.Pointer) C.int64_t {
	hd, ok := loadHandlePtr(vm)
	if !ok {
		return 0
	}
	hevm, ok := hd.Value().(*LATTIGO_HEVM)
	if !ok || hevm == nil {
		return 0
	}
	return C.int64_t(hevm.header.config_header.res_length)
}

//export setDebug
func setDebug(vm unsafe.Pointer, enable C._Bool) {
	hd, ok := loadHandlePtr(vm)
	if !ok {
		return
	}
	hevm, ok := hd.Value().(*LATTIGO_HEVM)
	if !ok || hevm == nil {
		return
	}
	hevm.debug = bool(enable)
}

//export setToGPU
func setToGPU(vm unsafe.Pointer, ongpu C._Bool) {
	hd, ok := loadHandlePtr(vm)
	if !ok {
		return
	}
	hevm, ok := hd.Value().(*LATTIGO_HEVM)
	if !ok || hevm == nil {
		return
	}
	hevm.togpu = bool(ongpu)
}

//export printMem
func printMem(vm unsafe.Pointer) {
	hd, ok := loadHandlePtr(vm)
	if !ok {
		return
	}
	hevm, ok := hd.Value().(*LATTIGO_HEVM)
	if !ok || hevm == nil {
		return
	}

	fmt.Printf("[LATTIGO_HEVM] printMem preset=%s logN=%d slots=%d maxLevel=%d boot_cnt=%d boot_time_us=%d\n",
		hevm.paramName, hevm.params.LogN(), hevm.slots, hevm.params.MaxLevel(), hevm.boot_cnt, hevm.boot_time)
}

func log2ScaleOf(ct *rlwe.Ciphertext, fallback float64) float64 {
	if ct == nil {
		return fallback
	}
	s := ct.Scale.Float64()
	if s <= 0 {
		return fallback
	}
	return math.Round(math.Log2(s))
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func main() {}
