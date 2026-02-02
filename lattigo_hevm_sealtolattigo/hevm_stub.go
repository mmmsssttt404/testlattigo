// hevm_stub_sealstyle_lattigo.go (Lattigo v6.1.1) - SEAL-like N15 backend, no real boot
//
// Build:
//   go build -buildmode=c-shared -o libLATTIGO_HEVM.so hevm_stub_sealstyle_lattigo.go
//
// Context files (directory):
//   parm.lattigo  (gob ParamDesc: only ckks.ParametersLiteral)
//   pub.lattigo   (PK MarshalBinary)
//   sec.lattigo   (SK MarshalBinary)         [full/client]
//   relin.lattigo (RLK MarshalBinary)        [full/server]
//   gal.lattigo   (pow2 galois key pack)     [full/server]
//
// HEVM:
// - OP_BOOT is pseudo-bootstrap: decrypt->decode->encode->encrypt (full/client only)

package main

/*
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
*/
import "C"

import (
	"bytes"
	"encoding/binary"
	"encoding/gob"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime/cgo"
	"unsafe"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

const (
	kMagicHEVM = uint32(0x4845564D) // "MVEH" little-endian

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

	ROLE_FULL   = 0
	ROLE_CLIENT = 1
	ROLE_SERVER = 2
)

// -------------------- files (*.lattigo) --------------------

const (
	fParm  = "parm.lattigo"
	fPub   = "pub.lattigo"
	fSec   = "sec.lattigo"
	fRelin = "relin.lattigo"
	fGal   = "gal.lattigo"
)

// -------------------- HEVM binary layout (pure Go structs) --------------------

// matches your hexdump:
// 0x00: u32 magic "MVEH"
// 0x04: u32 header_size (0x18)
// 0x08: u64 arg_len
// 0x10: u64 res_len
type hevmHeader struct {
	Magic      uint32
	HeaderSize uint32
	ArgLen     uint64
	ResLen     uint64
}

// ConfigBody is 6x u64 in your generators (your xxd shows 5th field init_level present,
// and a 6th reserved is expected right after).
type configBody struct {
	ConfigBodyLength uint64
	NumOperations    uint64
	NumCtxtBuffer    uint64
	NumPtxtBuffer    uint64
	InitLevel        uint64
	Reserved         uint64
	Extra            [4]uint64
}

type hevmOp struct {
	Opcode uint16
	Dst    uint16
	Lhs    uint16
	Rhs    uint16
}

// -------------------- on disk param desc --------------------

type ParamDesc struct {
	ParamName string
	ParamsLit ckks.ParametersLiteral
}

// gal.lattigo pack format:
// u64 count
// repeat count:
//
//	i32 step
//	u64 blen
//	[blen] bytes (MarshalBinary of rlwe.GaloisKey)
type galPack struct {
	steps []int
	keys  map[int]*rlwe.GaloisKey
}

// -------------------- VM --------------------

type VM struct {
	role  int
	debug bool
	togpu bool

	loadedOK bool

	// program
	hdr    hevmHeader
	cfg    configBody
	ops    []hevmOp
	consts [][]float64

	argScale []uint64
	argLevel []uint64
	resScale []uint64
	resLevel []uint64
	resDst   []uint64

	// ckks
	paramName string
	paramsLit ckks.ParametersLiteral
	params    ckks.Parameters
	slots     int

	encoder   *ckks.Encoder
	encryptor *rlwe.Encryptor
	decryptor *rlwe.Decryptor
	evaluator *ckks.Evaluator

	pk  *rlwe.PublicKey
	sk  *rlwe.SecretKey
	rlk *rlwe.RelinearizationKey

	galPow2  map[int]*rlwe.GaloisKey
	galSteps []int

	// buffers
	ciphers []rlwe.Ciphertext
	plains  []rlwe.Plaintext
}

// -------------------- handle helpers --------------------

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

type CtxtBox struct{ CT rlwe.Ciphertext }

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

// -------------------- io utils --------------------

func mustMkdir(dir string) error { return os.MkdirAll(dir, 0o755) }

func writeFile(path string, b []byte) error {
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}
func readFile(path string) ([]byte, error) { return os.ReadFile(path) }

func saveGob(path string, v any) error {
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(v); err != nil {
		return err
	}
	return writeFile(path, buf.Bytes())
}
func loadGob(path string, v any) error {
	b, err := readFile(path)
	if err != nil {
		return err
	}
	return gob.NewDecoder(bytes.NewReader(b)).Decode(v)
}
func saveBin(path string, b []byte) error { return writeFile(path, b) }
func loadBin(path string) ([]byte, error) { return readFile(path) }

// -------------------- params: SEAL-like N15, L14, 60-bit primes --------------------
// 注意：为了支持 MulRelin / Rotate，需要 LogP（这里给一个 60-bit P）。
func sealLikeParamsLiteralN15() ckks.ParametersLiteral {
	const N = 15
	const L = 14
	logQ := make([]int, 0, L)
	for i := 0; i < L; i++ {
		logQ = append(logQ, 60)
	}
	return ckks.ParametersLiteral{
		LogN:            N,
		LogQ:            logQ,
		LogP:            []int{60},
		LogDefaultScale: 40,
	}
}

func (v *VM) initParams(name string, lit ckks.ParametersLiteral) error {
	p, err := ckks.NewParametersFromLiteral(lit)
	if err != nil {
		return err
	}
	v.paramName = name
	v.paramsLit = lit
	v.params = p
	v.slots = p.MaxSlots()

	const encPrec = uint(53)
	v.encoder = ckks.NewEncoder(v.params, encPrec)
	return nil
}

// -------------------- gal pack --------------------

func saveGalPow2(path string, steps []int, keys map[int]*rlwe.GaloisKey) error {
	var buf bytes.Buffer
	_ = binary.Write(&buf, binary.LittleEndian, uint64(len(steps)))
	for _, s := range steps {
		gk := keys[s]
		_ = binary.Write(&buf, binary.LittleEndian, int32(s))
		if gk == nil {
			_ = binary.Write(&buf, binary.LittleEndian, uint64(0))
			continue
		}
		gb, err := gk.MarshalBinary()
		if err != nil {
			return err
		}
		_ = binary.Write(&buf, binary.LittleEndian, uint64(len(gb)))
		buf.Write(gb)
	}
	return writeFile(path, buf.Bytes())
}

func loadGalPow2(path string) (map[int]*rlwe.GaloisKey, []int, error) {
	b, err := loadBin(path)
	if err != nil {
		return nil, nil, err
	}
	r := bytes.NewReader(b)
	var cnt uint64
	if err := binary.Read(r, binary.LittleEndian, &cnt); err != nil {
		return nil, nil, err
	}
	keys := make(map[int]*rlwe.GaloisKey, int(cnt))
	steps := make([]int, 0, int(cnt))
	for i := uint64(0); i < cnt; i++ {
		var step int32
		var blen uint64
		if err := binary.Read(r, binary.LittleEndian, &step); err != nil {
			return nil, nil, err
		}
		if err := binary.Read(r, binary.LittleEndian, &blen); err != nil {
			return nil, nil, err
		}
		raw := make([]byte, blen)
		if blen > 0 {
			if _, err := io.ReadFull(r, raw); err != nil {
				return nil, nil, err
			}
			gk := new(rlwe.GaloisKey)
			if err := gk.UnmarshalBinary(raw); err != nil {
				return nil, nil, err
			}
			keys[int(step)] = gk
		} else {
			keys[int(step)] = nil
		}
		steps = append(steps, int(step))
	}
	return keys, steps, nil
}

// -------------------- context dir create/load --------------------

func createContextDir(dir string) error {
	if err := mustMkdir(dir); err != nil {
		return err
	}

	lit := sealLikeParamsLiteralN15()
	params, err := ckks.NewParametersFromLiteral(lit)
	if err != nil {
		return err
	}

	kgen := ckks.NewKeyGenerator(params)
	sk, pk := kgen.GenKeyPairNew()
	rlk := kgen.GenRelinearizationKeyNew(sk)

	// pow2 rotations: ±2^i for i in [0..logSlots-1]
	logSlots := int(math.Log2(float64(params.MaxSlots()))) // N-1 => 14
	els := make([]uint64, 0, 2*logSlots)
	steps := make([]int, 0, 2*logSlots)
	for i := 0; i < logSlots; i++ {
		s := 1 << i
		steps = append(steps, s)
		els = append(els, params.GaloisElementForRotation(s))
		steps = append(steps, -s)
		els = append(els, params.GaloisElementForRotation(-s))
	}
	gks := kgen.GenGaloisKeysNew(els, sk)

	keys := make(map[int]*rlwe.GaloisKey, len(steps))
	for i, s := range steps {
		if i < len(gks) && gks[i] != nil {
			keys[s] = gks[i]
		} else {
			keys[s] = nil
		}
	}

	desc := ParamDesc{ParamName: "SEAL-N15-L14", ParamsLit: lit}
	if err := saveGob(filepath.Join(dir, fParm), &desc); err != nil {
		return err
	}
	if b, err := pk.MarshalBinary(); err != nil {
		return err
	} else if err := saveBin(filepath.Join(dir, fPub), b); err != nil {
		return err
	}
	if b, err := sk.MarshalBinary(); err != nil {
		return err
	} else if err := saveBin(filepath.Join(dir, fSec), b); err != nil {
		return err
	}
	if b, err := rlk.MarshalBinary(); err != nil {
		return err
	} else if err := saveBin(filepath.Join(dir, fRelin), b); err != nil {
		return err
	}
	if err := saveGalPow2(filepath.Join(dir, fGal), steps, keys); err != nil {
		return err
	}
	return nil
}

func (v *VM) loadParm(dir string) error {
	var desc ParamDesc
	if err := loadGob(filepath.Join(dir, fParm), &desc); err != nil {
		return err
	}
	return v.initParams(desc.ParamName, desc.ParamsLit)
}

func (v *VM) buildEvaluator() {
	var gks []*rlwe.GaloisKey
	for _, s := range v.galSteps {
		if gk := v.galPow2[s]; gk != nil {
			gks = append(gks, gk)
		}
	}
	evk := rlwe.NewMemEvaluationKeySet(v.rlk, gks...)
	v.evaluator = ckks.NewEvaluator(v.params, evk)
}

func (v *VM) loadFull(dir string) error {
	if err := v.loadParm(dir); err != nil {
		return err
	}
	// pk
	{
		raw, err := loadBin(filepath.Join(dir, fPub))
		if err != nil {
			return err
		}
		pk := new(rlwe.PublicKey)
		if err := pk.UnmarshalBinary(raw); err != nil {
			return err
		}
		v.pk = pk
	}
	// sk
	{
		raw, err := loadBin(filepath.Join(dir, fSec))
		if err != nil {
			return err
		}
		sk := new(rlwe.SecretKey)
		if err := sk.UnmarshalBinary(raw); err != nil {
			return err
		}
		v.sk = sk
	}
	// rlk
	{
		raw, err := loadBin(filepath.Join(dir, fRelin))
		if err != nil {
			return err
		}
		rlk := new(rlwe.RelinearizationKey)
		if err := rlk.UnmarshalBinary(raw); err != nil {
			return err
		}
		v.rlk = rlk
	}
	// gal
	{
		m, steps, err := loadGalPow2(filepath.Join(dir, fGal))
		if err != nil {
			return err
		}
		v.galPow2 = m
		v.galSteps = steps
	}

	v.encryptor = ckks.NewEncryptor(v.params, v.pk)
	v.decryptor = ckks.NewDecryptor(v.params, v.sk)
	v.buildEvaluator()
	return nil
}

func (v *VM) loadClient(dir string) error {
	if err := v.loadParm(dir); err != nil {
		return err
	}
	// pk
	{
		raw, err := loadBin(filepath.Join(dir, fPub))
		if err != nil {
			return err
		}
		pk := new(rlwe.PublicKey)
		if err := pk.UnmarshalBinary(raw); err != nil {
			return err
		}
		v.pk = pk
	}
	// sk
	{
		raw, err := loadBin(filepath.Join(dir, fSec))
		if err != nil {
			return err
		}
		sk := new(rlwe.SecretKey)
		if err := sk.UnmarshalBinary(raw); err != nil {
			return err
		}
		v.sk = sk
	}
	v.encryptor = ckks.NewEncryptor(v.params, v.pk)
	v.decryptor = ckks.NewDecryptor(v.params, v.sk)
	return nil
}

func (v *VM) loadServer(dir string) error {
	if err := v.loadParm(dir); err != nil {
		return err
	}
	// rlk
	{
		raw, err := loadBin(filepath.Join(dir, fRelin))
		if err != nil {
			return err
		}
		rlk := new(rlwe.RelinearizationKey)
		if err := rlk.UnmarshalBinary(raw); err != nil {
			return err
		}
		v.rlk = rlk
	}
	// gal
	{
		m, steps, err := loadGalPow2(filepath.Join(dir, fGal))
		if err != nil {
			return err
		}
		v.galPow2 = m
		v.galSteps = steps
	}
	v.buildEvaluator()
	return nil
}

// -------------------- buffers --------------------

func (v *VM) ensureBuffers(ctN, ptN int) {
	if ctN < 1 {
		ctN = 1
	}
	if ptN < 1 {
		ptN = 1
	}
	if len(v.ciphers) < ctN {
		old := len(v.ciphers)
		v.ciphers = append(v.ciphers, make([]rlwe.Ciphertext, ctN-old)...)
		for i := old; i < ctN; i++ {
			v.ciphers[i] = *ckks.NewCiphertext(v.params, 1, v.params.MaxLevel())
		}
	}
	if len(v.plains) < ptN {
		old := len(v.plains)
		v.plains = append(v.plains, make([]rlwe.Plaintext, ptN-old)...)
		for i := old; i < ptN; i++ {
			v.plains[i] = *ckks.NewPlaintext(v.params, v.params.MaxLevel())
		}
	}
}

func (v *VM) clampCiphertextToMaxLevel(ct *rlwe.Ciphertext) {
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

func safeDropLevel(eval *ckks.Evaluator, ct *rlwe.Ciphertext, down int) {
	if eval == nil || ct == nil {
		return
	}
	if down <= 0 {
		return
	}
	if down > ct.Level() {
		// invalid, skip
		return
	}
	eval.DropLevel(ct, down)
}

// -------------------- constants (.cst) --------------------

func readConstantsBin(path string) ([][]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var len64 int64
	if err := binary.Read(f, binary.LittleEndian, &len64); err != nil {
		return nil, err
	}
	if len64 < 0 || len64 > 1_000_000 {
		return nil, fmt.Errorf("constants len invalid: %d", len64)
	}

	out := make([][]float64, len64)
	for i := int64(0); i < len64; i++ {
		var veclen int64
		if err := binary.Read(f, binary.LittleEndian, &veclen); err != nil {
			return nil, err
		}
		if veclen < 0 || veclen > 1_000_000 {
			return nil, fmt.Errorf("constants veclen invalid: %d", veclen)
		}
		vec := make([]float64, veclen)
		if err := binary.Read(f, binary.LittleEndian, vec); err != nil {
			return nil, err
		}
		out[i] = vec
	}
	return out, nil
}

// -------------------- HEVM loader (strict binary.Read) --------------------

func (v *VM) loadHEVM(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	var hdr hevmHeader
	if err := binary.Read(f, binary.LittleEndian, &hdr); err != nil {
		return err
	}
	if hdr.Magic != kMagicHEVM {
		return fmt.Errorf("bad magic: got 0x%x expect 0x%x", hdr.Magic, kMagicHEVM)
	}
	v.hdr = hdr

	var cfg configBody
	if err := binary.Read(f, binary.LittleEndian, &cfg); err != nil {
		return err
	}
	v.cfg = cfg

	aLen := int(hdr.ArgLen)
	rLen := int(hdr.ResLen)

	v.argScale = make([]uint64, aLen)
	v.argLevel = make([]uint64, aLen)
	v.resScale = make([]uint64, rLen)
	v.resLevel = make([]uint64, rLen)
	v.resDst = make([]uint64, rLen)

	readU64Arr := func(dst []uint64) error {
		if len(dst) == 0 {
			return nil
		}
		return binary.Read(f, binary.LittleEndian, dst)
	}
	if err := readU64Arr(v.argScale); err != nil {
		return err
	}
	if err := readU64Arr(v.argLevel); err != nil {
		return err
	}
	if err := readU64Arr(v.resScale); err != nil {
		return err
	}
	if err := readU64Arr(v.resLevel); err != nil {
		return err
	}
	if err := readU64Arr(v.resDst); err != nil {
		return err
	}

	numOps := int(cfg.NumOperations)
	if numOps < 0 || numOps > 10_000_000 {
		return fmt.Errorf("num_operations invalid: %d", numOps)
	}
	v.ops = make([]hevmOp, numOps)
	if numOps > 0 {
		if err := binary.Read(f, binary.LittleEndian, v.ops); err != nil {
			return err
		}
	}

	ctN := int(cfg.NumCtxtBuffer)
	ptN := int(cfg.NumPtxtBuffer)
	if ctN < aLen+rLen {
		ctN = aLen + rLen
	}
	v.ensureBuffers(ctN, ptN)

	return nil
}

// -------------------- encode helper --------------------

func (v *VM) encodeToPlainAt(dst *rlwe.Plaintext, values []float64, level int, log2Scale int) {
	if level < 0 {
		level = 0
	}
	if level > v.params.MaxLevel() {
		level = v.params.MaxLevel()
	}
	if log2Scale <= 0 {
		log2Scale = int(v.params.LogDefaultScale())
	}

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

// -------------------- rotate via pow2 decomposition --------------------

func (v *VM) rotateDecomp(dst, src int, k int) {
	if v.evaluator == nil {
		return
	}
	if k == 0 {
		v.ciphers[dst] = v.ciphers[src]
		return
	}
	slots := v.slots
	k = k % slots
	if k > slots/2 {
		k -= slots
	} else if k < -slots/2 {
		k += slots
	}

	cur := v.ciphers[src].CopyNew()
	tmp := ckks.NewCiphertext(v.params, 1, cur.Level())
	tmp.Scale = cur.Scale

	abs := k
	sign := 1
	if abs < 0 {
		sign = -1
		abs = -abs
	}

	for bit := 0; abs != 0; bit++ {
		if abs&1 == 1 {
			step := sign * (1 << bit)
			gk := v.galPow2[step]
			if gk == nil {
				*tmp = *cur
			} else {
				_ = v.evaluator.Rotate(cur, step, tmp)
			}
			cur = tmp.CopyNew()
		}
		abs >>= 1
	}

	v.ciphers[dst] = *cur
}

// -------------------- pseudo boot (decrypt->reencode->encrypt) --------------------

func (v *VM) pseudoBootstrap(dst int, src int, targetLevel int) {
	if v.decryptor == nil || v.encryptor == nil || v.encoder == nil {
		fmt.Printf("[LATTIGO_HEVM][BOOT][ERR] pseudo boot requires SK/PK (full/client only)\n")
		v.ciphers[dst] = v.ciphers[src]
		return
	}

	in := &v.ciphers[src]
	v.clampCiphertextToMaxLevel(in)

	pt := ckks.NewPlaintext(v.params, in.Level())
	v.decryptor.Decrypt(in, pt)
	pt.Scale = in.Scale

	vec := make([]complex128, v.slots)
	v.encoder.Decode(pt, vec)

	values := make([]float64, v.slots)
	for i := 0; i < v.slots; i++ {
		values[i] = real(vec[i])
	}

	log2Scale := int(v.params.LogDefaultScale())
	if s := in.Scale.Float64(); s > 0 {
		log2Scale = int(math.Round(math.Log2(s)))
	}

	lv := targetLevel
	if lv < 0 {
		lv = 0
	}
	if lv > v.params.MaxLevel() {
		lv = v.params.MaxLevel()
	}

	var pt2 rlwe.Plaintext
	v.encodeToPlainAt(&pt2, values, lv, log2Scale)

	ct := ckks.NewCiphertext(v.params, 1, lv)
	v.encryptor.Encrypt(&pt2, ct)
	v.ciphers[dst] = *ct
}

// -------------------- Exported C ABI --------------------

//export create_context
func create_context(dir *C.char) {
	d := C.GoString(dir)
	if err := createContextDir(d); err != nil {
		fmt.Printf("[LATTIGO_HEVM][ERR] create_context: %v\n", err)
	}
}

//export initFullVM
func initFullVM(dir *C.char, device C._Bool) C.uintptr_t {
	d := C.GoString(dir)
	vm := &VM{role: ROLE_FULL, togpu: bool(device)}
	if err := vm.loadFull(d); err != nil {
		fmt.Printf("[LATTIGO_HEVM][ERR] initFullVM loadFull: %v\n", err)
	}
	h := cgo.NewHandle(vm)
	return C.uintptr_t(h)
}

//export initClientVM
func initClientVM(dir *C.char) C.uintptr_t {
	d := C.GoString(dir)
	vm := &VM{role: ROLE_CLIENT}
	if err := vm.loadClient(d); err != nil {
		fmt.Printf("[LATTIGO_HEVM][ERR] initClientVM loadClient: %v\n", err)
	}
	h := cgo.NewHandle(vm)
	return C.uintptr_t(h)
}

//export initServerVM
func initServerVM(dir *C.char) C.uintptr_t {
	d := C.GoString(dir)
	vm := &VM{role: ROLE_SERVER}
	if err := vm.loadServer(d); err != nil {
		fmt.Printf("[LATTIGO_HEVM][ERR] initServerVM loadServer: %v\n", err)
	}
	h := cgo.NewHandle(vm)
	return C.uintptr_t(h)
}

//export freeVM
func freeVM(h C.uintptr_t) {
	if h != 0 {
		cgo.Handle(h).Delete()
	}
}

//export load
func load(h C.uintptr_t, constPath *C.char, hevmPath *C.char) {
	v := getVM(h)
	if v == nil {
		return
	}
	v.loadedOK = false

	cpath := C.GoString(constPath)
	hpath := C.GoString(hevmPath)

	consts, err := readConstantsBin(cpath)
	if err != nil {
		fmt.Printf("[LATTIGO_HEVM][ERR] read constants: %v\n", err)
		return
	}
	v.consts = consts

	if err := v.loadHEVM(hpath); err != nil {
		fmt.Printf("[LATTIGO_HEVM][ERR] load hevm: %v\n", err)
		return
	}

	v.loadedOK = true
}

//export loadClient
func loadClient(h C.uintptr_t, is unsafe.Pointer) {
	// 你们 python 传的是 bytes 指针，不是 istream*。
	// 为避免传错（把 .cst 传进来），这里只做“可选解析”：
	v := getVM(h)
	if v == nil || is == nil {
		return
	}
	path := C.GoString((*C.char)(is))
	if filepath.Ext(path) != ".hevm" {
		// ignore
		return
	}
	_ = v.loadHEVM(path)
	v.loadedOK = true
}

//export preprocess
func preprocess(h C.uintptr_t) {
	v := getVM(h)
	if v == nil || !v.loadedOK || v.encoder == nil {
		return
	}
	for _, op := range v.ops {
		if int(op.Opcode) != OP_ENCODE {
			continue
		}
		dst := int(op.Dst)
		lhs := int(op.Lhs)
		level := int(op.Rhs >> 10)
		log2Scale := int(op.Rhs & 0x03FF)

		if dst < 0 || dst >= len(v.plains) {
			continue
		}

		var src []float64
		if lhs == 65535 {
			src = []float64{1.0}
		} else if lhs >= 0 && lhs < len(v.consts) {
			src = v.consts[lhs]
		}
		v.encodeToPlainAt(&v.plains[dst], src, level, log2Scale)
	}
}

//export run
func run(h C.uintptr_t) {
	v := getVM(h)
	if v == nil || !v.loadedOK {
		return
	}
	if v.evaluator == nil && v.role != ROLE_CLIENT {
		// server/full should have evaluator
		// but don't panic
	}

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
		switch int(op.Opcode) {
		case OP_ENCODE:
			continue

		case OP_ROTATEC:
			dst := int(op.Dst)
			src := int(op.Lhs)
			k := int(int16(op.Rhs))
			if ctAt(dst) == nil || ctAt(src) == nil {
				continue
			}
			v.rotateDecomp(dst, src, k)

		case OP_NEGATEC:
			dst := int(op.Dst)
			src := int(op.Lhs)
			in, out := ctAt(src), ctAt(dst)
			if in == nil || out == nil || v.evaluator == nil {
				continue
			}
			v.clampCiphertextToMaxLevel(in)
			_ = v.evaluator.Mul(in, -1.0, out)
			v.clampCiphertextToMaxLevel(out)

		case OP_RESCALEC:
			dst := int(op.Dst)
			src := int(op.Lhs)
			in, out := ctAt(src), ctAt(dst)
			if in == nil || out == nil || v.evaluator == nil {
				continue
			}
			v.clampCiphertextToMaxLevel(in)
			_ = v.evaluator.Rescale(in, out)
			v.clampCiphertextToMaxLevel(out)

		case OP_MODSWC:
			dst := int(op.Dst)
			src := int(op.Lhs)
			down := int(op.Rhs)
			in, out := ctAt(src), ctAt(dst)
			if in == nil || out == nil || v.evaluator == nil {
				continue
			}
			*out = *in
			v.clampCiphertextToMaxLevel(out)
			safeDropLevel(v.evaluator, out, down)
			v.clampCiphertextToMaxLevel(out)

		case OP_UPSCALEC:
			continue

		case OP_ADDCC:
			dst := int(op.Dst)
			l := int(op.Lhs)
			r := int(op.Rhs)
			in0, in1, out := ctAt(l), ctAt(r), ctAt(dst)
			if in0 == nil || in1 == nil || out == nil || v.evaluator == nil {
				continue
			}
			a := in0.CopyNew()
			b := in1.CopyNew()
			v.clampCiphertextToMaxLevel(a)
			v.clampCiphertextToMaxLevel(b)

			minL := a.Level()
			if b.Level() < minL {
				minL = b.Level()
			}
			safeDropLevel(v.evaluator, a, a.Level()-minL)
			safeDropLevel(v.evaluator, b, b.Level()-minL)
			b.Scale = a.Scale

			tmp := ckks.NewCiphertext(v.params, 1, minL)
			tmp.Scale = a.Scale
			_ = v.evaluator.Add(a, b, tmp)
			*out = *tmp

		case OP_ADDCP:
			dst := int(op.Dst)
			l := int(op.Lhs)
			p := int(op.Rhs)
			in0, pt, out := ctAt(l), ptAt(p), ctAt(dst)
			if in0 == nil || pt == nil || out == nil || v.evaluator == nil {
				continue
			}
			a := in0.CopyNew()
			v.clampCiphertextToMaxLevel(a)
			if a.Level() > pt.Level() {
				safeDropLevel(v.evaluator, a, a.Level()-pt.Level())
			}
			tmp := ckks.NewCiphertext(v.params, 1, a.Level())
			tmp.Scale = a.Scale
			_ = v.evaluator.Add(a, pt, tmp)
			*out = *tmp

		case OP_MULCC:
			dst := int(op.Dst)
			l := int(op.Lhs)
			r := int(op.Rhs)
			in0, in1, out := ctAt(l), ctAt(r), ctAt(dst)
			if in0 == nil || in1 == nil || out == nil || v.evaluator == nil {
				continue
			}
			a := in0.CopyNew()
			b := in1.CopyNew()
			v.clampCiphertextToMaxLevel(a)
			v.clampCiphertextToMaxLevel(b)

			minL := a.Level()
			if b.Level() < minL {
				minL = b.Level()
			}
			safeDropLevel(v.evaluator, a, a.Level()-minL)
			safeDropLevel(v.evaluator, b, b.Level()-minL)

			tmp := ckks.NewCiphertext(v.params, 1, minL)
			_ = v.evaluator.MulRelin(a, b, tmp)
			*out = *tmp

		case OP_MULCP:
			dst := int(op.Dst)
			l := int(op.Lhs)
			p := int(op.Rhs)
			in0, pt, out := ctAt(l), ptAt(p), ctAt(dst)
			if in0 == nil || pt == nil || out == nil || v.evaluator == nil {
				continue
			}
			v.clampCiphertextToMaxLevel(in0)
			_ = v.evaluator.Mul(in0, pt, out)
			v.clampCiphertextToMaxLevel(out)

		case OP_BOOT:
			dst := int(op.Dst)
			src := int(op.Lhs)
			targetLevel := int(op.Rhs)
			if ctAt(dst) == nil || ctAt(src) == nil {
				continue
			}
			v.pseudoBootstrap(dst, src, targetLevel)

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
	return C.int64_t(v.hdr.ArgLen)
}

//export getResLen
func getResLen(h C.uintptr_t) C.int64_t {
	v := getVM(h)
	if v == nil {
		return 0
	}
	return C.int64_t(v.hdr.ResLen)
}

//export encrypt
func encrypt(h C.uintptr_t, i C.int64_t, dat *C.double, length C.int) {
	v := getVM(h)
	if v == nil || !v.loadedOK || v.encryptor == nil || v.encoder == nil || dat == nil || length <= 0 {
		return
	}
	id := int(i)
	if id < 0 {
		return
	}
	v.ensureBuffers(id+1, len(v.plains))

	lv := v.params.MaxLevel()
	ls := int(v.params.LogDefaultScale())
	if id < len(v.argLevel) && int(v.argLevel[id]) >= 0 && int(v.argLevel[id]) <= v.params.MaxLevel() {
		lv = int(v.argLevel[id])
	}
	if id < len(v.argScale) && v.argScale[id] > 0 {
		ls = int(v.argScale[id])
	}

	src := (*[1 << 30]C.double)(unsafe.Pointer(dat))[:int(length):int(length)]
	values := make([]float64, int(length))
	for k := 0; k < int(length); k++ {
		values[k] = float64(src[k])
	}

	var pt rlwe.Plaintext
	v.encodeToPlainAt(&pt, values, lv, ls)

	ct := ckks.NewCiphertext(v.params, 1, lv)
	v.encryptor.Encrypt(&pt, ct)
	v.ciphers[id] = *ct
}

//export decrypt
func decrypt(h C.uintptr_t, i C.int64_t, dat *C.double) {
	v := getVM(h)
	if v == nil || !v.loadedOK || v.decryptor == nil || v.encoder == nil || dat == nil {
		return
	}
	id := int(i)
	if id < 0 || id >= len(v.ciphers) {
		return
	}
	pt := ckks.NewPlaintext(v.params, v.ciphers[id].Level())
	v.decryptor.Decrypt(&v.ciphers[id], pt)
	pt.Scale = v.ciphers[id].Scale

	vec := make([]complex128, v.slots)
	v.encoder.Decode(pt, vec)

	out := (*[1 << 30]C.double)(unsafe.Pointer(dat))[:v.slots:v.slots]
	for k := 0; k < v.slots; k++ {
		out[k] = C.double(real(vec[k]))
	}
}

//export decrypt_result
func decrypt_result(h C.uintptr_t, i C.int64_t, dat *C.double) {
	v := getVM(h)
	if v == nil || !v.loadedOK {
		return
	}
	ii := int(i)
	if ii < 0 || ii >= len(v.resDst) {
		return
	}
	decrypt(h, C.int64_t(v.resDst[ii]), dat)
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
func getCtxt(h C.uintptr_t, id C.int64_t) C.uintptr_t {
	v := getVM(h)
	if v == nil || !v.loadedOK {
		return 0
	}
	ii := int(id)
	if ii < 0 || ii >= len(v.ciphers) {
		return 0
	}
	return exportCiphertextHandle(&v.ciphers[ii])
}

//export setCtxt
func setCtxt(h C.uintptr_t, id C.int64_t, ctH C.uintptr_t) {
	v := getVM(h)
	if v == nil || !v.loadedOK {
		return
	}
	ii := int(id)
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
	if ctH != 0 {
		cgo.Handle(ctH).Delete()
	}
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
		v.togpu = bool(ongpu)
	}
}

//export printMem
func printMem(h C.uintptr_t) {
	v := getVM(h)
	if v == nil {
		return
	}
	fmt.Printf("[LATTIGO_HEVM] preset=%s logN=%d slots=%d maxLevel=%d loaded=%v\n",
		v.paramName, v.params.LogN(), v.slots, v.params.MaxLevel(), v.loadedOK)
}

func main() {}
