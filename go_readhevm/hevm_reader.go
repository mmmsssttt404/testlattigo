// hevm_reader.go
//
// Reads .cst and .hevm in current directory, matching your C header:
//
// HEVMOperation on disk is interpreted as:
//
//	uint16 opcode; uint16 dst; uint16 lhs; uint16 rhs;
//
// Notes:
//   - opcode==0xFFFF appears in your file as raw8=ff ff 00 00 00 00 00 00,
//     which acts like NOP because your C++ VM switch(default) does nothing.
//   - rhs is often better interpreted as int16 for rotate offsets.
//
// Build:
//
//	go build -o hevm_reader hevm_reader.go
//
// Run (files in current directory):
//
//	./hevm_reader -cst ./_hecate_ResNet.cst -hevm ./ResNet.40._hecate_ResNet.hevm -ops 10
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
)

type HEVMHeader struct {
	MagicNumber    uint32
	HEVMHeaderSize uint32
	ArgLength      uint64
	ResLength      uint64
}

type ConfigBody struct {
	ConfigBodyLength uint64
	NumOperations    uint64
	NumCtxtBuffer    uint64
	NumPtxtBuffer    uint64
	InitLevel        uint64
}

// On-disk op layout (per your header)
type HEVMOperation struct {
	Opcode uint16
	Dst    uint16
	Lhs    uint16
	Rhs    uint16
}

type DecodedOp struct {
	HEVMOperation
	RhsI16 int16
	Raw8   [8]byte
}

type HEVMFile struct {
	Header HEVMHeader
	Config ConfigBody

	ArgScale []uint64
	ArgLevel []uint64
	ResScale []uint64
	ResLevel []uint64
	ResDst   []uint64

	Ops []DecodedOp
}

// -------------------- CST reader --------------------
// format:
// [len:int64]
// repeat len:
//
//	[veclen:int64]
//	[float64 * veclen] (little-endian raw)
func ReadConstants(path string) ([][]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open constants: %w", err)
	}
	defer f.Close()

	nVec, err := readI64LE(f)
	if err != nil {
		return nil, fmt.Errorf("read constants len: %w", err)
	}
	if nVec < 0 {
		return nil, fmt.Errorf("invalid constants len=%d", nVec)
	}

	buf := make([][]float64, nVec)
	for i := int64(0); i < nVec; i++ {
		veclen, err := readI64LE(f)
		if err != nil {
			return nil, fmt.Errorf("read constants veclen[%d]: %w", i, err)
		}
		if veclen < 0 {
			return nil, fmt.Errorf("invalid veclen[%d]=%d", i, veclen)
		}
		tmp := make([]float64, veclen)
		if err := readF64SliceLE(f, tmp); err != nil {
			return nil, fmt.Errorf("read constants data[%d] (len=%d): %w", i, veclen, err)
		}
		buf[i] = tmp
	}
	return buf, nil
}

// -------------------- HEVM reader --------------------
func ReadHEVM(path string) (*HEVMFile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open hevm: %w", err)
	}
	defer f.Close()

	var hdr HEVMHeader
	if err := binary.Read(f, binary.LittleEndian, &hdr); err != nil {
		return nil, fmt.Errorf("read HEVMHeader: %w", err)
	}
	if hdr.MagicNumber != 0x4845564D {
		return nil, fmt.Errorf("bad magic: got=0x%08X want=0x4845564D", hdr.MagicNumber)
	}

	var cfg ConfigBody
	if err := binary.Read(f, binary.LittleEndian, &cfg); err != nil {
		return nil, fmt.Errorf("read ConfigBody: %w", err)
	}

	argLen := hdr.ArgLength
	resLen := hdr.ResLength
	if argLen > (1<<31-1) || resLen > (1<<31-1) {
		return nil, fmt.Errorf("arg/res length too large: arg=%d res=%d", argLen, resLen)
	}

	argScale := make([]uint64, argLen)
	argLevel := make([]uint64, argLen)
	resScale := make([]uint64, resLen)
	resLevel := make([]uint64, resLen)
	resDst := make([]uint64, resLen)

	if err := readU64SliceLE(f, argScale); err != nil {
		return nil, fmt.Errorf("read arg_scale: %w", err)
	}
	if err := readU64SliceLE(f, argLevel); err != nil {
		return nil, fmt.Errorf("read arg_level: %w", err)
	}
	if err := readU64SliceLE(f, resScale); err != nil {
		return nil, fmt.Errorf("read res_scale: %w", err)
	}
	if err := readU64SliceLE(f, resLevel); err != nil {
		return nil, fmt.Errorf("read res_level: %w", err)
	}
	if err := readU64SliceLE(f, resDst); err != nil {
		return nil, fmt.Errorf("read res_dst: %w", err)
	}

	if cfg.NumOperations > (1<<31 - 1) {
		return nil, fmt.Errorf("num_operations too large: %d", cfg.NumOperations)
	}
	opCount := int(cfg.NumOperations)

	ops := make([]DecodedOp, opCount)
	for i := 0; i < opCount; i++ {
		var raw [8]byte
		if _, err := io.ReadFull(f, raw[:]); err != nil {
			return nil, fmt.Errorf("read op raw[%d]: %w", i, err)
		}
		// decode as 4 little-endian uint16: opcode,dst,lhs,rhs
		w0 := binary.LittleEndian.Uint16(raw[0:2])
		w1 := binary.LittleEndian.Uint16(raw[2:4])
		w2 := binary.LittleEndian.Uint16(raw[4:6])
		w3 := binary.LittleEndian.Uint16(raw[6:8])

		ops[i] = DecodedOp{
			HEVMOperation: HEVMOperation{
				Opcode: w0,
				Dst:    w1,
				Lhs:    w2,
				Rhs:    w3,
			},
			RhsI16: int16(w3),
			Raw8:   raw,
		}
	}

	return &HEVMFile{
		Header:   hdr,
		Config:   cfg,
		ArgScale: argScale,
		ArgLevel: argLevel,
		ResScale: resScale,
		ResLevel: resLevel,
		ResDst:   resDst,
		Ops:      ops,
	}, nil
}

// -------------------- helpers --------------------
func readI64LE(r io.Reader) (int64, error) {
	var v int64
	err := binary.Read(r, binary.LittleEndian, &v)
	return v, err
}

func readU64SliceLE(r io.Reader, dst []uint64) error {
	for i := range dst {
		if err := binary.Read(r, binary.LittleEndian, &dst[i]); err != nil {
			return err
		}
	}
	return nil
}

func readF64SliceLE(r io.Reader, dst []float64) error {
	b := make([]byte, 8*len(dst))
	if _, err := io.ReadFull(r, b); err != nil {
		return err
	}
	for i := range dst {
		u := binary.LittleEndian.Uint64(b[i*8 : i*8+8])
		dst[i] = math.Float64frombits(u)
	}
	return nil
}

// -------------------- main --------------------
func main() {
	cstPath := flag.String("cst", "", "path to .cst constants file (optional)")
	hevmPath := flag.String("hevm", "", "path to .hevm program file (required)")
	printOpsN := flag.Int("ops", 10, "print first N ops")
	printRaw := flag.Bool("raw", true, "print raw8 bytes for printed ops")
	flag.Parse()

	if *hevmPath == "" {
		fmt.Fprintf(os.Stderr, "error: -hevm is required\n\n")
		flag.Usage()
		os.Exit(2)
	}

	if *cstPath != "" {
		cst, err := ReadConstants(*cstPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "ReadConstants failed: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("[CST] vectors=%d\n", len(cst))
		if len(cst) > 0 {
			fmt.Printf("[CST] vec0_len=%d\n", len(cst[0]))
		}
	}

	h, err := ReadHEVM(*hevmPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ReadHEVM failed: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("[HEVM] magic=0x%08X header_size=%d arg=%d res=%d\n",
		h.Header.MagicNumber, h.Header.HEVMHeaderSize, h.Header.ArgLength, h.Header.ResLength)
	fmt.Printf("[HEVM] body_len=%d ops=%d ctxt_buf=%d ptxt_buf=%d init_level=%d\n",
		h.Config.ConfigBodyLength, h.Config.NumOperations, h.Config.NumCtxtBuffer, h.Config.NumPtxtBuffer, h.Config.InitLevel)

	if len(h.ArgScale) > 0 {
		fmt.Printf("[HEVM] arg0: scale=%d level=%d\n", h.ArgScale[0], h.ArgLevel[0])
	}
	if len(h.ResScale) > 0 {
		fmt.Printf("[HEVM] res0: scale=%d level=%d dst=%d\n", h.ResScale[0], h.ResLevel[0], h.ResDst[0])
	}

	n := *printOpsN
	if n < 0 {
		n = 0
	}
	if n > len(h.Ops) {
		n = len(h.Ops)
	}
	for i := 0; i < n; i++ {
		op := h.Ops[i]
		tag := ""
		if op.Opcode == 0xFFFF {
			tag = " (NOP/Pad)"
		}
		fmt.Printf("[OP%06d] opcode=%d dst=%d lhs=%d rhs_u16=%d rhs_i16=%d%s\n",
			i, op.Opcode, op.Dst, op.Lhs, op.Rhs, op.RhsI16, tag)
		if *printRaw {
			fmt.Printf("          raw8=%02x %02x %02x %02x %02x %02x %02x %02x\n",
				op.Raw8[0], op.Raw8[1], op.Raw8[2], op.Raw8[3], op.Raw8[4], op.Raw8[5], op.Raw8[6], op.Raw8[7])
		}
	}
}
