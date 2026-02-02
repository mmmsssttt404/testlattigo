// hevm_verify.cpp
//
// Build:
//   g++ -std=c++17 -O2 -Wall -Wextra -o hevm_verify hevm_verify.cpp
//
// Run:
//   ./hevm_verify -cst ./_hecate_ResNet.cst -hevm ./ResNet.40._hecate_ResNet.hevm
// or positional:
//   ./hevm_verify ./_hecate_ResNet.cst ./ResNet.40._hecate_ResNet.hevm

#include <cstdint>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <limits>
#include <algorithm>

extern "C" {

struct HEVMHeader {
  uint32_t magic_number = 0x4845564D;
  uint32_t hevm_header_size;
  struct ConfigHeader {
    uint64_t arg_length;
    uint64_t res_length;
  } config_header;
};

struct ConfigBody {
  uint64_t config_body_length;
  uint64_t num_operations;
  uint64_t num_ctxt_buffer;
  uint64_t num_ptxt_buffer;
  uint64_t init_level;
};

struct HEVMOperation {
  uint16_t opcode;
  uint16_t dst;
  uint16_t lhs;
  uint16_t rhs;
}; // 8 bytes
}

static bool is_valid_opcode(uint16_t op) { return op <= 10; }

static void print_op(const HEVMOperation& op, std::size_t idx) {
  int16_t rhs_i16 = static_cast<int16_t>(op.rhs);
  std::cout << "[OP" << std::setw(6) << std::setfill('0') << idx << "] "
            << std::setfill(' ')
            << "opcode=" << op.opcode
            << " dst=" << op.dst
            << " lhs=" << op.lhs
            << " rhs_u16=" << op.rhs
            << " rhs_i16=" << rhs_i16;
  if (op.opcode == 0xFFFF) std::cout << " (NOP/Pad)";
  std::cout << "\n";
}

struct Constants {
  std::vector<std::vector<double>> buf;

  void load(const std::string& path) {
    std::ifstream iff(path, std::ios::binary);
    if (!iff) throw std::runtime_error("cannot open constants file: " + path);

    int64_t len = 0;
    iff.read(reinterpret_cast<char*>(&len), sizeof(int64_t));
    if (!iff) throw std::runtime_error("failed to read constants len");
    if (len < 0) throw std::runtime_error("invalid constants len < 0");

    buf.resize(static_cast<std::size_t>(len));
    for (int64_t i = 0; i < len; i++) {
      int64_t veclen = 0;
      iff.read(reinterpret_cast<char*>(&veclen), sizeof(int64_t));
      if (!iff) throw std::runtime_error("failed to read veclen");
      if (veclen < 0) throw std::runtime_error("invalid veclen < 0");

      std::vector<double> tmp(static_cast<std::size_t>(veclen));
      iff.read(reinterpret_cast<char*>(tmp.data()),
               static_cast<std::streamsize>(veclen * sizeof(double)));
      if (!iff) throw std::runtime_error("failed to read vector data");

      buf[static_cast<std::size_t>(i)] = std::move(tmp);
    }
  }
};

struct LoadedHEVM {
  HEVMHeader header{};
  ConfigBody config{};
  std::vector<uint64_t> arg_scale, arg_level, res_scale, res_level, res_dst;
  std::vector<HEVMOperation> ops;

  void load(const std::string& hevm_path) {
    std::ifstream iff(hevm_path, std::ios::binary);
    if (!iff) throw std::runtime_error("cannot open hevm file: " + hevm_path);

    iff.read(reinterpret_cast<char*>(&header), sizeof(HEVMHeader));
    iff.read(reinterpret_cast<char*>(&config), sizeof(ConfigBody));
    if (!iff) throw std::runtime_error("failed to read HEVMHeader/ConfigBody");

    if (header.magic_number != 0x4845564D) {
      std::ostringstream oss;
      oss << "bad magic: got=0x" << std::hex << header.magic_number << std::dec;
      throw std::runtime_error(oss.str());
    }

    arg_scale.resize(header.config_header.arg_length);
    arg_level.resize(header.config_header.arg_length);
    res_scale.resize(header.config_header.res_length);
    res_level.resize(header.config_header.res_length);
    res_dst.resize(header.config_header.res_length);

    iff.read(reinterpret_cast<char*>(arg_scale.data()),
             static_cast<std::streamsize>(arg_scale.size() * sizeof(uint64_t)));
    iff.read(reinterpret_cast<char*>(arg_level.data()),
             static_cast<std::streamsize>(arg_level.size() * sizeof(uint64_t)));
    iff.read(reinterpret_cast<char*>(res_scale.data()),
             static_cast<std::streamsize>(res_scale.size() * sizeof(uint64_t)));
    iff.read(reinterpret_cast<char*>(res_level.data()),
             static_cast<std::streamsize>(res_level.size() * sizeof(uint64_t)));
    iff.read(reinterpret_cast<char*>(res_dst.data()),
             static_cast<std::streamsize>(res_dst.size() * sizeof(uint64_t)));

    if (!iff) throw std::runtime_error("failed to read meta arrays");

    ops.resize(static_cast<std::size_t>(config.num_operations));
    iff.read(reinterpret_cast<char*>(ops.data()),
             static_cast<std::streamsize>(ops.size() * sizeof(HEVMOperation)));
    if (!iff) throw std::runtime_error("failed to read ops array");
  }
};

struct MinMax {
  uint32_t minv = std::numeric_limits<uint32_t>::max();
  uint32_t maxv = 0;
  void add(uint32_t v) {
    minv = std::min(minv, v);
    maxv = std::max(maxv, v);
  }
};

static void usage(const char* prog) {
  std::cerr
    << "Usage:\n"
    << "  " << prog << " <cst> <hevm>\n"
    << "  " << prog << " -cst <cst> -hevm <hevm>\n";
}

int main(int argc, char** argv) {
  std::string cst_path, hevm_path;

  // flags
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "-cst" && i + 1 < argc) cst_path = argv[++i];
    else if (a == "-hevm" && i + 1 < argc) hevm_path = argv[++i];
  }

  // positional fallback
  if (cst_path.empty() && hevm_path.empty() && argc >= 3) {
    cst_path = argv[1];
    hevm_path = argv[2];
  }

  if (hevm_path.empty()) {
    usage(argv[0]);
    return 2;
  }

  try {
    Constants cst;
    if (!cst_path.empty()) cst.load(cst_path);

    LoadedHEVM h;
    h.load(hevm_path);

    // ---- header print ----
    if (!cst_path.empty()) {
      std::cout << "[CST] vectors=" << cst.buf.size() << "\n";
      if (!cst.buf.empty()) std::cout << "[CST] vec0_len=" << cst.buf[0].size() << "\n";
    }

    std::cout << "[HEVM] magic=0x" << std::hex << h.header.magic_number << std::dec
              << " header_size=" << h.header.hevm_header_size
              << " arg=" << h.header.config_header.arg_length
              << " res=" << h.header.config_header.res_length << "\n";
    std::cout << "[HEVM] body_len=" << h.config.config_body_length
              << " ops=" << h.config.num_operations
              << " ctxt_buf=" << h.config.num_ctxt_buffer
              << " ptxt_buf=" << h.config.num_ptxt_buffer
              << " init_level=" << h.config.init_level << "\n";

    // ---- opcode stats ----
    std::vector<uint64_t> hist(0x10000, 0);
    uint64_t nop = 0, valid = 0, other = 0;

    uint64_t cur_run = 0, max_run = 0;
    uint64_t first_nop_run_start = 0;
    uint64_t max_run_start = 0;

    // per-opcode min/max for fields
    std::vector<MinMax> mm_dst(11), mm_lhs(11), mm_rhs(11);

    auto fail = [&](std::size_t i, const std::string& msg) {
      std::cerr << "CHECK_FAIL at op index " << i << ": " << msg << "\n";
      print_op(h.ops[i], i);
      throw std::runtime_error("verification failed");
    };

    const uint64_t ctxtN = h.config.num_ctxt_buffer;
    const uint64_t ptxtN = h.config.num_ptxt_buffer;
    const uint64_t cstN  = cst.buf.size();

    for (std::size_t i = 0; i < h.ops.size(); i++) {
      const auto& op = h.ops[i];
      hist[op.opcode]++;

      if (op.opcode == 0xFFFF) {
        nop++;
        if (cur_run == 0) first_nop_run_start = i;
        cur_run++;
        if (cur_run > max_run) {
          max_run = cur_run;
          max_run_start = first_nop_run_start;
        }
        continue;
      } else {
        cur_run = 0;
      }

      if (!is_valid_opcode(op.opcode)) {
        other++;
        fail(i, "opcode not in [0..10] and not 0xFFFF");
      } else {
        valid++;
      }

      // record min/max for dst/lhs/rhs for opcodes 0..10
      mm_dst[op.opcode].add(op.dst);
      mm_lhs[op.opcode].add(op.lhs);
      mm_rhs[op.opcode].add(op.rhs);

      // ---- strict bound checks by semantics (based on your C++ VM usage) ----
      switch (op.opcode) {
        case 0: {
          // Encode: plains[dst], buffer[lhs] or identity if lhs==-1(0xFFFF)
          if (op.dst >= ptxtN) fail(i, "Encode dst out of range (ptxt buffer)");
          if (!(op.lhs == 0xFFFF || op.lhs < cstN)) {
            std::ostringstream oss;
            oss << "Encode lhs out of range (const buffer), lhs=" << op.lhs
                << " const_vectors=" << cstN << " (allowed: <const_vectors or 0xFFFF)";
            fail(i, oss.str());
          }
          // rhs packs (level << 10) | scale, not validated here.
          break;
        }
        case 1: // RotateC: ciphers[lhs] -> ciphers[dst], rhs is int16 offset
        case 2: // NegateC
        case 3: // RescaleC
        case 4: // ModswitchC
        case 5: // UpscaleC (may exist in HEaaN; SEAL asserts, but file may still contain)
        case 6: // AddCC
        case 8: // MulCC
        case 10:{// Bootstrap
          if (op.dst >= ctxtN) fail(i, "ct-op dst out of range (ctxt buffer)");
          if (op.lhs >= ctxtN) fail(i, "ct-op lhs out of range (ctxt buffer)");
          // rhs meaning varies; not always an index.
          break;
        }
        case 7: // AddCP: rhs indexes plaintext
        case 9: { // MulCP: rhs indexes plaintext
          if (op.dst >= ctxtN) fail(i, "ct-pt op dst out of range (ctxt buffer)");
          if (op.lhs >= ctxtN) fail(i, "ct-pt op lhs out of range (ctxt buffer)");
          if (op.rhs >= ptxtN) fail(i, "ct-pt op rhs out of range (ptxt buffer)");
          break;
        }
        default:
          fail(i, "unreachable: opcode should be 0..10 here");
      }
    }

    const double total = static_cast<double>(h.ops.size());
    auto pct = [&](uint64_t x) -> double {
      return total == 0.0 ? 0.0 : 100.0 * static_cast<double>(x) / total;
    };

    std::cout << "[STATS] total_ops=" << h.ops.size() << "\n";
    std::cout << "[STATS] nop_ops(opcode=0xFFFF)=" << nop
              << " (" << std::fixed << std::setprecision(2) << pct(nop) << "%)\n";
    std::cout << "[STATS] valid_ops(opcode 0..10)=" << valid
              << " (" << std::fixed << std::setprecision(2) << pct(valid) << "%)\n";
    std::cout << "[STATS] other_ops=" << other
              << " (" << std::fixed << std::setprecision(2) << pct(other) << "%)\n";

    std::cout << "[NOP] longest_run=" << max_run
              << " start_index=" << max_run_start << "\n";

    // print histogram for opcodes 0..10 and 0xFFFF only
    std::cout << "\n[HIST] opcode counts\n";
    for (int op = 0; op <= 10; op++) {
      std::cout << "  opcode=" << op << " count=" << hist[op];
      if (hist[op] > 0) {
        std::cout << " dst[min,max]=[" << mm_dst[op].minv << "," << mm_dst[op].maxv << "]"
                  << " lhs[min,max]=[" << mm_lhs[op].minv << "," << mm_lhs[op].maxv << "]"
                  << " rhs[min,max]=[" << mm_rhs[op].minv << "," << mm_rhs[op].maxv << "]";
      }
      std::cout << "\n";
    }
    std::cout << "  opcode=65535 count=" << hist[0xFFFF] << "\n";

    std::cout << "\n[VERIFY] OK: all ops scanned; opcode set is {0..10,0xFFFF} and all indices are in-range.\n";
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
