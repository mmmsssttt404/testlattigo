// hevm_nop_stats.cpp
//
// Build:
//   g++ -std=c++17 -O2 -Wall -Wextra -o hevm_nop_stats hevm_nop_stats.cpp
//
// Run (two positional args):
//   ./hevm_nop_stats ./_hecate_ResNet.cst ./ResNet.40._hecate_ResNet.hevm
//
// Run (with flags):
//   ./hevm_nop_stats -cst ./_hecate_ResNet.cst -hevm ./ResNet.40._hecate_ResNet.hevm -n 30
//
// If you only want HEVM:
//   ./hevm_nop_stats -hevm ./ResNet.40._hecate_ResNet.hevm -n 30

#include <cstdint>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <cmath>

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
  bool is_nop = (op.opcode == 0xFFFF);

  std::cout << "[OP" << std::setw(6) << std::setfill('0') << idx << "] "
            << std::setfill(' ')
            << "opcode=" << op.opcode
            << " dst=" << op.dst
            << " lhs=" << op.lhs
            << " rhs_u16=" << op.rhs
            << " rhs_i16=" << rhs_i16;
  if (is_nop) std::cout << " (NOP/Pad)";
  std::cout << "\n";
}

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

static void read_constants_brief(const std::string& cst_path) {
  std::ifstream iff(cst_path, std::ios::binary);
  if (!iff) throw std::runtime_error("cannot open constants file: " + cst_path);

  int64_t len = 0;
  iff.read(reinterpret_cast<char*>(&len), sizeof(int64_t));
  if (!iff) throw std::runtime_error("failed to read constants len");
  if (len < 0) throw std::runtime_error("invalid constants len < 0");

  std::cout << "[CST] vectors=" << len << "\n";
  if (len > 0) {
    int64_t veclen = 0;
    iff.read(reinterpret_cast<char*>(&veclen), sizeof(int64_t));
    if (!iff) throw std::runtime_error("failed to read vec0_len");
    std::cout << "[CST] vec0_len=" << veclen << "\n";
  }
}

static void usage(const char* prog) {
  std::cerr
    << "Usage:\n"
    << "  " << prog << " <cst> <hevm> [-n N]\n"
    << "  " << prog << " -hevm <hevm> [-cst <cst>] [-n N]\n"
    << "\nExamples:\n"
    << "  " << prog << " ./_hecate_ResNet.cst ./ResNet.40._hecate_ResNet.hevm -n 30\n"
    << "  " << prog << " -hevm ./ResNet.40._hecate_ResNet.hevm -n 30\n";
}

int main(int argc, char** argv) {
  std::string cst_path;
  std::string hevm_path;
  std::size_t printN = 30;

  // Accept both positional and flag style.
  // Parse flags first.
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "-cst" && i + 1 < argc) {
      cst_path = argv[++i];
    } else if (a == "-hevm" && i + 1 < argc) {
      hevm_path = argv[++i];
    } else if (a == "-n" && i + 1 < argc) {
      printN = static_cast<std::size_t>(std::stoll(argv[++i]));
    }
  }

  // If not provided by flags, try positional: argv[1]=cst argv[2]=hevm
  // But avoid treating "-something" as positional.
  if (hevm_path.empty()) {
    if (argc >= 3 && argv[1][0] != '-' && argv[2][0] != '-') {
      cst_path = argv[1];
      hevm_path = argv[2];
    } else if (argc >= 2 && argv[1][0] != '-') {
      // allow: prog <hevm> (single positional)
      hevm_path = argv[1];
    }
  }

  if (hevm_path.empty()) {
    usage(argv[0]);
    return 2;
  }

  try {
    if (!cst_path.empty()) {
      read_constants_brief(cst_path);
    }

    LoadedHEVM h;
    h.load(hevm_path);

    std::cout << "[HEVM] magic=0x" << std::hex << h.header.magic_number << std::dec
              << " header_size=" << h.header.hevm_header_size
              << " arg=" << h.header.config_header.arg_length
              << " res=" << h.header.config_header.res_length << "\n";
    std::cout << "[HEVM] body_len=" << h.config.config_body_length
              << " ops=" << h.config.num_operations
              << " ctxt_buf=" << h.config.num_ctxt_buffer
              << " ptxt_buf=" << h.config.num_ptxt_buffer
              << " init_level=" << h.config.init_level << "\n";

    if (!h.arg_scale.empty()) {
      std::cout << "[HEVM] arg0: scale=" << h.arg_scale[0]
                << " level=" << h.arg_level[0] << "\n";
    }
    if (!h.res_scale.empty()) {
      std::cout << "[HEVM] res0: scale=" << h.res_scale[0]
                << " level=" << h.res_level[0]
                << " dst=" << h.res_dst[0] << "\n";
    }

    // Stats
    std::size_t nop = 0, valid = 0, other = 0;
    for (const auto& op : h.ops) {
      if (op.opcode == 0xFFFF) nop++;
      else if (is_valid_opcode(op.opcode)) valid++;
      else other++;
    }

    const double total = static_cast<double>(h.ops.size());
    auto pct = [&](std::size_t x) -> double {
      return total == 0.0 ? 0.0 : 100.0 * static_cast<double>(x) / total;
    };

    std::cout << "[STATS] total_ops=" << h.ops.size() << "\n";
    std::cout << "[STATS] nop_ops(opcode=0xFFFF)=" << nop
              << " (" << std::fixed << std::setprecision(2) << pct(nop) << "%)\n";
    std::cout << "[STATS] valid_ops(opcode 0..10)=" << valid
              << " (" << std::fixed << std::setprecision(2) << pct(valid) << "%)\n";
    std::cout << "[STATS] other_ops=" << other
              << " (" << std::fixed << std::setprecision(2) << pct(other) << "%)\n";

    // Print first N valid ops
    std::cout << "\n[FIRST_VALID_OPS] N=" << printN << "\n";
    std::size_t printed = 0;
    for (std::size_t i = 0; i < h.ops.size() && printed < printN; i++) {
      if (is_valid_opcode(h.ops[i].opcode)) {
        print_op(h.ops[i], i);
        printed++;
      }
    }
    if (printed < printN) {
      std::cout << "[FIRST_VALID_OPS] only printed " << printed << " valid ops\n";
    }

  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
