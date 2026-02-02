// hevm_load_demo.cpp
//
// Build:
//   g++ -std=c++17 -O2 -Wall -Wextra -o hevm_load_demo hevm_load_demo.cpp
//
// Run:
//   ./hevm_load_demo ./_hecate_ResNet.cst ./ResNet.40._hecate_ResNet.hevm 10
// (最后的 10 表示打印前 10 条 op；不写默认 10)

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <iomanip>

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
};
}

struct DemoVM {
  std::vector<std::vector<double>> buffer;

  HEVMHeader header{};
  ConfigBody config{};
  std::vector<uint64_t> arg_scale, arg_level, res_scale, res_level, res_dst;
  std::vector<HEVMOperation> ops;

  void loadConstants(const char* name) {
    std::ifstream iff(std::string(name), std::ios::binary);
    if (!iff) throw std::runtime_error(std::string("cannot open constants file: ") + name);

    int64_t len = 0;
    iff.read(reinterpret_cast<char*>(&len), sizeof(int64_t));
    if (!iff) throw std::runtime_error("failed to read constants len");
    if (len < 0) throw std::runtime_error("invalid constants len < 0");

    buffer.resize(static_cast<std::size_t>(len));

    for (int64_t i = 0; i < len; i++) {
      int64_t veclen = 0;
      iff.read(reinterpret_cast<char*>(&veclen), sizeof(int64_t));
      if (!iff) throw std::runtime_error("failed to read veclen");
      if (veclen < 0) throw std::runtime_error("invalid veclen < 0");

      std::vector<double> tmp(static_cast<std::size_t>(veclen));
      iff.read(reinterpret_cast<char*>(tmp.data()),
               static_cast<std::streamsize>(veclen * sizeof(double)));
      if (!iff) throw std::runtime_error("failed to read vector data");

      buffer[static_cast<std::size_t>(i)] = std::move(tmp);
    }
  }

  void loadHeader(std::istream& iff) {
    iff.read(reinterpret_cast<char*>(&header), sizeof(HEVMHeader));
    iff.read(reinterpret_cast<char*>(&config), sizeof(ConfigBody));
    if (!iff) throw std::runtime_error("failed to read header+config");

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
  }

  void loadHEVM(const char* name) {
    std::ifstream iff(std::string(name), std::ios::binary);
    if (!iff) throw std::runtime_error(std::string("cannot open hevm file: ") + name);

    loadHeader(iff);

    ops.resize(static_cast<std::size_t>(config.num_operations));
    iff.read(reinterpret_cast<char*>(ops.data()),
             static_cast<std::streamsize>(ops.size() * sizeof(HEVMOperation)));
    if (!iff) throw std::runtime_error("failed to read ops array");
  }
};

static void printOp(const HEVMOperation& op, std::size_t idx) {
  int16_t rhs_i16 = static_cast<int16_t>(op.rhs);
  bool is_nop = (op.opcode == 0xFFFF);

  std::cout << "[OP" << std::setw(6) << std::setfill('0') << idx << "] "
            << "opcode=" << std::setfill(' ') << op.opcode
            << " dst=" << op.dst
            << " lhs=" << op.lhs
            << " rhs_u16=" << op.rhs
            << " rhs_i16=" << rhs_i16;

  if (is_nop) std::cout << " (NOP/Pad)";
  std::cout << "\n";
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <constants.cst> <program.hevm> [print_ops]\n";
    return 2;
  }
  const char* cst = argv[1];
  const char* hevm = argv[2];
  int printN = 10;
  if (argc >= 4) {
    printN = std::stoi(argv[3]);
    if (printN < 0) printN = 0;
  }

  try {
    DemoVM vm;
    vm.loadConstants(cst);
    vm.loadHEVM(hevm);

    std::cout << "[CST] vectors=" << vm.buffer.size() << "\n";
    if (!vm.buffer.empty()) std::cout << "[CST] vec0_len=" << vm.buffer[0].size() << "\n";

    std::cout << "[HEVM] magic=0x" << std::hex << vm.header.magic_number << std::dec
              << " header_size=" << vm.header.hevm_header_size
              << " arg=" << vm.header.config_header.arg_length
              << " res=" << vm.header.config_header.res_length << "\n";

    std::cout << "[HEVM] body_len=" << vm.config.config_body_length
              << " ops=" << vm.config.num_operations
              << " ctxt_buf=" << vm.config.num_ctxt_buffer
              << " ptxt_buf=" << vm.config.num_ptxt_buffer
              << " init_level=" << vm.config.init_level << "\n";

    if (!vm.arg_scale.empty()) {
      std::cout << "[HEVM] arg0: scale=" << vm.arg_scale[0]
                << " level=" << vm.arg_level[0] << "\n";
    }
    if (!vm.res_scale.empty()) {
      std::cout << "[HEVM] res0: scale=" << vm.res_scale[0]
                << " level=" << vm.res_level[0]
                << " dst=" << vm.res_dst[0] << "\n";
    }

    std::size_t n = static_cast<std::size_t>(printN);
    if (n > vm.ops.size()) n = vm.ops.size();
    for (std::size_t i = 0; i < n; i++) printOp(vm.ops[i], i);

  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
