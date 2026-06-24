#pragma once

#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SourceMgr.h"

#define DEFAULT_ENTRY "main"

struct MemRef {
  std::vector<int64_t> shape;
  std::vector<float> data;

  MemRef(std::vector<int64_t> s) : shape(std::move(s)) {
    int64_t totalSize = std::accumulate(shape.begin(), shape.end(), 1LL,
                                        std::multiplies<int64_t>());
    data.resize(totalSize, 0.0f);
  }

  int64_t getFlatIndex(const std::vector<int64_t> &indices) const {
    if (indices.size() != shape.size()) {
      throw std::runtime_error("Index/shape dimensionality mismatch");
    }
    int64_t flatIndex = 0;
    int64_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
      if (indices[i] >= shape[i] || indices[i] < 0) {
        throw std::out_of_range("Index out of bounds");
      }
      flatIndex += indices[i] * stride;
      stride *= shape[i];
    }
    return flatIndex;
  }
};

struct RuntimeValue {
  std::variant<bool, int, long, float, std::shared_ptr<MemRef>> data;

  template <typename T> T get() const { return std::get<T>(data); }
};

struct StackFrame {
  std::string funcName;
  llvm::DenseMap<mlir::Value, RuntimeValue> state;
};

class Interpreter {
public:
  Interpreter();
  void runFromEntry(mlir::ModuleOp runModule,
                    llvm::StringRef entryFuncName = DEFAULT_ENTRY);
  void runAll(mlir::ModuleOp runModule);
  void printStackTrace();

private:
  mlir::ModuleOp module;
  std::vector<StackFrame> callStack;
  llvm::StringMap<std::function<bool(mlir::Operation *)>> opRegistry;

  void runFunction(mlir::func::FuncOp fn);
  void registerOps();
  StackFrame &currentFrame() { return callStack.back(); }
  std::vector<RuntimeValue> execute(mlir::Block &block);
  bool dispatch(mlir::Operation *op);
  std::vector<int64_t>
  extractIndices(mlir::Operation::operand_range indicesOps);

  void evalIf(mlir::scf::IfOp op);
  void evalCall(mlir::func::CallOp op);
  void evalConstant(mlir::arith::ConstantOp op);
  void evalAddF(mlir::arith::AddFOp op);
  void evalAddI(mlir::arith::AddIOp op);
  void evalAlloc(mlir::memref::AllocOp op);
  void evalDealloc(mlir::memref::DeallocOp op);
  void evalStore(mlir::memref::StoreOp op);
  void evalLoad(mlir::memref::LoadOp op);
  void evalMulF(mlir::arith::MulFOp op);
  void evalFor(mlir::scf::ForOp op);
  void evalMatmul(mlir::linalg::MatmulOp op);
  void printRuntimeValue(const RuntimeValue &val);
};