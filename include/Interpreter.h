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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SourceMgr.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

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
  std::variant<bool, int64_t, int32_t, float, std::shared_ptr<MemRef>> data;

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

  template <typename OpT>
  void registerOp(std::function<void(Interpreter &, OpT)> handler) {
    opRegistry[OpT::getOperationName()] = [this, handler](mlir::Operation *op) {
      handler(*this, llvm::cast<OpT>(op));
      return true;
    };
  }

  void setValue(mlir::Value key, RuntimeValue val) {
    currentFrame().state[key] = val;
  }
  RuntimeValue getValue(mlir::Value key) { return currentFrame().state[key]; }

  std::vector<RuntimeValue> executeBlock(mlir::Block &block) {
    return execute(block);
  }
  void pushFrame(const StackFrame &frame) { callStack.push_back(frame); }
  void popFrame() { callStack.pop_back(); }
  mlir::ModuleOp getModule() { return module; }
  void eraseValue(mlir::Value key) { currentFrame().state.erase(key); }
  std::vector<int64_t>
  extractIndices(mlir::Operation::operand_range indicesOps);
  bool dispatch(mlir::Operation *op);

private:
  mlir::ModuleOp module;
  std::vector<StackFrame> callStack;
  llvm::StringMap<std::function<bool(Interpreter &, mlir::Operation *)>>
      opRegistry;

  void runFunction(mlir::func::FuncOp fn);
  void registerOps();
  StackFrame &currentFrame() { return callStack.back(); }
  std::vector<RuntimeValue> execute(mlir::Block &block);

  void printRuntimeValue(const RuntimeValue &val);
};