#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

// 1. MemRef Data Structure for N-dimensional Host Memory
struct MemRef {
  std::vector<int64_t> shape;
  std::vector<float> data;

  MemRef(std::vector<int64_t> s) : shape(std::move(s)) {
    int64_t totalSize = std::accumulate(shape.begin(), shape.end(), 1LL,
                                        std::multiplies<int64_t>());
    data.resize(totalSize, 0.0f);
  }

  int64_t getFlatIndex(const std::vector<int64_t>& indices) const {
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

// 2. Dynamic Runtime Value
struct RuntimeValue {
  std::variant<int32_t, int64_t, float, std::shared_ptr<MemRef>> data;

  template <typename T>
  T get() const {
    return std::get<T>(data);
  }
};

struct StackFrame {
  llvm::DenseMap<mlir::Value, RuntimeValue> state;
};

// 3. Interpreter Class
class Interpreter {
 public:
   void run(mlir::ModuleOp module) {
     auto mainFunc = module.lookupSymbol<mlir::func::FuncOp>("main");
     if (!mainFunc || mainFunc.getBlocks().empty()) {
       throw std::runtime_error("Error: 'main' function not found or empty.");
     }

     struct StackFrame mainFrame;
     callStack.push_back(mainFrame);

     executeBlock(mainFunc.getBlocks().front());
   }

 private:
   std::vector<StackFrame> callStack;

   StackFrame &currentFrame() { return callStack.back(); }

   void executeBlock(mlir::Block &block) {
     for (mlir::Operation &op : block) {
       if (dispatch(&op))
         continue;

       if (auto returnOp = llvm::dyn_cast<mlir::func::ReturnOp>(op)) {
         if (returnOp.getNumOperands() > 0) {
           printValue(currentFrame().state[returnOp.getOperand(0)]);
         }
         return;
       }
       throw std::runtime_error("Unsupported op: " +
                                op.getName().getStringRef().str());
     }
   }

   bool dispatch(mlir::Operation *op) {
     if (auto cOp = llvm::dyn_cast<mlir::arith::ConstantOp>(op)) {
       evalConstant(cOp);
       return true;
     }
     if (auto aOp = llvm::dyn_cast<mlir::arith::AddFOp>(op)) {
       evalAddF(aOp);
       return true;
     }
     if (auto alOp = llvm::dyn_cast<mlir::memref::AllocOp>(op)) {
       evalAlloc(alOp);
       return true;
     }
     if (auto dOp = llvm::dyn_cast<mlir::memref::DeallocOp>(op)) {
       evalDealloc(dOp);
       return true;
     }
     if (auto sOp = llvm::dyn_cast<mlir::memref::StoreOp>(op)) {
       evalStore(sOp);
       return true;
     }
     if (auto lOp = llvm::dyn_cast<mlir::memref::LoadOp>(op)) {
       evalLoad(lOp);
       return true;
     }
     if (auto forOp = llvm::dyn_cast<mlir::scf::ForOp>(op)) {
       evalFor(forOp);
       return true;
     }
     return false;
   }

  void evalConstant(mlir::arith::ConstantOp op) {
    if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(op.getValue())) {
      // Check if it's an 'index' type or standard int
      if (op.getType().isIndex()) {
        currentFrame().state[op.getResult()] =
            RuntimeValue{intAttr.getInt()}; // Stores as int64_t
      } else {
        currentFrame().state[op.getResult()] =
            RuntimeValue{static_cast<int32_t>(intAttr.getInt())};
      }
    } else if (auto floatAttr =
                   llvm::dyn_cast<mlir::FloatAttr>(op.getValue())) {
      currentFrame().state[op.getResult()] = RuntimeValue{
          static_cast<float>(floatAttr.getValue().convertToFloat())};
    }
  }

  void evalAddF(mlir::arith::AddFOp op) {
    float lhs = currentFrame().state[op.getLhs()].get<float>();
    float rhs = currentFrame().state[op.getRhs()].get<float>();
    currentFrame().state[op.getResult()] = RuntimeValue{lhs + rhs};
  }

  void evalAlloc(mlir::memref::AllocOp op) {
    auto memRefType = llvm::cast<mlir::MemRefType>(op.getType());
    std::vector<int64_t> shape = memRefType.getShape().vec();
    currentFrame().state[op.getResult()] =
        RuntimeValue{std::make_shared<MemRef>(shape)};
  }

  void evalDealloc(mlir::memref::DeallocOp op) {
    currentFrame().state.erase(op.getMemref());
  }

  std::vector<int64_t> extractIndices(
      mlir::Operation::operand_range indicesOps) {
    std::vector<int64_t> indices;
    for (auto idxVal : indicesOps) {
      indices.push_back(currentFrame().state[idxVal].get<int64_t>());
    }
    return indices;
  }

  void evalStore(mlir::memref::StoreOp op) {
    float val = currentFrame().state[op.getValueToStore()].get<float>();
    auto memref =
        currentFrame().state[op.getMemref()].get<std::shared_ptr<MemRef>>();
    auto indices = extractIndices(op.getIndices());
    memref->data[memref->getFlatIndex(indices)] = val;
  }

  void evalLoad(mlir::memref::LoadOp op) {
    auto memref =
        currentFrame().state[op.getMemref()].get<std::shared_ptr<MemRef>>();
    auto indices = extractIndices(op.getIndices());
    float val = memref->data[memref->getFlatIndex(indices)];
    currentFrame().state[op.getResult()] = RuntimeValue{val};
  }

  void evalFor(mlir::scf::ForOp op) {
    int64_t lower = currentFrame().state[op.getLowerBound()].get<int64_t>();
    int64_t upper = currentFrame().state[op.getUpperBound()].get<int64_t>();
    int64_t step = currentFrame().state[op.getStep()].get<int64_t>();

    std::cout << "Loop bound:" << lower << " " << upper << " " << step
              << std::endl;

    mlir::Value iv = op.getInductionVar();
    mlir::Block* bodyBlock = op.getBody();

    for (int64_t i = lower; i < upper; i += step) {
      currentFrame().state[iv] = RuntimeValue{i};
      for (mlir::Operation& bodyOp : bodyBlock->getOperations()) {
        if (llvm::isa<mlir::scf::YieldOp>(bodyOp)) continue;
        dispatch(&bodyOp);
      }
    }
  }

  void printValue(const RuntimeValue& val) {
    if (std::holds_alternative<int32_t>(val.data)) {
      std::cout << "Interpreter Output (i32): " << val.get<int32_t>() << "\n";
    } else if (std::holds_alternative<float>(val.data)) {
      std::cout << "Interpreter Output (f32): " << val.get<float>() << "\n";
    }
  }
};

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <path to .mlir file>\n";
    return 1;
  }

  mlir::MLIRContext context;
  mlir::DialectRegistry registry;

  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
  if (!module) {
    std::cerr << "Failed to parse MLIR file.\n";
    return 1;
  }

  try {
    Interpreter interpreter;
    interpreter.run(*module);
  } catch (const std::exception& e) {
    std::cerr << "Runtime Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}