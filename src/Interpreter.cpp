#include "Interpreter.h"
#include "OpRegistry.h"
#include <iostream>

Interpreter::Interpreter() { opRegistry = OpRegistry::get().handlers; }

// void Interpreter::registerOps() {
//   opRegistry[mlir::arith::AddFOp::getOperationName()] =
//       [this](mlir::Operation *op) {
//         this->evalAddF(llvm::cast<mlir::arith::AddFOp>(op));
//         return true;
//       };

//   opRegistry[mlir::arith::AddIOp::getOperationName()] =
//       [this](mlir::Operation *op) {
//         this->evalAddI(llvm::cast<mlir::arith::AddIOp>(op));
//         return true;
//       };
//   opRegistry[mlir::arith::MulFOp::getOperationName()] =
//       [this](mlir::Operation *op) {
//         this->evalMulF(llvm::cast<mlir::arith::MulFOp>(op));
//         return true;
//       };

//   opRegistry[mlir::scf::IfOp::getOperationName()] =
//       [this](mlir::Operation *op) {
//         this->evalIf(llvm::cast<mlir::scf::IfOp>(op));
//         return true;
//       };

//   opRegistry[mlir::arith::ConstantOp::getOperationName()] =
//       [this](mlir::Operation *op) {
//         this->evalConstant(llvm::cast<mlir::arith::ConstantOp>(op));
//         return true;
//       };

//   opRegistry[mlir::memref::AllocOp::getOperationName()] =
//       [this](mlir::Operation *op) {
//         this->evalAlloc(llvm::cast<mlir::memref::AllocOp>(op));
//         return true;
//       };

//   opRegistry[mlir::memref::DeallocOp::getOperationName()] =
//       [this](mlir::Operation *op) {
//         this->evalDealloc(llvm::cast<mlir::memref::DeallocOp>(op));
//         return true;
//       };

//   opRegistry[mlir::memref::StoreOp::getOperationName()] =
//       [this](mlir::Operation *op) {
//         this->evalStore(llvm::cast<mlir::memref::StoreOp>(op));
//         return true;
//       };

//   opRegistry[mlir::memref::LoadOp::getOperationName()] =
//       [this](mlir::Operation *op) {
//         this->evalLoad(llvm::cast<mlir::memref::LoadOp>(op));
//         return true;
//       };

//   opRegistry[mlir::scf::ForOp::getOperationName()] =
//       [this](mlir::Operation *op) {
//         this->evalFor(llvm::cast<mlir::scf::ForOp>(op));
//         return true;
//       };

//   opRegistry[mlir::linalg::MatmulOp::getOperationName()] =
//       [this](mlir::Operation *op) {
//         this->evalMatmul(llvm::cast<mlir::linalg::MatmulOp>(op));
//         return true;
//       };

//   opRegistry[mlir::func::CallOp::getOperationName()] =
//       [this](mlir::Operation *op) {
//         this->evalCall(llvm::cast<mlir::func::CallOp>(op));
//         return true;
//       };
//   opRegistry[mlir::func::ReturnOp::getOperationName()] =
//       [](mlir::Operation *op) { return true; };
//   opRegistry[mlir::scf::YieldOp::getOperationName()] = [](mlir::Operation
//   *op) {
//     return true;
//   };
// }

void Interpreter::runFromEntry(mlir::ModuleOp runModule,
                               llvm::StringRef entryFuncName) {
  module = runModule;
  auto entryFunc = module.lookupSymbol<mlir::func::FuncOp>(entryFuncName);
  if (!entryFunc || entryFunc.getBlocks().empty()) {
    throw std::runtime_error("Entry function not found.");
  }

  runFunction(entryFunc);
}

void Interpreter::runAll(mlir::ModuleOp runModule) {
  module = runModule;
  for (mlir::Operation &op : module.getBody()->getOperations()) {
    if (auto fn = llvm::dyn_cast<mlir::func::FuncOp>(op)) {
      if (fn.getBlocks().empty())
        continue;
      std::cout << "-- " << fn.getName().str() << " --\n";
      runFunction(fn);
    }
  }
}

void Interpreter::runFunction(mlir::func::FuncOp fn) {
  struct StackFrame frame;
  frame.funcName = fn.getName();
  callStack.push_back(frame);

  std::vector<RuntimeValue> results;

  try {
    results = execute(fn.getBlocks().front());
  } catch (const std::exception &e) {
    std::cerr << "\nFatal Execution Error: " << e.what() << "\n";
    printStackTrace();
    exit(1);
  }
  for (RuntimeValue result : results) {
    printRuntimeValue(result);
    std::cout << std::endl;
  }
}

void Interpreter::printStackTrace() {
  std::cerr << "Stack trace: \n";
  for (auto it = callStack.rbegin(); it != callStack.rend(); it++) {
    std::cerr << " at @" << it->funcName << "\n";
  }
}

std::vector<RuntimeValue> Interpreter::execute(mlir::Block &block) {
  for (mlir::Operation &op : block) {
    std::cout << "Ops: " << op.getName().getStringRef().str() << std::endl;
    dispatch(&op);
  }

  std::vector<RuntimeValue> results;
  mlir::Operation *terminator = block.getTerminator();

  if (auto returnOp = llvm::dyn_cast<mlir::func::ReturnOp>(terminator)) {
    for (mlir::Value operand : returnOp.getOperands()) {
      results.push_back(currentFrame().state[operand]);
    }
  }

  return results;
}

bool Interpreter::dispatch(mlir::Operation *op) {
  llvm::StringRef opName = op->getName().getStringRef();
  auto it = opRegistry.find(opName);

  if (it != opRegistry.end()) {
    return it->second(*this, op);
  }

  throw std::runtime_error("Unsupported op: " + opName.str());
  return false;
}

std::vector<int64_t>
Interpreter::extractIndices(mlir::Operation::operand_range indicesOps) {
  std::vector<int64_t> indices;
  for (auto idxVal : indicesOps) {
    indices.push_back(getValue(idxVal).get<int64_t>());
  }
  return indices;
}

void Interpreter::printRuntimeValue(const RuntimeValue &val) {
  std::visit([](const auto &v) { std::cout << v; }, val.data);
}