#include "Interpreter.h"

#include <iostream>

void Interpreter::run(mlir::ModuleOp runModule) {
  std::cout << "LOG: --run--" << std::endl;
  module = runModule;
  auto mainFunc = module.lookupSymbol<mlir::func::FuncOp>("main");
  if (!mainFunc || mainFunc.getBlocks().empty()) {
    throw std::runtime_error("Error: 'main' function not found or empty.");
  }

  struct StackFrame mainFrame;
  mainFrame.funcName = "main";
  callStack.push_back(mainFrame);

  try {
    executeBlock(mainFunc.getBlocks().front());
  } catch (const std::exception &e) {
    std::cerr << "\nFatal Execution Error: " << e.what() << "\n";
    printStackTrace();
    exit(1);
  }
}

void Interpreter::printStackTrace() {
  std::cerr << "Stack trace: \n";
  for (auto it = callStack.rbegin(); it != callStack.rend(); it++) {
    std::cerr << " at @" << it->funcName << "\n";
  }
}

std::vector<RuntimeValue> Interpreter::executeBlock(mlir::Block &block) {
  std::cout << "LOG: --executeBlock--" << std::endl;
  // Iterate through the ops
  for (mlir::Operation &op : block) {
    std::cout << "LOG: --executeBlock-- " << op.getName().getStringRef().str()
              << std::endl;
    // Evaluate supported ops
    if (dispatch(&op))
      continue;

    if (auto returnOp = llvm::dyn_cast<mlir::func::ReturnOp>(op)) {
      auto numOperands = returnOp.getNumOperands();
      std::vector<RuntimeValue> results;
      for (int i = 0; i < numOperands; i++) {
        results.push_back(currentFrame().state[returnOp.getOperand(i)]);
      }
      return results;
    }
    throw std::runtime_error("Unsupported op: " +
                             op.getName().getStringRef().str());
  }
  return {};
}

bool Interpreter::dispatch(mlir::Operation *op) {
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
  if (auto mulFOp = llvm::dyn_cast<mlir::arith::MulFOp>(op)) {
    evalMulF(mulFOp);
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
  if (auto callOp = llvm::dyn_cast<mlir::func::CallOp>(op)) {
    evalCall(callOp);
    return true;
  }
  if (auto ifOp = llvm::dyn_cast<mlir::scf::IfOp>(op)) {
    evalIf(ifOp);
    return true;
  }
  return false;
}

void Interpreter::evalIf(mlir::scf::IfOp op) {
  // 1. Read the boolean condition from the current frame.
  // MLIR i1 types are usually stored as bool or int8_t in C++ variants.
  bool condition = currentFrame().state[op.getCondition()].get<bool>();

  // 2. Select the correct block to execute.
  mlir::Block *blockToExecute = nullptr;
  if (condition) {
    blockToExecute = op.thenBlock();
  } else if (op.elseBlock()) {
    blockToExecute = op.elseBlock();
  }

  // 3. Execute the block if one exists
  if (blockToExecute) {
    executeBlock(*blockToExecute);

    // 4. Handle the return values (scf.yield)
    // The last operation in an scf block is always a yield.
    auto yieldOp =
        llvm::cast<mlir::scf::YieldOp>(blockToExecute->getTerminator());

    // Map the yielded values to the results of the scf.if operation
    for (auto [yieldVal, result] :
         llvm::zip(yieldOp.getOperands(), op.getResults())) {
      currentFrame().state[result] = currentFrame().state[yieldVal];
    }
  }
}
void Interpreter::evalCall(mlir::func::CallOp op) {
  std::vector<RuntimeValue> args;

  for (auto operand : op.getOperands()) {
    args.push_back(currentFrame().state[operand]);
  }

  auto callee = module.lookupSymbol<mlir::func::FuncOp>(op.getCallee());
  if (!callee || callee.getBlocks().empty()) {
    throw std::runtime_error("Callee not found or empty");
  }

  StackFrame calleeFrame;
  calleeFrame.funcName = op.getCallee().str();
  callStack.push_back(calleeFrame);
  mlir::Block &entryBlock = callee.getBlocks().front();

  for (int i = 0; i < args.size(); i++) {
    currentFrame().state[entryBlock.getArgument(i)] = args[i];
  }

  std::vector<RuntimeValue> results = executeBlock(entryBlock);
  callStack.pop_back();

  for (auto i = 0; i < results.size(); i++) {
    currentFrame().state[op.getResult(i)] = results[i];
  }
}

void Interpreter::evalConstant(mlir::arith::ConstantOp op) {
  if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(op.getValue())) {
    // Check if it's an 'index' type or standard int
    if (op.getType().isIndex()) {
      currentFrame().state[op.getResult()] =
          RuntimeValue{intAttr.getInt()}; // Stores as int64_t
    } else {
      currentFrame().state[op.getResult()] =
          RuntimeValue{static_cast<int32_t>(intAttr.getInt())};
    }
  } else if (auto floatAttr = llvm::dyn_cast<mlir::FloatAttr>(op.getValue())) {
    currentFrame().state[op.getResult()] =
        RuntimeValue{static_cast<float>(floatAttr.getValue().convertToFloat())};
  }
}

void Interpreter::evalAddF(mlir::arith::AddFOp op) {
  float lhs = currentFrame().state[op.getLhs()].get<float>();
  float rhs = currentFrame().state[op.getRhs()].get<float>();
  currentFrame().state[op.getResult()] = RuntimeValue{lhs + rhs};
}

void Interpreter::evalAlloc(mlir::memref::AllocOp op) {
  auto memRefType = llvm::cast<mlir::MemRefType>(op.getType());
  std::vector<int64_t> shape = memRefType.getShape().vec();
  currentFrame().state[op.getResult()] =
      RuntimeValue{std::make_shared<MemRef>(shape)};
}

void Interpreter::evalDealloc(mlir::memref::DeallocOp op) {
  currentFrame().state.erase(op.getMemref());
}

std::vector<int64_t>
Interpreter::extractIndices(mlir::Operation::operand_range indicesOps) {
  std::vector<int64_t> indices;
  for (auto idxVal : indicesOps) {
    indices.push_back(currentFrame().state[idxVal].get<int64_t>());
  }
  return indices;
}

void Interpreter::evalStore(mlir::memref::StoreOp op) {
  float val = currentFrame().state[op.getValueToStore()].get<float>();
  auto memref =
      currentFrame().state[op.getMemref()].get<std::shared_ptr<MemRef>>();
  auto indices = extractIndices(op.getIndices());
  memref->data[memref->getFlatIndex(indices)] = val;
}

void Interpreter::evalLoad(mlir::memref::LoadOp op) {
  auto memref =
      currentFrame().state[op.getMemref()].get<std::shared_ptr<MemRef>>();
  auto indices = extractIndices(op.getIndices());
  float val = memref->data[memref->getFlatIndex(indices)];
  currentFrame().state[op.getResult()] = RuntimeValue{val};
}

void Interpreter::evalMulF(mlir::arith::MulFOp op) {
  float lhs = currentFrame().state[op.getLhs()].get<float>();
  float rhs = currentFrame().state[op.getRhs()].get<float>();
  currentFrame().state[op.getResult()] = RuntimeValue{lhs * rhs};
}

void Interpreter::evalFor(mlir::scf::ForOp op) {
  int64_t lower = currentFrame().state[op.getLowerBound()].get<int64_t>();
  int64_t upper = currentFrame().state[op.getUpperBound()].get<int64_t>();
  int64_t step = currentFrame().state[op.getStep()].get<int64_t>();

  std::cout << "Loop bound:" << lower << " " << upper << " " << step
            << std::endl;

  mlir::Value iv = op.getInductionVar();
  mlir::Block *bodyBlock = op.getBody();

  for (int64_t i = lower; i < upper; i += step) {
    currentFrame().state[iv] = RuntimeValue{i};
    for (mlir::Operation &bodyOp : bodyBlock->getOperations()) {
      if (llvm::isa<mlir::scf::YieldOp>(bodyOp))
        continue;
      if (!dispatch(&bodyOp)) {
        throw std::runtime_error("Unsupported op inside loop: " +
                                 bodyOp.getName().getStringRef().str());
      }
    }
  }
}

void Interpreter::printValue(const RuntimeValue &val) {
  if (std::holds_alternative<int32_t>(val.data)) {
    std::cout << "Interpreter Output (i32): " << val.get<int32_t>() << "\n";
  } else if (std::holds_alternative<float>(val.data)) {
    std::cout << "Interpreter Output (f32): " << val.get<float>() << "\n";
  }
}