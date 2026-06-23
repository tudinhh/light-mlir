#include "Interpreter.h"

#include <iostream>

Interpreter::Interpreter() { registerOps(); }

void Interpreter::registerOps() {
  opRegistry[mlir::arith::AddFOp::getOperationName()] =
      [this](mlir::Operation *op) {
        this->evalAddF(llvm::cast<mlir::arith::AddFOp>(op));
        return true;
      };

  opRegistry[mlir::arith::AddIOp::getOperationName()] =
      [this](mlir::Operation *op) {
        this->evalAddI(llvm::cast<mlir::arith::AddIOp>(op));
        return true;
      };
  opRegistry[mlir::arith::MulFOp::getOperationName()] =
      [this](mlir::Operation *op) {
        this->evalMulF(llvm::cast<mlir::arith::MulFOp>(op));
        return true;
      };

  opRegistry[mlir::scf::IfOp::getOperationName()] =
      [this](mlir::Operation *op) {
        this->evalIf(llvm::cast<mlir::scf::IfOp>(op));
        return true;
      };

  opRegistry[mlir::arith::ConstantOp::getOperationName()] =
      [this](mlir::Operation *op) {
        this->evalConstant(llvm::cast<mlir::arith::ConstantOp>(op));
        return true;
      };

  opRegistry[mlir::memref::AllocOp::getOperationName()] =
      [this](mlir::Operation *op) {
        this->evalAlloc(llvm::cast<mlir::memref::AllocOp>(op));
        return true;
      };

  opRegistry[mlir::memref::DeallocOp::getOperationName()] =
      [this](mlir::Operation *op) {
        this->evalDealloc(llvm::cast<mlir::memref::DeallocOp>(op));
        return true;
      };

  opRegistry[mlir::memref::StoreOp::getOperationName()] =
      [this](mlir::Operation *op) {
        this->evalStore(llvm::cast<mlir::memref::StoreOp>(op));
        return true;
      };

  opRegistry[mlir::memref::LoadOp::getOperationName()] =
      [this](mlir::Operation *op) {
        this->evalLoad(llvm::cast<mlir::memref::LoadOp>(op));
        return true;
      };

  opRegistry[mlir::scf::ForOp::getOperationName()] =
      [this](mlir::Operation *op) {
        this->evalFor(llvm::cast<mlir::scf::ForOp>(op));
        return true;
      };

  opRegistry[mlir::func::CallOp::getOperationName()] =
      [this](mlir::Operation *op) {
        this->evalCall(llvm::cast<mlir::func::CallOp>(op));
        return true;
      };
  opRegistry[mlir::func::ReturnOp::getOperationName()] =
      [](mlir::Operation *op) { return true; };
  opRegistry[mlir::scf::YieldOp::getOperationName()] = [](mlir::Operation *op) {
    return true;
  };
}

void Interpreter::run(mlir::ModuleOp runModule, llvm::StringRef entryFuncName) {
  module = runModule;
  auto entryFunc = module.lookupSymbol<mlir::func::FuncOp>(entryFuncName);
  if (!entryFunc || entryFunc.getBlocks().empty()) {
    throw std::runtime_error("Error: Entry function not found.");
  }

  struct StackFrame entryFrame;
  entryFrame.funcName = entryFuncName;
  callStack.push_back(entryFrame);

  std::vector<RuntimeValue> results;

  try {
    results = execute(entryFunc.getBlocks().front());
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
    return it->second(op);
  }

  throw std::runtime_error("Unsupported op: " + opName.str());
  return false;
}

void Interpreter::evalIf(mlir::scf::IfOp op) {
  int32_t conditionVal = currentFrame().state[op.getCondition()].get<int32_t>();
  bool condition = (conditionVal != 0);
  mlir::Block *blockToExecute = nullptr;
  if (condition) {
    blockToExecute = op.thenBlock();
  } else if (op.elseBlock()) {
    blockToExecute = op.elseBlock();
  }

  if (blockToExecute) {
    execute(*blockToExecute);

    auto yieldOp =
        llvm::cast<mlir::scf::YieldOp>(blockToExecute->getTerminator());

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

  std::vector<RuntimeValue> results = execute(entryBlock);
  callStack.pop_back();

  for (auto i = 0; i < results.size(); i++) {
    currentFrame().state[op.getResult(i)] = results[i];
  }
}

void Interpreter::evalConstant(mlir::arith::ConstantOp op) {
  if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(op.getValue())) {
    // Check if it's an 'index' type or standard int
    if (op.getType().isIndex()) {
      currentFrame().state[op.getResult()] = RuntimeValue{intAttr.getInt()};
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

void Interpreter::evalAddI(mlir::arith::AddIOp op) {
  float lhs = currentFrame().state[op.getLhs()].get<int>();
  float rhs = currentFrame().state[op.getRhs()].get<int>();
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
  // Step 1: Store the initial iter_args in a local C++ vector
  std::vector<RuntimeValue> localIterArgs;
  for (mlir::Value initArg : op.getInitArgs()) {
    localIterArgs.push_back(currentFrame().state[initArg]);
  }

  // Retrieve the evaluated bounds (assuming MLIR index types are stored as
  // 'long')
  long lowerBound = currentFrame().state[op.getLowerBound()].get<long>();
  long upperBound = currentFrame().state[op.getUpperBound()].get<long>();
  long step = currentFrame().state[op.getStep()].get<long>();

  mlir::Block *block = op.getBody();

  // Step 2: Start the main C++ loop
  for (long i = lowerBound; i < upperBound; i += step) {
    // Step 3: Map the loop index to the block's first argument
    currentFrame().state[block->getArgument(0)] = RuntimeValue{i};

    // Step 4: Map your local iter_args to the remaining block arguments
    for (size_t j = 0; j < localIterArgs.size(); ++j) {
      currentFrame().state[block->getArgument(j + 1)] = localIterArgs[j];
    }

    // Step 5: Execute the block's operations (excluding the terminator)
    for (mlir::Operation &nestedOp : block->without_terminator()) {
      dispatch(&nestedOp);
    }

    // Step 6 & 7: Extract yielded values and overwrite local iter_args
    auto yieldOp = llvm::cast<mlir::scf::YieldOp>(block->getTerminator());
    for (size_t j = 0; j < yieldOp.getOperands().size(); ++j) {
      localIterArgs[j] = currentFrame().state[yieldOp.getOperand(j)];
    }
  }

  // Step 8: Assign the final iter_args vector to the operation's results
  for (size_t j = 0; j < op.getNumResults(); ++j) {
    currentFrame().state[op.getResult(j)] = localIterArgs[j];
  }
}

void Interpreter::printRuntimeValue(const RuntimeValue &val) {
  std::visit([](const auto &v) { std::cout << v; }, val.data);
}