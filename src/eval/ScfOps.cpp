#include "eval.h"

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