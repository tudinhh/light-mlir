#include "OpRegistry.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

REGISTER_OP(ForOp, mlir::scf::ForOp) {
  std::vector<RuntimeValue> localIterArgs;
  for (mlir::Value initArg : op.getInitArgs()) {
    localIterArgs.push_back(interp.getValue(initArg));
  }

  int64_t lowerBound = interp.getValue(op.getLowerBound()).get<int64_t>();
  int64_t upperBound = interp.getValue(op.getUpperBound()).get<int64_t>();
  int64_t step = interp.getValue(op.getStep()).get<int64_t>();

  mlir::Block *block = op.getBody();
  for (int64_t i = lowerBound; i < upperBound; i += step) {
    interp.setValue(block->getArgument(0), RuntimeValue{i});

    for (size_t j = 0; j < localIterArgs.size(); ++j) {
      interp.setValue(block->getArgument(j + 1), localIterArgs[j]);
    }

    for (mlir::Operation &nestedOp : block->without_terminator()) {
      interp.dispatch(&nestedOp);
    }

    auto yieldOp = llvm::cast<mlir::scf::YieldOp>(block->getTerminator());
    for (size_t j = 0; j < yieldOp.getOperands().size(); ++j) {
      localIterArgs[j] = interp.getValue(yieldOp.getOperand(j));
    }
  }

  for (size_t j = 0; j < op.getNumResults(); ++j) {
    interp.setValue(op.getResult(j), localIterArgs[j]);
  }
}

REGISTER_OP(IfOp, mlir::scf::IfOp) {
  RuntimeValue condVal = interp.getValue(op.getCondition());
  bool condition = false;

  if (std::holds_alternative<bool>(condVal.data)) {
    condition = condVal.get<bool>();
  } else if (std::holds_alternative<int32_t>(condVal.data)) {
    condition = (condVal.get<int32_t>() != 0);
  } else if (std::holds_alternative<int64_t>(condVal.data)) {
    condition = (condVal.get<int64_t>() != 0);
  } else {
    throw std::runtime_error(
        "IfOp condition is not a valid boolean or integer type");
  }

  mlir::Block *blockToExecute = nullptr;

  if (condition) {
    blockToExecute = op.thenBlock();
  } else if (op.elseBlock()) {
    blockToExecute = op.elseBlock();
  }

  if (blockToExecute) {
    interp.executeBlock(*blockToExecute);

    auto yieldOp =
        llvm::cast<mlir::scf::YieldOp>(blockToExecute->getTerminator());

    for (auto [yieldVal, result] :
         llvm::zip(yieldOp.getOperands(), op.getResults())) {
      interp.setValue(result, interp.getValue(yieldVal));
    }
  }
}

REGISTER_OP(ScfYieldOp, mlir::scf::YieldOp) {}