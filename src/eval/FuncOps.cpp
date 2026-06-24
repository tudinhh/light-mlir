#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "OpRegistry.h"

REGISTER_OP(CallOp, mlir::func::CallOp) {
  std::vector<RuntimeValue> args;

  for (auto operand : op.getOperands()) {
    args.push_back(interp.getValue(operand));
  }

  auto callee =
      interp.getModule().lookupSymbol<mlir::func::FuncOp>(op.getCallee());
  if (!callee || callee.getBlocks().empty()) {
    throw std::runtime_error("Callee not found or empty");
  }

  StackFrame calleeFrame;
  calleeFrame.funcName = op.getCallee().str();
  std::cout << "-- " + calleeFrame.funcName + " --\n";

  interp.pushFrame(calleeFrame);
  mlir::Block &entryBlock = callee.getBlocks().front();

  for (size_t i = 0; i < args.size(); i++) {
    interp.setValue(entryBlock.getArgument(i), args[i]);
  }
  std::vector<RuntimeValue> results = interp.executeBlock(entryBlock);
  interp.popFrame();

  for (size_t i = 0; i < results.size(); i++) {
    interp.setValue(op.getResult(i), results[i]);
  }
}

REGISTER_OP(FuncReturnOp, mlir::func::ReturnOp) {}