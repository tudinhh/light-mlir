#include "eval.h"

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
  std::cout << "-- " + calleeFrame.funcName + " --\n";
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