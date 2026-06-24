#include "eval.h"

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

void Interpreter::evalMulF(mlir::arith::MulFOp op) {
  float lhs = currentFrame().state[op.getLhs()].get<float>();
  float rhs = currentFrame().state[op.getRhs()].get<float>();
  currentFrame().state[op.getResult()] = RuntimeValue{lhs * rhs};
}