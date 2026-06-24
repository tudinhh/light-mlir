#include "OpRegistry.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

REGISTER_OP(AddIOp, mlir::arith::AddIOp) {
  RuntimeValue lhsVal = interp.getValue(op.getLhs());
  RuntimeValue rhsVal = interp.getValue(op.getRhs());

  std::visit(
      [&](auto &&lhs, auto &&rhs) {
        using LType = std::decay_t<decltype(lhs)>;
        using RType = std::decay_t<decltype(rhs)>;

        if constexpr (std::is_same_v<LType, RType> &&
                      std::is_integral_v<LType> &&
                      !std::is_same_v<LType, bool>) {
          interp.setValue(op.getResult(), RuntimeValue{lhs + rhs});
        } else {
          throw std::runtime_error("Type mismatch or non-integer in AddIOp");
        }
      },
      lhsVal.data, rhsVal.data);
}

REGISTER_OP(AddFOp, mlir::arith::AddFOp) {
  auto lhs = interp.getValue(op.getLhs()).get<float>();
  auto rhs = interp.getValue(op.getRhs()).get<float>();
  interp.setValue(op.getResult(), RuntimeValue{lhs + rhs});
}

REGISTER_OP(ConstantOp, mlir::arith::ConstantOp) {
  if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(op.getValue())) {
    if (op.getType().isIndex()) {
      // Must explicitly cast to int64_t
      interp.setValue(op.getResult(),
                      RuntimeValue{static_cast<int64_t>(intAttr.getInt())});
    } else {
      interp.setValue(op.getResult(),
                      RuntimeValue{static_cast<int32_t>(intAttr.getInt())});
    }
  } else if (auto floatAttr = llvm::dyn_cast<mlir::FloatAttr>(op.getValue())) {
    interp.setValue(op.getResult(),
                    RuntimeValue{static_cast<float>(
                        floatAttr.getValue().convertToFloat())});
  }
}

REGISTER_OP(MulFOp, mlir::arith::MulFOp) {
  float lhs = interp.getValue(op.getLhs()).get<float>();
  float rhs = interp.getValue(op.getRhs()).get<float>();
  interp.setValue(op.getResult(), RuntimeValue{lhs * rhs});
}