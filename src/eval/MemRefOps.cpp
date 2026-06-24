#include "OpRegistry.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

REGISTER_OP(AllocOp, mlir::memref::AllocOp) {
  auto memRefType = llvm::cast<mlir::MemRefType>(op.getType());
  std::vector<int64_t> shape = memRefType.getShape().vec();
  interp.getValue(op.getResult()) =
      RuntimeValue{std::make_shared<MemRef>(shape)};
}

REGISTER_OP(DeallocOp, mlir::memref::DeallocOp) {
  interp.eraseValue(op.getMemref());
}

REGISTER_OP(AddIOp, mlir::arith::AddIOp) {
  auto lhs = interp.getValue(op.getLhs()).get<int64_t>();
  auto rhs = interp.getValue(op.getRhs()).get<int64_t>();
  interp.setValue(op.getResult(), RuntimeValue{lhs + rhs});
}

REGISTER_OP(StoreOp, mlir::memref::StoreOp) {
  float val = interp.getValue(op.getValueToStore()).get<float>();
  auto memref = interp.getValue(op.getMemref()).get<std::shared_ptr<MemRef>>();
  auto indices = interp.extractIndices(op.getIndices());

  memref->data[memref->getFlatIndex(indices)] = val;
}

REGISTER_OP(LoadOp, mlir::memref::LoadOp) {
  auto memref = interp.getValue(op.getMemref()).get<std::shared_ptr<MemRef>>();
  auto indices = interp.extractIndices(op.getIndices());
  float val = memref->data[memref->getFlatIndex(indices)];

  interp.setValue(op.getResult(), RuntimeValue{val});
}