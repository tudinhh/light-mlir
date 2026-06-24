#include "eval.h"

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