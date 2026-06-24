#include "OpRegistry.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

REGISTER_OP(MatmulOp, mlir::linalg::MatmulOp) {
  auto memrefA =
      interp.getValue(op.getOperand(0)).get<std::shared_ptr<MemRef>>();
  auto memrefB =
      interp.getValue(op.getOperand(1)).get<std::shared_ptr<MemRef>>();
  auto memrefC =
      interp.getValue(op.getOperand(2)).get<std::shared_ptr<MemRef>>();

  int64_t M = memrefA->shape[0];
  int64_t K = memrefA->shape[1];
  int64_t N = memrefB->shape[1];

  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      for (int64_t k = 0; k < K; ++k) {
        float valA = memrefA->data[memrefA->getFlatIndex({i, k})];
        float valB = memrefB->data[memrefB->getFlatIndex({k, j})];

        memrefC->data[memrefC->getFlatIndex({i, j})] += valA * valB;
      }
    }
  }
}
