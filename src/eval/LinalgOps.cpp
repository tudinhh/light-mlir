#include "eval.h"

void Interpreter::evalMatmul(mlir::linalg::MatmulOp op) {
  // Extract the memrefs for A, B, and C
  // Standard linalg.matmul operand order: [A, B, C]
  auto memrefA =
      currentFrame().state[op.getOperand(0)].get<std::shared_ptr<MemRef>>();
  auto memrefB =
      currentFrame().state[op.getOperand(1)].get<std::shared_ptr<MemRef>>();
  auto memrefC =
      currentFrame().state[op.getOperand(2)].get<std::shared_ptr<MemRef>>();

  // Extract dimensions
  // A is (M x K), B is (K x N), C is (M x N)
  int64_t M = memrefA->shape[0];
  int64_t K = memrefA->shape[1];
  int64_t N = memrefB->shape[1];

  // Execute the standard O(N^3) matrix multiplication
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      for (int64_t k = 0; k < K; ++k) {
        float valA = memrefA->data[memrefA->getFlatIndex({i, k})];
        float valB = memrefB->data[memrefB->getFlatIndex({k, j})];

        // Linalg matmul acts as an accumulator: C[i, j] += A[i, k] * B[k, j]
        memrefC->data[memrefC->getFlatIndex({i, j})] += valA * valB;
      }
    }
  }
}