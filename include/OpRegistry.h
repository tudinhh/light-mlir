#pragma once
#include "Interpreter.h"
#include <functional>

using OpHandler = std::function<bool(Interpreter &, mlir::Operation *)>;

class OpRegistry {
public:
  static OpRegistry &get() {
    static OpRegistry instance;
    return instance;
  }

  llvm::StringMap<OpHandler> handlers;

  template <typename OpT>
  void registerHandler(std::function<void(Interpreter &, OpT)> handler) {
    handlers[OpT::getOperationName()] = [handler](Interpreter &interp,
                                                  mlir::Operation *op) {
      handler(interp, llvm::cast<OpT>(op));
      return true;
    };
  }
};

template <typename OpT> struct AutoRegOp {
  AutoRegOp(std::function<void(Interpreter &, OpT)> handler) {
    OpRegistry::get().registerHandler<OpT>(handler);
  }
};

#define REGISTER_OP(Id, OpT)                                                   \
  static void eval_##Id(Interpreter &interp, OpT op);                          \
  static AutoRegOp<OpT> reg_##Id(eval_##Id);                                   \
  static void eval_##Id(Interpreter &interp, OpT op)