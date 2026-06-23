#include "Interpreter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SourceMgr.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <path to .mlir file>\n";
    return 1;
  }

  mlir::MLIRContext context;
  mlir::DialectRegistry registry;

  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                  mlir::linalg::LinalgDialect>();
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
  if (!module) {
    std::cerr << "Failed to parse MLIR file.\n";
    return 1;
  }

  try {
    Interpreter interpreter;
    interpreter.run(*module);
  } catch (const std::exception& e) {
    std::cerr << "Runtime Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}