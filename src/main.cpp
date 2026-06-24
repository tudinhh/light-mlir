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

  // Handle input arguments
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " <path to .mlir file> [--entry=<func>]\n";
    return 1;
  }

  std::string filePath;
  std::string entryFunc;

  for (int i = 1; i < argc; i++) {
    llvm::StringRef arg(argv[i]);
    if (arg.consume_front("--entry=")) {
      entryFunc = arg.str();
    } else {
      filePath = argv[i];
    }
  }

  if (filePath.empty()) {
    std::cerr << "Usage: " << argv[0]
              << " <path to .mlir file> [--entry=<func>]\n";
    return 1;
  }
  // MLIR setup
  mlir::MLIRContext context;
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                  mlir::linalg::LinalgDialect>();
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();

  // Run MLIR module
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(filePath, &context);
  if (!module) {
    std::cerr << "Failed to parse MLIR file.\n";
    return 1;
  }

  try {
    Interpreter interpreter;
    if (!entryFunc.empty()) {
      interpreter.runFromEntry(*module, entryFunc);
    } else {
      interpreter.runAll(*module);
    }
  } catch (const std::exception& e) {
    std::cerr << "Runtime Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}