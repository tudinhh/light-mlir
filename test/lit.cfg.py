import lit.formats
import os

config.name = 'lim'
config.test_format = lit.formats.ShTest(True)
config.suffixes = ['.mlir']
config.test_source_root = os.path.dirname(__file__)

config.substitutions.append(('%lim', os.path.join(config.light_mlir_tools_dir, 'lim')))
config.substitutions.append(('%FileCheck', os.path.join(config.llvm_tools_dir, 'FileCheck')))
config.substitutions.append(('%not', os.path.join(config.llvm_tools_dir, 'not')))
config.substitutions.append(('%mlir-opt', os.path.join(config.llvm_tools_dir, 'mlir-opt')))