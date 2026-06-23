configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  @ONLY)

find_program(LIT_EXECUTABLE 
  NAMES llvm-lit lit
  PATHS ${LLVM_TOOLS_BINARY_DIR}
  NO_DEFAULT_PATH)

if(NOT LIT_EXECUTABLE)
  message(WARNING "llvm-lit not found! Make sure you built 'llvm-lit' in your LLVM build tree.")
endif()

add_custom_target(check-light
  COMMAND ${LIT_EXECUTABLE} -v ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS lim
  COMMENT "Running light-mlir lit tests"
  USES_TERMINAL)