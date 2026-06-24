cmake_minimum_required(VERSION 3.20)
project(light-mlir CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_PREFIX_PATH "/home/anhtu/llvm-project/build")
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake from ${MLIR_DIR}")

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})

add_subdirectory(test)

add_executable(lim src/main.cpp src/Interpreter.cpp 
    src/eval/ArithOps.cpp
    src/eval/FuncOps.cpp
    src/eval/LinalgOps.cpp
    src/eval/MemRefOps.cpp
    src/eval/ScfOps.cpp)
target_include_directories(lim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link essential MLIR core and dialect libraries
target_link_libraries(lim
    PRIVATE
    MLIRIR
    MLIRParser
    MLIRSupport
    MLIRArithDialect
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRPass
    MLIRTransforms
    MLIRLinalgTransforms
    MLIRLinalgDialect
)