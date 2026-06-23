// RUN: %lim %s | %FileCheck %s
// CHECK: 7.75
func.func @main() -> f32 {
    %mem = memref.alloc() : memref<2xf32>
    
    %idx0 = arith.constant 0 : index
    %idx1 = arith.constant 1 : index
    %val0 = arith.constant 3.500000e+00 : f32
    %val1 = arith.constant 4.250000e+00 : f32
    
    memref.store %val0, %mem[%idx0] : memref<2xf32>
    memref.store %val1, %mem[%idx1] : memref<2xf32>
    
    %ret0 = memref.load %mem[%idx0] : memref<2xf32>
    %ret1 = memref.load %mem[%idx1] : memref<2xf32>
    
    %sum = arith.addf %ret0, %ret1 : f32
    
    memref.dealloc %mem : memref<2xf32>
    
    return %sum : f32
}

