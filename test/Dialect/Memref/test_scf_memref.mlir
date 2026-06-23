// RUN: %lim %s | %FileCheck %s
// CHECK: 7.5

func.func @main() -> f32 {
    %mem = memref.alloc() : memref<3xf32>
    
    %lb = arith.constant 0 : index
    %ub = arith.constant 3 : index
    %step = arith.constant 1 : index
    
    %val = arith.constant 2.500000e+00 : f32
    
    scf.for %iv = %lb to %ub step %step {
        memref.store %val, %mem[%iv] : memref<3xf32>
    }
    
    %idx0 = arith.constant 0 : index
    %idx1 = arith.constant 1 : index
    %idx2 = arith.constant 2 : index
    
    %r0 = memref.load %mem[%idx0] : memref<3xf32>
    %r1 = memref.load %mem[%idx1] : memref<3xf32>
    %r2 = memref.load %mem[%idx2] : memref<3xf32>
    
    %sum0 = arith.addf %r0, %r1 : f32
    %sum1 = arith.addf %sum0, %r2 : f32
    
    memref.dealloc %mem : memref<3xf32>
    return %sum1 : f32
}