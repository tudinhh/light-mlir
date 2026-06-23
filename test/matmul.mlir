// RUN: %lim %s | %FileCheck %s
// CHECK: 4
func.func @main() -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    
    %f0 = arith.constant 0.0 : f32
    %f1 = arith.constant 1.0 : f32
    %f2 = arith.constant 2.0 : f32

    %A = memref.alloc() : memref<2x2xf32>
    %B = memref.alloc() : memref<2x2xf32>
    %C = memref.alloc() : memref<2x2xf32>

    memref.store %f1, %A[%c0, %c0] : memref<2x2xf32>
    memref.store %f2, %A[%c0, %c1] : memref<2x2xf32>
    memref.store %f1, %A[%c1, %c0] : memref<2x2xf32>
    memref.store %f2, %A[%c1, %c1] : memref<2x2xf32>

    memref.store %f2, %B[%c0, %c0] : memref<2x2xf32>
    memref.store %f0, %B[%c0, %c1] : memref<2x2xf32>
    memref.store %f0, %B[%c1, %c0] : memref<2x2xf32>
    memref.store %f2, %B[%c1, %c1] : memref<2x2xf32>

    memref.store %f0, %C[%c0, %c0] : memref<2x2xf32>
    memref.store %f0, %C[%c0, %c1] : memref<2x2xf32>
    memref.store %f0, %C[%c1, %c0] : memref<2x2xf32>
    memref.store %f0, %C[%c1, %c1] : memref<2x2xf32>

    linalg.matmul ins(%A, %B : memref<2x2xf32>, memref<2x2xf32>) 
                  outs(%C : memref<2x2xf32>)

    %res = memref.load %C[%c0, %c1] : memref<2x2xf32>

    memref.dealloc %A : memref<2x2xf32>
    memref.dealloc %B : memref<2x2xf32>
    memref.dealloc %C : memref<2x2xf32>

    return %res : f32
}