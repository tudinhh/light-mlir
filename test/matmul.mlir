func.func @main() -> f32 {
    // 1. Constants for indices and values
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    
    %f0 = arith.constant 0.0 : f32
    %f1 = arith.constant 1.0 : f32
    %f2 = arith.constant 2.0 : f32

    // 2. Allocate 2D buffers
    %A = memref.alloc() : memref<2x2xf32>
    %B = memref.alloc() : memref<2x2xf32>
    %C = memref.alloc() : memref<2x2xf32>

    // 3. Initialize Matrix A: [[1.0, 2.0], [1.0, 2.0]]
    memref.store %f1, %A[%c0, %c0] : memref<2x2xf32>
    memref.store %f2, %A[%c0, %c1] : memref<2x2xf32>
    memref.store %f1, %A[%c1, %c0] : memref<2x2xf32>
    memref.store %f2, %A[%c1, %c1] : memref<2x2xf32>

    // 4. Initialize Matrix B: [[2.0, 0.0], [0.0, 2.0]] (Identity * 2)
    memref.store %f2, %B[%c0, %c0] : memref<2x2xf32>
    memref.store %f0, %B[%c0, %c1] : memref<2x2xf32>
    memref.store %f0, %B[%c1, %c0] : memref<2x2xf32>
    memref.store %f2, %B[%c1, %c1] : memref<2x2xf32>

    // 5. Initialize Matrix C: [[0.0, 0.0], [0.0, 0.0]] (Required for accumulation)
    memref.store %f0, %C[%c0, %c0] : memref<2x2xf32>
    memref.store %f0, %C[%c0, %c1] : memref<2x2xf32>
    memref.store %f0, %C[%c1, %c0] : memref<2x2xf32>
    memref.store %f0, %C[%c1, %c1] : memref<2x2xf32>

    // 6. The High-Level Matrix Multiplication
    // The PassManager will expand this into 3 nested scf.for loops
    linalg.matmul ins(%A, %B : memref<2x2xf32>, memref<2x2xf32>) 
                  outs(%C : memref<2x2xf32>)

    // 7. Extract C[0, 1]. 
    // Math: (1.0 * 0.0) + (2.0 * 2.0) = 4.0
    %res = memref.load %C[%c0, %c1] : memref<2x2xf32>

    // 8. Cleanup memory
    memref.dealloc %A : memref<2x2xf32>
    memref.dealloc %B : memref<2x2xf32>
    memref.dealloc %C : memref<2x2xf32>

    return %res : f32
}