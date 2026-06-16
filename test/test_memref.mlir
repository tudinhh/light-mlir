func.func @main() -> f32 {
    // 1. Allocate a 1D memref of size 2
    %mem = memref.alloc() : memref<2xf32>
    
    // 2. Define indices and float constants
    %idx0 = arith.constant 0 : index
    %idx1 = arith.constant 1 : index
    %val0 = arith.constant 3.500000e+00 : f32
    %val1 = arith.constant 4.250000e+00 : f32
    
    // 3. Store values into the allocated memref
    memref.store %val0, %mem[%idx0] : memref<2xf32>
    memref.store %val1, %mem[%idx1] : memref<2xf32>
    
    // 4. Load the values back from memory
    %ret0 = memref.load %mem[%idx0] : memref<2xf32>
    %ret1 = memref.load %mem[%idx1] : memref<2xf32>
    
    // 5. Compute the sum (3.5 + 4.25)
    %sum = arith.addf %ret0, %ret1 : f32
    
    // 6. Free the buffer
    memref.dealloc %mem : memref<2xf32>
    
    return %sum : f32
}