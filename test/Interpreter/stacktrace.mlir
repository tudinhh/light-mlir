// 4. The bottom of the stack. 
// This will crash because 'arith.subf' is not in our C++ dispatch switch.
func.func @trigger_crash(%arg0: f32) -> f32 {
    %crash = arith.subf %arg0, %arg0 : f32
    return %crash : f32
}

// 3. Middle layer
func.func @level_2(%arg0: f32) -> f32 {
    %res = func.call @trigger_crash(%arg0) : (f32) -> f32
    return %res : f32
}

// 2. First layer
func.func @level_1(%arg0: f32) -> f32 {
    %res = func.call @level_2(%arg0) : (f32) -> f32
    return %res : f32
}

// 1. Entry point
func.func @main() -> f32 {
    %cst = arith.constant 42.0 : f32
    %res = func.call @level_1(%cst) : (f32) -> f32
    return %res : f32
}