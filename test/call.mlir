func.func @add_one(%a1 : f32) -> f32 {
    %one = arith.constant 1.0 : f32
    %a2 = arith.addf %a1, %one : f32
    return %a2 : f32
}

func.func @add_num(%a: f32, %b: f32) -> f32 {
    %sum1 = arith.addf %a, %b : f32
    %sum2 = func.call @add_one(%sum1) : (f32) -> f32
    return %sum2 : f32
}

func.func @main() {
    %a = arith.constant 5.0 : f32
    %b = arith.constant 2.0 : f32
    
    %sum = func.call @add_num(%a, %b) : (f32, f32) -> f32
    return
}