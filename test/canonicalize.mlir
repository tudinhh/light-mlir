func.func @main() {
    %a = arith.constant 1.0 : f32
    %zero = arith.constant 0.0 : f32
    %sum = arith.addf %a, %zero : f32
    return
}