// RUN: %lim %s | %FileCheck %s
// CHECK: 15

func.func @main() -> f32 {
    %lb = arith.constant 2 : index
    %ub = arith.constant 11 : index
    %step = arith.constant 3 : index
    
    %init_sum = arith.constant 0.0 : f32
    %c5 = arith.constant 5.0 : f32
    
    %final_sum = scf.for %iv = %lb to %ub step %step iter_args(%loop_sum = %init_sum) -> (f32) {
        %next_sum = arith.addf %loop_sum, %c5 : f32
        scf.yield %next_sum : f32
    }
    
    return %final_sum : f32
}