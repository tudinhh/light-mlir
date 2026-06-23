// RUN: %lim %s | %FileCheck %s

func.func @main() -> f32 {
    %true_cond = arith.constant true
    %false_cond = arith.constant false
    %val_a = arith.constant 10.0 : f32
    %val_b = arith.constant 20.0 : f32

    %res1 = scf.if %true_cond -> (f32) {
        scf.yield %val_a : f32
    } else {
        scf.yield %val_b : f32
    }

    %res2 = scf.if %false_cond -> (f32) {
        scf.yield %val_b : f32
    } else {
        scf.yield %val_a : f32
    }

    %final = arith.addf %res1, %res2 : f32
    return %final : f32
    // CHECK: 20
}