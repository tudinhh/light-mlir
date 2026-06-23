func.func @main() -> f32 {
    %true_cond = arith.constant true
    %false_cond = arith.constant false
    %val_a = arith.constant 10.0 : f32
    %val_b = arith.constant 20.0 : f32

    // Test 1: Should execute 'then' block and return val_a (10.0)
    %res1 = scf.if %true_cond -> (f32) {
        scf.yield %val_a : f32
    } else {
        scf.yield %val_b : f32
    }

    // Test 2: Should execute 'else' block and return val_a (10.0)
    %res2 = scf.if %false_cond -> (f32) {
        scf.yield %val_b : f32
    } else {
        scf.yield %val_a : f32
    }

    // Add results together: 10.0 + 10.0 = 20.0
    %final = arith.addf %res1, %res2 : f32
    return %final : f32
}