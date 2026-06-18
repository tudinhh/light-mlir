func.func @main() {
    %lb = arith.constant 2 : index
    %ub = arith.constant 11 : index
    %step = arith.constant 3 : index
    
    scf.for %iv = %lb to %ub step %step {
        %sum = arith.constant 5: i32
    }
    return
}