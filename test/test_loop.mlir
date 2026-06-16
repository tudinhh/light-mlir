func.func @main() {
    %lb = arith.constant 2 : index
    %ub = arith.constant 11 : index
    %step = arith.constant 3 : index
    
    scf.for %iv = %lb to %ub step %step {
        // The body can remain empty for this test
    }
    return
}