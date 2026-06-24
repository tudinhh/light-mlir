// RUN: %lim %s | %FileCheck %s

// CHECK-LABEL: -- test_addi --
func.func @test_addi() -> i32 {
  %c1 = arith.constant 5 : i32
  %c2 = arith.constant 10 : i32
  %res = arith.addi %c1, %c2 : i32

  // CHECK: 15
  func.return %res : i32
}

// CHECK-LABEL: -- test_addf --
func.func @test_addf() -> f32 {
  %c1 = arith.constant 5.5 : f32
  %c2 = arith.constant 2.0 : f32
  %res = arith.addf %c1, %c2 : f32

  // CHECK: 7.5
  func.return %res : f32
}