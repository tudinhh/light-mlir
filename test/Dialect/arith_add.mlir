// RUN: %lim %s | %FileCheck %s

func.func @main() -> i32 {
  %c1 = arith.constant 5 : i32
  %c2 = arith.constant 10 : i32
  %res = arith.addi %c1, %c2 : i32
  
  
  func.return %res : i32
  // CHECK: 15
}