// RUN: %not %lim --entry=main %s 2>&1 | %FileCheck %s

// CHECK: Stack trace:
// CHECK-NEXT: at @trigger_crash
// CHECK-NEXT: at @level_2
// CHECK-NEXT: at @level_1
// CHECK-NEXT: at @main

func.func @trigger_crash(%arg0: f32) -> f32 {
  %crash = arith.subf %arg0, %arg0 : f32
  return %crash : f32
}

func.func @level_2(%arg0: f32) -> f32 {
  %res = func.call @trigger_crash(%arg0) : (f32) -> f32
  return %res : f32
}

func.func @level_1(%arg0: f32) -> f32 {
  %res = func.call @level_2(%arg0) : (f32) -> f32
  return %res : f32
}

func.func @main() -> f32 {
  %cst = arith.constant 42.0 : f32
  %res = func.call @level_1(%cst) : (f32) -> f32
  return %res : f32
}