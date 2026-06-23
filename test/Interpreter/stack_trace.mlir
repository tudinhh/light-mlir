
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