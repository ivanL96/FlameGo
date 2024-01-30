package types

func Input_b_scalar_to_float32[T TensorType](a []T, b T, out []T) ([]float32, float32, []float32) {
	afl, ok_a := any(a).([]float32)
	bfl, ok_b := any(b).(float32)
	cfl, ok_c := any(out).([]float32)
	if !ok_a || !ok_b || !ok_c {
		return nil, 0, nil
	}
	return afl, bfl, cfl
}

func Input_to_float32[T TensorType](a, b, out []T) ([]float32, []float32, []float32) {
	afl, ok_a := any(a).([]float32)
	bfl, ok_b := any(b).([]float32)
	cfl, ok_c := any(out).([]float32)
	if !ok_a || !ok_b || !ok_c {
		return nil, nil, nil
	}
	return afl, bfl, cfl
}

func Reduce_input_to_float32[T TensorType](a, out []T) ([]float32, []float32) {
	afl, ok_a := any(a).([]float32)
	cfl, ok_c := any(out).([]float32)
	if !ok_a || !ok_c {
		return nil, nil
	}
	return afl, cfl
}
