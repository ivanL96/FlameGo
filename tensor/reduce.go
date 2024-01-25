package tensor

func (tensor *Tensor[T]) Sum() T {
	// TODO to be able to specify axis for sum
	var sum T = 0
	for _, val := range tensor.data() {
		sum += val
	}
	return sum
}

func (tensor *Tensor[T]) Mean() float64 {
	var sum T = 0
	for _, val := range tensor.data() {
		sum += val
	}
	size := T(len(tensor.data()))
	return float64(sum / size)
}
