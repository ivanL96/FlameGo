package tensor

func (tensor *Tensor[T]) Sum() T {
	// TODO to be able to specify axis for sum
	var sum T = 0
	for _, val := range tensor.data() {
		sum += val
	}
	return sum
}
