package tensor

import types "gograd/tensor/types"

func reduce_shape[T types.TensorType](
	init_tensor *Tensor[T],
	value T,
	keep_dims bool,
) *Tensor[T] {
	if !keep_dims {
		return Scalar[T](value)
	} else {
		ones := make(types.Shape, len(init_tensor.Shape()))
		for i := range ones {
			ones[i] = 1
		}
		return CreateTensor[T]([]T{value}, ones)
	}
}

func (tensor *Tensor[T]) Sum(keep_dims bool) *Tensor[T] {
	// TODO to be able to specify axis for sum
	var sum T = 0
	for _, val := range tensor.data() {
		sum += val
	}
	return reduce_shape(tensor, T(sum), keep_dims)
}

func (tensor *Tensor[T]) Mean(keep_dims bool) *Tensor[T] {
	var sum T = 0
	for _, val := range tensor.data() {
		sum += val
	}
	size := T(len(tensor.data()))
	return reduce_shape(tensor, T(float64(sum/size)), keep_dims)
}
