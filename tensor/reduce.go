package tensor

import (
	"gograd/tensor/internal/cpu"
	types "gograd/tensor/types"
	"strconv"
	"strings"
)

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
	sum := []T{0}
	cpu.Sum(auto_impl, tensor.data(), sum)
	return reduce_shape(tensor, sum[0], keep_dims)
}

// Example:
// a = [
// [1,2],
// [3,4]]
// a.SumAlongAxis(0, false) => [4, 6]
// a.SumAlongAxis(0, true) => [[4, 6]]
func (tensor *Tensor[T]) SumAlongAxis(
	axis uint,
	keep_dims bool,
) *Tensor[T] {
	// create args for IndexAdv. Initially [":",":",..."0",...]
	args := make([]string, 0, len(tensor.Shape()))
	for i := 0; i < len(tensor.Shape()); i++ {
		args = append(args, ":")
	}
	args[axis] = "0"
	reduced := tensor.IndexAdv(strings.Join(args, ","))
	dim := int(tensor.Shape()[axis])
	for i := 1; i < dim; i++ {
		args[axis] = strconv.Itoa(i)
		reduced.Add(tensor.IndexAdv(strings.Join(args, ",")), reduced)
	}
	if keep_dims {
		return reduced.Unsqueeze(axis, nil)
	}
	return reduced
}

func (tensor *Tensor[T]) Mean(keep_dims bool) *Tensor[T] {
	var sum T = 0
	for _, val := range tensor.data() {
		sum += val
	}
	size := T(len(tensor.data()))
	_mean := T(float64(sum) / float64(size))
	return reduce_shape(tensor, _mean, keep_dims)
}
