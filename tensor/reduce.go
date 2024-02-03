package tensor

import (
	"gograd/tensor/internal/device"
	types "gograd/tensor/types"
	"strconv"
	"strings"
)

func reduce_shape[T types.TensorType](
	init_dims int,
	value T,
	keep_dims bool,
) *Tensor[T] {
	if !keep_dims {
		return Scalar[T](value)
	} else {
		ones := make(types.Shape, init_dims)
		for i := range ones {
			ones[i] = 1
		}
		return CreateTensor([]T{value}, ones)
	}
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
	if tensor.Err != nil {
		return tensor
	}
	// create args for IndexAdv. Initially [":",":",..."0",...]
	args := make([]string, 0, len(tensor.Shape()))
	for i := 0; i < len(tensor.Shape()); i++ {
		args = append(args, ":")
	}
	args[axis] = "0"

	// get sub tensor by axis
	reduced := tensor.IndexAdv(strings.Join(args, ","))
	dim := int(tensor.Shape()[axis])
	// iterate over remaining subtensors along axis and sum them
	for i := 1; i < dim; i++ {
		args[axis] = strconv.Itoa(i)
		reduced.Add(tensor.IndexAdv(strings.Join(args, ",")), reduced)
	}
	if keep_dims {
		return reduced.Unsqueeze(axis, nil)
	}
	return reduced
}

func (tensor *Tensor[T]) Sum(keep_dims bool) *Tensor[T] {
	if tensor.Err != nil {
		return tensor
	}
	sum := []T{0}
	device.Sum(AUTO_IMPL, tensor.data(), sum)
	return reduce_shape(len(tensor.Shape()), sum[0], keep_dims)
}

func (tensor *Tensor[T]) Mean(keep_dims bool) *Tensor[T] {
	if tensor.Err != nil {
		return tensor
	}
	sum := []T{0}
	device.Sum(AUTO_IMPL, tensor.data(), sum)

	size := len(tensor.data())
	_mean := T(float64(sum[0]) / float64(size))
	return reduce_shape(len(tensor.Shape()), _mean, keep_dims)
}

func (tensor *Tensor[T]) Max(keep_dims bool) *Tensor[T] {
	if tensor.Err != nil {
		return tensor
	}
	max := []T{tensor.data()[0]}
	device.Max(AUTO_IMPL, tensor.data(), max)
	return reduce_shape(len(tensor.Shape()), max[0], keep_dims)
}

func (tensor *Tensor[T]) Min(keep_dims bool) *Tensor[T] {
	if tensor.Err != nil {
		return tensor
	}
	min := []T{tensor.data()[0]}
	device.Min(AUTO_IMPL, tensor.data(), min)
	return reduce_shape(len(tensor.Shape()), min[0], keep_dims)
}
