package tensor

import (
	"gograd/tensor/internal/device"
	types "gograd/tensor/types"
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
	if len(tensor.Shape()) == 2 {
		tensor = tensor.AsContinuous()
		out_shape := tensor.Shape().ReduceDim(int(axis))
		out := CreateEmptyTensor[T](out_shape...)
		device.SumAxis(AUTO_IMPL, tensor.Data(), out.Data(), tensor.Shape(), int(axis))
		if !keep_dims {
			return out.Squeeze()
		}
		return out
	}

	args := make([]*idxRange, len(tensor.Shape()))
	for i := 0; i < len(tensor.Shape()); i++ {
		if i == int(axis) {
			args[i] = I(0)
			continue
		}
		args[i] = Axis()
	}

	// get sub tensor by axis
	reduced := tensor.IndexAdv_(args...)
	dim := int(tensor.Shape()[axis])
	// iterate over remaining subtensors along axis and sum them
	for i := 1; i < dim; i++ {
		args[axis] = I(i)
		reduced.Add(tensor.IndexAdv_(args...), reduced)
	}
	if keep_dims {
		return reduced.Unsqueeze(int(axis))
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
