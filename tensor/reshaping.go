package tensor

import (
	"fmt"
	types "gograd/tensor/types"
)

// set of operations for shaping routines

func (tensor *Tensor[T]) Flatten(out *Tensor[T]) *Tensor[T] {
	if len(tensor.shape) == 1 {
		return tensor
	}
	outTensor := PrepareOutTensor(out, tensor.shape)
	outTensor.shape = types.Shape{types.Dim(len(tensor.data))}
	outTensor.strides = []int{1}
	return outTensor
}

func (tensor *Tensor[T]) Squeeze(out *Tensor[T]) *Tensor[T] {
	if IsScalarLike(tensor.shape) {
		return tensor
	}
	outTensor := PrepareOutTensor(out, tensor.shape)
	outTensor.shape = squeeze_shape(tensor.shape)
	outTensor.strides = getStrides(outTensor.shape)
	outTensor.dim_order = initDimOrder(outTensor.shape)
	return outTensor
}

func (tensor *Tensor[T]) Reshape(newShape ...types.Dim) *Tensor[T] {
	var new_shape_prod types.Dim = 1
	for _, dim := range newShape {
		new_shape_prod *= dim
	}
	if new_shape_prod == 0 {
		panic("Shape cannot have 0 dim size.")
	}
	if len(tensor.data) != int(new_shape_prod) {
		panic(fmt.Sprintf("Cannot reshape tensor with size %v to shape %v", len(tensor.data), newShape))
	}
	tensor.shape = newShape
	tensor.strides = getStrides(newShape)
	tensor.dim_order = initDimOrder(newShape)
	return tensor
}

func (tensor *Tensor[T]) Transpose(axes ...uint) *Tensor[T] {
	n_dims := len(tensor.shape)
	if n_dims == 1 {
		return tensor
	}
	if len(axes) == 0 {
		axes = make([]uint, n_dims)
		for i := range axes {
			axes[i] = uint(len(axes) - i - 1)
		}
	}

	outTensor := CreateTensor(tensor.data, tensor.shape)

	for i, axis := range axes {
		outTensor.shape[i] = tensor.shape[axis]
		outTensor.strides[i] = tensor.strides[axis]
		outTensor.dim_order[i] = tensor.dim_order[axis]
	}
	return outTensor
}

// stacks tensors together. All tensors should have the same shape
func Unite[T types.TensorType](tensors ...*Tensor[T]) *Tensor[T] {
	if len(tensors) < 2 {
		panic("At least 2 tensors is required.")
	}

	// validate shapes
	var prev_shape types.Shape
	for _, tensor := range tensors {
		if prev_shape == nil {
			prev_shape = tensor.shape
			continue
		}
		if !Equal_1D_slices(tensor.shape, prev_shape) {
			panic("All shapes must be equal.")
		}
	}

	united_shape := make(types.Shape, len(tensors)+1)
	united_shape[0] = types.Dim(len(tensors))
	copy(united_shape[1:], tensors[0].shape)

	united := CreateEmptyTensor[T](united_shape...)
	for i := 0; i < len(tensors); i++ {
		tensor := tensors[i]
		size := len(tensor.data)
		copy(united.data[size*i:size*(i+1)], tensor.data)
	}
	return united
}
