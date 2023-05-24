package tensor

import (
	types "gograd/tensor/types"
	"reflect"
)

type Tensor[T types.TensorType] struct {
	dtype     reflect.Type
	data      []T
	shape     types.Shape
	strides   []int
	dim_order []int
}

func (tensor *Tensor[T]) Shape() types.Shape {
	return tensor.shape
}

func (tensor *Tensor[T]) Strides() []int {
	return tensor.strides
}

func (tensor *Tensor[T]) Order() []int {
	return tensor.dim_order
}

func (tensor *Tensor[T]) Data() []T {
	return tensor.data
}

func (tensor *Tensor[T]) DType() reflect.Type {
	return tensor.dtype
}
