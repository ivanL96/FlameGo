package tensor

import (
	types "gograd/tensor/types"
	"reflect"
)

// fieldalignment -fix gograd/tensor
type Tensor[T types.TensorType] struct {
	data_buff []T
	shape     types.Shape
	strides   []int
	dim_order []uint16
}

type TensorList[T types.TensorType] []*Tensor[T]

func (tensor *Tensor[T]) Shape() types.Shape {
	return tensor.shape
}

func (tensor *Tensor[T]) Strides() []int {
	return tensor.strides
}

func (tensor *Tensor[T]) Order() []uint16 {
	return tensor.dim_order
}

func (tensor *Tensor[T]) Data() []T {
	return tensor.data_buff
}

func (tensor *Tensor[T]) data() []T {
	return tensor.data_buff
}

func (tensor *Tensor[T]) DType() reflect.Type {
	return getTypeArray(tensor.data())
}
