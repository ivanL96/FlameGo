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
	flags     uint8
}

const (
	SameValuesFlag uint8 = 1 << iota
)

func (tensor *Tensor[T]) setFlag(flag uint8) {
	tensor.flags |= flag
}
func (tensor *Tensor[T]) clearFlag(flag uint8) {
	tensor.flags &^= flag
}
func (tensor *Tensor[T]) toggleFlag(flag uint8) {
	tensor.flags ^= flag
}
func (tensor *Tensor[T]) hasFlag(flag uint8) bool {
	return tensor.flags&flag != 0
}
func (tensor *Tensor[T]) ResetFlags() {
	tensor.flags = 0
}

// func (tensor *Tensor[T]) Flags() {
// 	for i, flag := range []uint8{SameValuesFlag} {
// 		tensor.hasFlag(flag)
// 	}
// }

type TensorList[T types.TensorType] []*Tensor[T]

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
