package tensor

import (
	"reflect"

	"golang.org/x/exp/constraints"
)

type Any interface{}

type Float interface {
	float32 | float64
}

type Number interface {
	constraints.Float | constraints.Integer
}

type Dim uint
type Shape []Dim

type Tensor[T Number] struct {
	data  []T
	shape Shape
	dtype reflect.Type
}

func (tensor *Tensor[T]) Shape() Shape {
	return tensor.shape
}

func (tensor *Tensor[T]) Data() []T {
	return tensor.data
}

func (tensor *Tensor[T]) DType() reflect.Type {
	return tensor.dtype
}
