package tensor

import (
	"fmt"
	"reflect"

	"golang.org/x/exp/constraints"
)

type Any interface{}

type Number interface {
	constraints.Float | constraints.Integer
}

type Dim uint
type Shape []Dim

type Tensor[T Number] struct {
	shape Shape
	ndim  Dim
	data  []T
	dtype reflect.Type
	len   uint
}

func (tensor *Tensor[T]) Shape() Shape {
	return tensor.shape
}

func (tensor *Tensor[T]) Data() []T {
	return tensor.data
}

func (tensor *Tensor[T]) ToString() string {
	str := fmt.Sprintf("Tensor(data=%v, shape=%v, ndim=%v, dtype=%v)", tensor.data, tensor.shape, tensor.ndim, tensor.dtype.String())
	return str
}
