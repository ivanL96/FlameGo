package tensor

import (
	"fmt"

	"golang.org/x/exp/constraints"
)

type Any interface{}

type Number interface {
	constraints.Float | constraints.Integer
}

type Shape []uint

type Tensor[T Number] struct {
	shape Shape
	ndim  uint
	data  []T
	len   uint
}

func (tensor *Tensor[T]) Shape() Shape {
	return tensor.shape
}

func (tensor *Tensor[T]) ToString() string {
	str := fmt.Sprintf("Tensor(data=%x, shape=%x, ndim=%x)", tensor.data, tensor.shape, tensor.ndim)
	return str
}
