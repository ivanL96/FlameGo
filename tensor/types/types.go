package types

import "golang.org/x/exp/constraints"

type Any interface{}

type Float interface {
	float32 | float64
}

type TensorType interface {
	constraints.Float | constraints.Integer | ~byte
}

type Dim uint32
type Shape []Dim

type ITensor[T TensorType] interface {
	Shape() Shape
	Strides() []int
	Data() []T
	Order() []uint16

	// indexing ops
	Get(indices ...int) T
}
