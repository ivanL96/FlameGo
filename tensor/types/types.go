package types

import "golang.org/x/exp/constraints"

type Any interface{}

type Float interface {
	float32 | float64
}

type TensorType interface {
	constraints.Float | constraints.Integer
}

type Dim uint32
type Shape []Dim
