package ops

import (
	"gograd/tensor/types"
	"math"
)

// binary
func Add[T types.TensorType](a, b T) T {
	return a + b
}

func SubAtomic[T types.TensorType](a, b T) T {
	return a - b
}
func MulAtomic[T types.TensorType](a, b T) T {
	return a * b
}
func DivAtomic[T types.TensorType](a, b T) T {
	return a / b
}

// unary
func NegAtomic[T types.TensorType](a T) T {
	return -a
}

func SigmoidAtomic[T types.TensorType](a T) T {
	return T(1. / (1. + math.Pow(math.E, float64(-a))))
}
