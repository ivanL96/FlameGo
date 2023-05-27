package ops

import (
	"gograd/tensor/types"
	"math"
)

// binary
func Add[T types.TensorType](a, b T) T {
	return a + b
}

func Sub[T types.TensorType](a, b T) T {
	return a - b
}
func Mul[T types.TensorType](a, b T) T {
	return a * b
}
func Div[T types.TensorType](a, b T) T {
	return a / b
}

// unary
func Neg[T types.TensorType](a T) T {
	return -a
}

func Sigmoid[T types.TensorType](a T) T {
	return T(1. / (1. + math.Pow(math.E, float64(-a))))
}
