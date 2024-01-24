package ops

import (
	"gograd/tensor/types"
	"math"
)

// this is set of scalar operations used in generic tensor/ops.go

// binary
func AddAtomic[T types.TensorType](a, b T) T {
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

func PowAtomic[T types.TensorType](a, b T) T {
	return any(math.Pow(float64(a), float64(b))).(T)
}

// unary
func NegAtomic[T types.TensorType](a T) T {
	return -a
}

func SigmoidAtomic[T types.TensorType](a T) T {
	return T(1. / (1. + math.Pow(math.E, float64(-a))))
}
