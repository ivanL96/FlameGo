package internal

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
	return T(math.Pow(float64(a), float64(b)))
}

// unary
func NegAtomic[T types.TensorType](a T) T {
	return -a
}

func SigmoidAtomic[T types.TensorType](a T) T {
	return T(1. / (1. + math.Pow(math.E, float64(-a))))
}

func LnAtomic[T types.TensorType](a T) T {
	return T(math.Log(float64(a)))
}

func ReluAtomic[T types.TensorType](a T) T {
	if a > 0 {
		return a
	}
	return 0
}

func ExpAtomic[T types.TensorType](a T) T {
	return T(math.Exp(float64(a)))
}
