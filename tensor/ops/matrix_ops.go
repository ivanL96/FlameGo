package ops

import (
	"gograd/tensor/types"
)

func makeOutMat[T types.TensorType](out []T, size int) []T {
	if out == nil {
		return make([]T, size)
	}
	return out
}

// addition for two-dim matrices
func AddMatx[T types.TensorType](a, b, out []T) []T {
	out_ := makeOutMat(out, len(a))
	for i, val := range a {
		out_[i] = val + b[i]
	}
	return out_
}

func MulMatx[T types.TensorType](a, b, out []T) []T {
	out_ := makeOutMat(out, len(a))
	for i, val := range a {
		out_[i] = val * b[i]
	}
	return out_
}

func SubMatx[T types.TensorType](a, b, out []T) []T {
	out_ := makeOutMat(out, len(a))
	for i, val := range a {
		out_[i] = val - b[i]
	}
	return out_
}

// func MulAggregareMatx[T types.TensorType](a, b, c []T) []T {

// }
