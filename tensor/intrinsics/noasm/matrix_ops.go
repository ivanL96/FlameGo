package noasm

import (
	"flamego/tensor/types"
)

// noasm vector operations for two-dim matrices

func makeOutMat[T types.TensorType](out []T, size int) []T {
	if out == nil {
		return make([]T, size)
	}
	return out
}

func AddMatx[T types.TensorType](a, b, out []T) {
	out_ := makeOutMat(out, len(a))
	for i, val := range a {
		out_[i] = val + b[i]
	}
}

func SubMatx[T types.TensorType](a, b, out []T) {
	out_ := makeOutMat(out, len(a))
	for i, val := range a {
		out_[i] = val - b[i]
	}
}

func Dot[T types.TensorType](a, b []T, c T) T {
	for i := 0; i < len(a); i++ {
		c += a[i] * b[i]
	}
	return c
}

func MulMatx[T types.TensorType](a, b, out []T) {
	out_ := makeOutMat(out, len(a))
	for i, val := range a {
		out_[i] = val * b[i]
	}
}

func DivMatx[T types.TensorType](a, b, out []T) {
	out_ := makeOutMat(out, len(a))
	for i, val := range a {
		out_[i] = val / b[i]
	}
}
