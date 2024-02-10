package internal

import (
	"gograd/tensor/types"
	"sync"
)

func ElementwiseNoSimdUnary[T types.TensorType](a, out []T, atomic func(T) T) {
	chunk := func(start, end int, a, b, out []T, mu *sync.Mutex) {
		for i := start; i < end; i++ {
			out[i] = atomic(a[i])
		}
	}
	parallel(chunk, a, []T{}, makeOutMat(out, len(a)))
}

func ElementwiseNoSimd[T types.TensorType](a, b, out []T, atomic func(T, T) T) {
	var chunk func(int, int, []T, []T, []T, *sync.Mutex)
	if len(a) == 1 {
		chunk = func(start, end int, a, b, out []T, mu *sync.Mutex) {
			for i := start; i < end; i++ {
				out[i] = atomic(a[0], b[i])
			}
		}
	} else if len(b) == 1 {
		chunk = func(start, end int, a, b, out []T, mu *sync.Mutex) {
			for i := start; i < end; i++ {
				out[i] = atomic(a[i], b[0])
			}
		}
	} else {
		chunk = func(start, end int, a, b, out []T, mu *sync.Mutex) {
			for i := start; i < end; i++ {
				out[i] = atomic(a[i], b[i])
			}
		}
	}
	parallel(chunk, a, b, makeOutMat(out, len(a)))
}
