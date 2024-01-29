package noasm

import (
	"gograd/tensor/types"
	"math"
	"runtime"
	"sync"
)

// noasm vector operations for two-dim matrices

func makeOutMat[T types.TensorType](out []T, size int) []T {
	if out == nil {
		return make([]T, size)
	}
	return out
}

var numCPU int = runtime.NumCPU()

func MatxParallel[T types.TensorType](
	f func(int, int, []T, []T, []T),
	a, b, out []T,
) {
	chunk_size := (len(a) + numCPU - 1) / numCPU

	var wg sync.WaitGroup
	wg.Add(numCPU)

	for i := 0; i < numCPU; i++ {
		start := i * chunk_size
		end := (i + 1) * chunk_size
		if end > len(a) {
			end = len(a)
		}

		go func(start, end int) {
			defer wg.Done()
			f(start, end, a, b, out)
		}(start, end)
	}
	wg.Wait()
}

func AddMatx[T types.TensorType](a, b, out []T) {
	add_chunk := func(start, end int, a, b, out []T) {
		for i := start; i < end; i++ {
			out[i] = a[i] + b[i]
		}
	}
	MatxParallel[T](add_chunk, a, b, makeOutMat(out, len(a)))
}

func SubMatx[T types.TensorType](a, b, out []T) {
	sub_chunk := func(start, end int, a, b, out []T) {
		for i := start; i < end; i++ {
			out[i] = a[i] - b[i]
		}
	}
	MatxParallel[T](sub_chunk, a, b, makeOutMat(out, len(a)))
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

func MulMatxToConst[T types.TensorType](a []T, b T, out []T) {
	out_ := makeOutMat(out, len(a))
	for i, val := range a {
		out_[i] = val * b
	}
}

func DivMatx[T types.TensorType](a, b, out []T) {
	out_ := makeOutMat(out, len(a))
	for i, val := range a {
		out_[i] = val / b[i]
	}
}

func PowMatx[T types.TensorType](a, b, out []T) {
	af := any(a).([]float64)
	bf := any(b).([]float64)
	outf := any(makeOutMat(out, len(a))).([]float64)
	for i, val := range af {
		outf[i] = math.Pow(val, bf[i])
	}
}

func SigmoidMatx[T types.TensorType](a, out []T) {
	sigm_chunk := func(start, end int, a, dummy, out []T) {
		for i := start; i < end; i++ {
			out[i] = T(1. / (1. + math.Pow(math.E, float64(-a[i]))))
		}
	}
	var dummy []T
	MatxParallel[T](sigm_chunk, a, dummy, makeOutMat(out, len(a)))
}
