package noasm

import (
	"gograd/tensor/types"
	"math"
	"runtime"
	"sync"
)

// noasm vector operations for two-dim matrices

func identical[T types.TensorType](s1, s2 []T) bool {
	if len(s1) != len(s2) {
		return false
	}

	return len(s1) == 0 || &s1[0] == &s2[0]
}

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
	la := len(a)
	chunk_size := (la + numCPU - 1) / numCPU

	var wg sync.WaitGroup
	wg.Add(numCPU)

	for i := 0; i < numCPU; i++ {
		start := i * chunk_size
		end := (i + 1) * chunk_size
		if end > la {
			end = la
		}

		go func(start, end int) {
			defer wg.Done()
			f(start, end, a, b, out)
		}(start, end)
	}
	wg.Wait()
}

func AddMatx[T types.TensorType](a, b, out []T) {
	// identical(b, out)
	var add_chunk func(int, int, []T, []T, []T)
	if identical(a, out) {
		add_chunk = func(start, end int, a, b, out []T) {
			for i := start; i < end; i++ {
				out[i] += b[i]
			}
		}
	} else if identical(b, out) {
		add_chunk = func(start, end int, a, b, out []T) {
			for i := start; i < end; i++ {
				out[i] += a[i]
			}
		}
	} else {
		add_chunk = func(start, end int, a, b, out []T) {
			for i := start; i < end; i++ {
				out[i] = a[i] + b[i]
			}
		}
	}
	MatxParallel[T](add_chunk, a, b, makeOutMat(out, len(a)))
}

func SubMatx[T types.TensorType](a, b, out []T) {
	var sub_chunk func(int, int, []T, []T, []T)
	if identical(a, out) {
		sub_chunk = func(start, end int, a, b, out []T) {
			for i := start; i < end; i++ {
				out[i] -= b[i]
			}
		}
	} else if identical(b, out) {
		sub_chunk = func(start, end int, a, b, out []T) {
			for i := start; i < end; i++ {
				out[i] -= a[i]
			}
		}
	} else {
		sub_chunk = func(start, end int, a, b, out []T) {
			for i := start; i < end; i++ {
				out[i] = a[i] - b[i]
			}
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
	div_chunk := func(start, end int, a, b, out []T) {
		for i := start; i < end; i++ {
			out[i] = a[i] / b[i]
		}
	}
	MatxParallel[T](div_chunk, a, b, makeOutMat(out, len(a)))
}

func PowMatx[T types.TensorType](a, b, out []T) {
	pow_chunk := func(start, end int, a, b, out []T) {
		for i := start; i < end; i++ {
			out[i] = T(math.Pow(float64(a[i]), float64(b[i])))
		}
	}
	MatxParallel[T](pow_chunk, a, b, makeOutMat(out, len(a)))
}

func SigmoidMatx[T types.TensorType](a, out []T) {
	// x / (1 + abs(x))
	sigm_chunk := func(start, end int, a, dummy, out []T) {
		for i := start; i < end; i++ {
			// out[i] = a[i] / (1. + T(math.Abs(float64(a[i]))))
			out[i] = T(1. / (1. + math.Pow(math.E, float64(-a[i]))))
		}
	}
	var dummy []T
	MatxParallel[T](sigm_chunk, a, dummy, makeOutMat(out, len(a)))
}

func NegMatx[T types.TensorType](a, out []T) {
	neg_chunk := func(start, end int, a, dummy, out []T) {
		for i := start; i < end; i++ {
			out[i] = -a[i]
		}
	}
	var dummy []T
	MatxParallel[T](neg_chunk, a, dummy, makeOutMat(out, len(a)))
}
