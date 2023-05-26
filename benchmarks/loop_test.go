package main

import (
	"gograd/tensor"
	"gograd/tensor/types"
	"testing"
	"unsafe"
)

func sum_float64_go(buf []float64) float64 {
	acc := float64(0)
	for i := range buf {
		acc += buf[i]
	}
	return acc
}

func sum_float64_go_unroll4[T types.TensorType](buf []T) T {
	var (
		acc0, acc1, acc2, acc3 T
	)

	for i := 0; i < len(buf); i += 4 {
		bb := (*[4]T)(unsafe.Pointer(&buf[i]))
		acc0 += bb[0]
		acc1 += bb[1]
		acc2 += bb[2]
		acc3 += bb[3]
	}
	return acc0 + acc1 + acc2 + acc3
}

// BenchmarkLoop-8            15199             89881 ns/op              52 B/op          0 allocs/op
// BenchmarkLoop-8            13575             91785 ns/op              59 B/op          0 allocs/op
// BenchmarkLoop-8            15489             79947 ns/op              51 B/op          0 allocs/op
func BenchmarkLoop(b *testing.B) {
	data := tensor.Range[float64](100 * 100 * 10).Data()
	for i := 0; i < b.N; i++ {
		sum_float64_go(data)
	}
}

// 4,5x speed up
// BenchmarkLoopUnroll-8              55106             19717 ns/op              14 B/op          0 allocs/op
// BenchmarkLoopUnroll-8              55365             23707 ns/op              14 B/op          0 allocs/op
// BenchmarkLoopUnroll-8              51512             22441 ns/op              15 B/op          0 allocs/op
func BenchmarkLoopUnroll(b *testing.B) {
	data := tensor.Range[float64](100 * 100 * 10).Data()
	for i := 0; i < b.N; i++ {
		sum_float64_go_unroll4(data)
	}
}
