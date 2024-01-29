package main

import (
	"fmt"
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

// goos: windows
// goarch: amd64
// pkg: gograd/benchmarks
// cpu: 11th Gen Intel(R) Core(TM) i5-11400H @ 2.70GHz
// BenchmarkLoop
// BenchmarkLoop-12           50310             23327 ns/op              15 B/op          0 allocs/op
// BenchmarkLoop-12           50791             23276 ns/op              15 B/op          0 allocs/op
// BenchmarkLoop-12           49707             23137 ns/op              16 B/op          0 allocs/op
// BenchmarkLoop-12           51193             23085 ns/op              15 B/op          0 allocs/op
func BenchmarkLoop(b *testing.B) {
	data := tensor.Range[float64](100 * 100 * 10).Data()
	var res float64
	for i := 0; i < b.N; i++ {
		sum_float64_go(data)
	}
	fmt.Println(res) //4.99995e+09
}

// 4,5x speed up
// BenchmarkLoopUnroll-12            191854              5902 ns/op               4 B/op          0 allocs/op
// BenchmarkLoopUnroll-12            196690              5824 ns/op               4 B/op          0 allocs/op
// BenchmarkLoopUnroll-12            195310              5818 ns/op               4 B/op          0 allocs/op
// BenchmarkLoopUnroll-12            196312              5793 ns/op               4 B/op          0 allocs/op
// BenchmarkLoopUnroll-12            194678              5779 ns/op               4 B/op          0 allocs/op
func BenchmarkLoopUnroll(b *testing.B) {
	data := tensor.Range[float64](100 * 100 * 10).Data()
	var res float64
	for i := 0; i < b.N; i++ {
		sum_float64_go_unroll4(data)
	}
	fmt.Println(res) //4.99995e+09
}

func BenchmarkTensorCreate(b *testing.B) {
	for i := 0; i < b.N; i++ {
		tensor.CreateEmptyTensor[float64](100, 100, 10)
	}
}

func BenchmarkIndexFast(b *testing.B) {
	for i := 0; i < b.N; i++ {
		a := tensor.Range[int32](50).Reshape(10, 5)
		// a.Get_fast(6, 1)
		a.Get(6, 1)
	}
}
