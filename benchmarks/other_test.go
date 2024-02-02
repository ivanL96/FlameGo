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
		tensor.CreateEmptyTensor[float64](100)
	}
}

func BenchmarkIndexFast(b *testing.B) {
	for i := 0; i < b.N; i++ {
		a := tensor.Range[int32](1000).Reshape(1, 1, 10, 10, 10, 1, 1, 1)
		// a.Get_fast(0, 0, 6, 6, 6, 0, 0, 0)
		a.Get(0, 0, 6, 6, 6, 0, 0, 0)
	}
}

// goos: windows
// goarch: amd64
// pkg: gograd/benchmarks
// cpu: 11th Gen Intel(R) Core(TM) i5-11400H @ 2.70GHz
// single thread
// BenchmarkFill-12            4172            251.399 ns/op             960 B/op          0 allocs/op
// BenchmarkFill-12            4581            252.454 ns/op             874 B/op          0 allocs/op
// BenchmarkFill-12            4402            250.280 ns/op             910 B/op          0 allocs/op
// goroutines
// BenchmarkFill-12           13357             88.549 ns/op            1468 B/op         25 allocs/op
// BenchmarkFill-12           13587             89.687 ns/op            1462 B/op         25 allocs/op
// BenchmarkFill-12           13484             88.113 ns/op            1465 B/op         25 allocs/op
// goroutines + unroll 8
// BenchmarkFill-12           52269             20.747 ns/op            1245 B/op         25 allocs/op
// BenchmarkFill-12           59397             20.690 ns/op            1235 B/op         25 allocs/op
// BenchmarkFill-12           55560             20.324 ns/op            1240 B/op         25 allocs/op
func BenchmarkFill(b *testing.B) {
	a := tensor.CreateEmptyTensor[float32](1003, 1001)
	for i := 0; i < b.N; i++ {
		a.Fill(float32(i))
	}
	a.MustAssert()
}

// goos: windows
// goarch: amd64
// pkg: gograd/benchmarks
// cpu: 11th Gen Intel(R) Core(TM) i5-11400H @ 2.70GHz
// BenchmarkAsType
// BenchmarkAsType-12          1395            735.043 ns/op         4008906 B/op          5 allocs/op
// BenchmarkAsType-12          1539            776.848 ns/op         4008639 B/op          5 allocs/op
// BenchmarkAsType-12          1610            753.102 ns/op         4008522 B/op          5 allocs/op
// unroll
// BenchmarkAsType-12          2344            483.109 ns/op         4008920 B/op         30 allocs/op
// BenchmarkAsType-12          2524            469.145 ns/op         4008800 B/op         30 allocs/op
// BenchmarkAsType-12          2090            485.649 ns/op         4009133 B/op         30 allocs/op
func BenchmarkAsType(b *testing.B) {
	a := tensor.CreateEmptyTensor[float32](1003, 1001)
	for i := 0; i < b.N; i++ {
		tensor.AsType[float32, int32](a)
	}
	a.MustAssert()
}
