package main

import (
	"gograd/tensor"
	"gograd/tensor/types"
	"testing"
)

// go test -bench . ./benchmarks -benchmem -v -count=5
// go test -benchmem -run=^$ -bench ^BenchmarkIndex$ gograd/benchmarks
// 1879749	       618.7 ns/op	     178 B/op	       7 allocs/op -- as get()
// 1429494	       951.3 ns/op	     578 B/op	       6 allocs/op -- subdata
// 1444742	       856.9 ns/op	     594 B/op	       6 allocs/op -- fill
func BenchmarkIndex(b *testing.B) {
	a := tensor.Range[int32](100*100*100).Reshape(100, 100, 100)
	for i := 0; i < b.N; i++ {
		a.Index(88, 88)
	}
}

// shape: 1000, 100, 1000
// index: 888, 88
// BenchmarkTransposeIndex-8          54966             20922 ns/op           22890 B/op          7 allocs/op
// BenchmarkTransposeIndex-8          73450             21780 ns/op           19227 B/op          7 allocs/op
// index: 888, 88, 900
// BenchmarkTransposeIndex-8        1567056               784.9 ns/op           654 B/op          6 allocs/op
// BenchmarkTransposeIndex-8        1685970               729.4 ns/op           618 B/op          6 allocs/op
func BenchmarkTransposeIndex(b *testing.B) {
	a := tensor.Range[int32](1000*100*1000).Reshape(1000, 100, 1000).T()
	for i := 0; i < b.N; i++ {
		a.Index(888, 88)
	}
}

func BenchmarkTranspose(b *testing.B) {
	a := tensor.Range[int32](1000).Reshape(10, 10, 10)
	for i := 0; i < b.N; i++ {
		a.T()
	}
}

// 3,2,1 => 3,4,1,1,3
// BenchmarkBroadcast-8      257005              4052 ns/op            1440 B/op         19 allocs/op
// BenchmarkBroadcast-8      405019              4463 ns/op            1440 B/op         19 allocs/op
// BenchmarkBroadcast-8      498444              4505 ns/op            1440 B/op         19 allocs/op
func BenchmarkBroadcast(b *testing.B) {
	for i := 0; i < b.N; i++ {
		a := tensor.Range[int32](6).Reshape(3, 2, 1)
		a.Broadcast(3, 4, 1, 1, 3)
	}
}

func BenchmarkToString(b *testing.B) {
	a := tensor.Range[int32](1000) //.Reshape(10, 10, 10)
	for i := 0; i < b.N; i++ {
		a.ToString()
	}
	a.MustAssert()
}

// goos: windows
// goarch: amd64
// pkg: gograd/benchmarks
// cpu: 11th Gen Intel(R) Core(TM) i5-11400H @ 2.70GHz
// 1000x1000
// 1 thread
// BenchmarkAsContiguous-12             352           3.305.912 ns/op         6.020.435 B/op          9 allocs/op
// BenchmarkAsContiguous-12             367           3.312.905 ns/op         6.025.425 B/op          9 allocs/op
// BenchmarkAsContiguous-12             363           3.293.073 ns/op         6.025.606 B/op          9 allocs/op
// optimized for 2D matrix
// BenchmarkAsContiguous-12            1024           1.106.555 ns/op         6.064.986 B/op       1009 allocs/op
// BenchmarkAsContiguous-12            1010           1.103.096 ns/op         6.065.024 B/op       1009 allocs/op
// BenchmarkAsContiguous-12             996           1.101.434 ns/op         6.065.081 B/op       1009 allocs/op
// loop unroll
// BenchmarkAsContiguous-12            1225             912.125 ns/op         6.074.341 B/op       1010 allocs/op
// BenchmarkAsContiguous-12            1090             925.909 ns/op         6.072.793 B/op       1009 allocs/op
// BenchmarkAsContiguous-12            1287             904.302 ns/op         6.073.798 B/op       1009 allocs/op
// goroutine per cpu
// BenchmarkAsContiguous-12            1612             688.423 ns/op         6.012.334 B/op         21 allocs/op
// BenchmarkAsContiguous-12            1648             710.814 ns/op         6.012.286 B/op         21 allocs/op
// BenchmarkAsContiguous-12            1590             696.024 ns/op         6.012.369 B/op         21 allocs/op
func BenchmarkAsContiguous(b *testing.B) {
	var side types.Dim = 1000
	a := tensor.Range[int32](int(side*side)).Reshape(side, side)
	for i := 0; i < b.N; i++ {
		a = a.T()
		a.AsContiguous()
	}
	a.MustAssert()
}

// goarch: amd64
// pkg: gograd/benchmarks
// cpu: 11th Gen Intel(R) Core(TM) i5-11400H @ 2.70GHz
// default
// BenchmarkTrC2D-12          22525             53.875 ns/op           53131 B/op        207 allocs/op
// BenchmarkTrC2D-12          22410             54.940 ns/op           53131 B/op        207 allocs/op
// BenchmarkTrC2D-12          21188             55.510 ns/op           53131 B/op        207 allocs/op
// unroll
// BenchmarkTrC2D-12          28478             41.330 ns/op           53130 B/op        207 allocs/op
// BenchmarkTrC2D-12          28988             41.902 ns/op           53130 B/op        207 allocs/op
// BenchmarkTrC2D-12          29118             43.685 ns/op           53131 B/op        207 allocs/op
func BenchmarkTrC2D(b *testing.B) {
	a := tensor.Range[float32](10000).Reshape(100, 100)
	for i := 0; i < b.N; i++ {
		a.TrC2D()
	}
	a.MustAssert()
}
