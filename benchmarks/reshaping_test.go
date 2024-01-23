package main

import (
	"flamego/tensor"
	"testing"
)

// go test -bench . ./benchmarks -benchmem -v -count=5
// go test -benchmem -run=^$ -bench ^BenchmarkIndex$ flamego/benchmarks
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
	a := tensor.Range[int32](1000*100*1000).Reshape(1000, 100, 1000).Transpose()
	for i := 0; i < b.N; i++ {
		a.Index(888, 88)
	}
}

func BenchmarkTranspose(b *testing.B) {
	a := tensor.Range[int32](1000).Reshape(10, 10, 10)
	for i := 0; i < b.N; i++ {
		a.Transpose()
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
}

func BenchmarkAsContinuous(b *testing.B) {
	a := tensor.Range[int32](1000*1000).Reshape(1000, 1000)
	for i := 0; i < b.N; i++ {
		a = a.Transpose()
		a.AsContinuous(nil)
	}
}
