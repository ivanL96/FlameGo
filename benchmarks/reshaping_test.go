package main

import (
	"gograd/tensor"
	"testing"
)

// go test -bench . ./benchmarks -benchmem -v -count=5
// go test -benchmem -run=^$ -bench ^BenchmarkIndex$ gograd/benchmarks
// 1879749	       618.7 ns/op	     178 B/op	       7 allocs/op -- as get()
// 1429494	       951.3 ns/op	     578 B/op	       6 allocs/op -- subdata
func BenchmarkIndex(b *testing.B) {
	a := tensor.Range[int32](100*100*100).Reshape(100, 100, 100)
	for i := 0; i < b.N; i++ {
		a.Index(88, 88)
	}
}

// 1962690	       692.9 ns/op	     180 B/op	       7 allocs/op -- as get()
// 661308	      2020 ns/op	    1020 B/op	       8 allocs/op
func BenchmarkTranposeIndex(b *testing.B) {
	a := tensor.Range[int32](100*100*100).Reshape(100, 100, 100).Transpose()
	for i := 0; i < b.N; i++ {
		a.Index(88, 88)
	}
}

func BenchmarkTranspose(b *testing.B) {
	a := tensor.Range[int32](1000).Reshape(10, 10, 10)
	for i := 0; i < b.N; i++ {
		a.Transpose()
	}
}

func BenchmarkBroadcast(b *testing.B) {
	for i := 0; i < b.N; i++ {
		a := tensor.InitEmptyTensor[int32](3, 2)
		a.Broadcast(3, 1, 1)
	}
}

func BenchmarkToString(b *testing.B) {
	a := tensor.Range[int32](1000) //.Reshape(10, 10, 10)
	for i := 0; i < b.N; i++ {
		a.ToString()
	}
}
