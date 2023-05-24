package main

import (
	"gograd/tensor"
	"testing"
)

// go test -bench . ./benchmarks -benchmem -v -count=5
func BenchmarkIndex(b *testing.B) {
	a := tensor.Range[int32](8*5*10).Reshape(8, 5, 10)
	for i := 0; i < b.N; i++ {
		a.Index(1, 2)
	}
}

func BenchmarkTranposeIndex(b *testing.B) {
	a := tensor.Range[int32](8*5*10).Reshape(8, 5, 10).Transpose()
	for i := 0; i < b.N; i++ {
		a.Index(1, 2)
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
