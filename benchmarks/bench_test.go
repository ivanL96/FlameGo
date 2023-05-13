package main

import (
	"gograd/tensor"
	"testing"
)

// go test -bench . ./benchmarks -benchmem -v -count=5
func BenchmarkBroadcast(b *testing.B) {
	for i := 0; i < b.N; i++ {
		a := tensor.InitEmptyTensor[int32](3, 2)
		a.Broadcast(3, 1, 1)
	}
}

func BenchmarkAdd(b *testing.B) {
	a1 := tensor.InitEmptyTensor[int32](100, 100).Fill(100)
	a2 := tensor.InitEmptyTensor[int32](100, 100).Fill(99)
	for i := 0; i < b.N; i++ {
		a1.Add(a2)
	}
}

func BenchmarkMul(b *testing.B) {
	a1 := tensor.InitEmptyTensor[int32](100, 100).Fill(100)
	a2 := tensor.InitEmptyTensor[int32](100, 100).Fill(99)
	for i := 0; i < b.N; i++ {
		a1.Mul(a2)
	}
}

func BenchmarkMulScalar(b *testing.B) {
	a1 := tensor.InitEmptyTensor[int32](1).Fill(100)
	a2 := tensor.InitEmptyTensor[int32](1).Fill(99)
	for i := 0; i < b.N; i++ {
		a1.Mul(a2)
	}
}
