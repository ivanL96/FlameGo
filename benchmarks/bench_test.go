package main

import (
	"gograd/tensor"
	"testing"
)

// go test -bench . ./benchmarks -benchmem -v -count=5
func BenchmarkIndex(b *testing.B) {
	a := tensor.Range[int32](1000).Reshape(10, 10, 10)
	for i := 0; i < b.N; i++ {
		a.Index(1, 2, 3)
	}
}

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

func BenchmarkSigmoid(b *testing.B) {
	a1 := tensor.RandomFloat32Tensor(tensor.Shape{100, 100}, 0)
	// a1 := tensor.InitEmptyTensor[float32](100, 100).Fill(10)
	for i := 0; i < b.N; i++ {
		a1.Sigmoid()
	}
}

func BenchmarkToString(b *testing.B) {
	a := tensor.Range[int32](1000) //.Reshape(10, 10, 10)
	for i := 0; i < b.N; i++ {
		a.ToString()
	}
}
