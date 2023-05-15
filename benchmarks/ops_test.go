package main

import (
	"gograd/tensor"
	"testing"
)

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
