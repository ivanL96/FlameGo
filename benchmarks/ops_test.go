package main

import (
	"gograd/tensor"
	types "gograd/tensor/types"
	"testing"
)

func BenchmarkAdd(b *testing.B) {
	a1 := tensor.InitEmptyTensor[int32](100, 100).Fill(100)
	a2 := tensor.InitEmptyTensor[int32](100, 100).Fill(99)
	for i := 0; i < b.N; i++ {
		a1.Add(a2)
	}
}

func BenchmarkBigAdd(b *testing.B) {
	a1 := tensor.Range[int32](1000000).Reshape(1000, 1000)
	a2 := tensor.Range[int32](1000000).Reshape(1000, 1000)
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
	a1 := tensor.RandomFloat32Tensor(types.Shape{100, 100}, 0)
	for i := 0; i < b.N; i++ {
		a1.Sigmoid()
	}
}

// type X struct {
// 	data []int
// }

// func (x *X) Data() []int {
// 	return x.data
// }

// func BenchmarkProp(b *testing.B) {
// 	x := X{data: []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}}
// 	c := make([]int, 10)
// 	for i := 0; i < b.N; i++ {
// 		c = x.data
// 	}
// 	x.data = c
// }

// func BenchmarkPropFunc(b *testing.B) {
// 	x := X{data: []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}}
// 	c := make([]int, 10)
// 	c = x.Data()
// 	for i := 0; i < b.N; i++ {
// 		c = x.Data()
// 	}
// 	x.data = c
// }
