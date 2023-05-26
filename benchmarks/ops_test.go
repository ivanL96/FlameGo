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

// 26	  42.891.900 ns/op	 4314271 B/op	       7 allocs/op
// SAME VALUE FLAG
// 445	   2.637.615 ns/op	 4024107 B/op	       6 allocs/op
// fast get
// 44     23.705.093 ns/op         4188202 B/op          7 allocs/op
func BenchmarkBigAdd(b *testing.B) {
	a1 := tensor.Range[int32](1000000).Reshape(1000, 1000).Fill(1)
	a2 := tensor.Range[int32](1000000).Reshape(1000, 1000).Fill(1)
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

// BenchmarkMatMul-8   	   26259	     43096 ns/op	     616 B/op	       7 allocs/op
// BenchmarkMatMul-8   	       1	69653349600 ns/op	12020000 B/op	      31 allocs/op
// go matmul 69.653.349.600
// numpy matmul 4.910.045.600
func BenchmarkMatMul(b *testing.B) {
	a1 := tensor.Range[int32](1000*1000).Reshape(1000, 1000)
	b1 := tensor.Range[int32](1000*1000).Reshape(1000, 1000)
	for i := 0; i < b.N; i++ {
		a1.MatMul(b1)
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
