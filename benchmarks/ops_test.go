package main

import (
	"flamego/tensor"
	types "flamego/tensor/types"
	"fmt"
	"testing"
)

func BenchmarkAdd(b *testing.B) {
	a1 := tensor.CreateEmptyTensor[int32](100, 100).Fill(100)
	a2 := tensor.CreateEmptyTensor[int32](100, 100).Fill(99)
	for i := 0; i < b.N; i++ {
		a1.Add(a2, nil)
	}
}

// init/range data
// 26	  42.891.900 ns/op	 4.314.271 B/op	       7 allocs/op
// Get_fast/range data
// 44     23.705.093 ns/op   4.188.202 B/op          7 allocs/op
// out Tensor prepared/Get_fast/range data
// 50     23.525.360 ns/op   160.285 B/op          1 allocs/op
// same values tests
// Fill() unrolled 4
// 654    1.885.201 ns/op    4.018.343 B/op          6 allocs/op
// Fill() SAME VALUE FLAG
// 445	   2.637.615 ns/op	 4.024.107 B/op	       6 allocs/op
func BenchmarkBigAdd(b *testing.B) {
	a1 := tensor.Range[int32](1000000).Reshape(1000, 1000) //.Fill(1)
	a2 := tensor.Range[int32](1000000).Reshape(1000, 1000) //.Fill(1)
	out := tensor.CreateEmptyTensor[int32](1000, 1000)
	for i := 0; i < b.N; i++ {
		a1.Add(a2, out)
	}
	fmt.Println(out.Get(999, 999))
}

func BenchmarkMul(b *testing.B) {
	a1 := tensor.CreateEmptyTensor[int32](100, 100).Fill(100)
	a2 := tensor.CreateEmptyTensor[int32](100, 100).Fill(99)
	for i := 0; i < b.N; i++ {
		a1.Mul(a2, nil)
	}
}

func BenchmarkMulScalar(b *testing.B) {
	a1 := tensor.CreateEmptyTensor[int32](1).Fill(100)
	a2 := tensor.CreateEmptyTensor[int32](1).Fill(99)
	for i := 0; i < b.N; i++ {
		a1.Mul(a2, nil)
	}
}

func BenchmarkSigmoid(b *testing.B) {
	a1 := tensor.RandomFloat32Tensor(types.Shape{100, 100}, 0)
	for i := 0; i < b.N; i++ {
		a1.Sigmoid(nil)
	}
}

// type X struct {
// 	data []int
// }

// func (x *X) Data() []int {
// 	return x.data()
// }

// func BenchmarkProp(b *testing.B) {
// 	x := X{data: []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}}
// 	c := make([]int, 10)
// 	for i := 0; i < b.N; i++ {
// 		c = x.data()
// 	}
// 	x.data() = c
// }

// func BenchmarkPropFunc(b *testing.B) {
// 	x := X{data: []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}}
// 	c := make([]int, 10)
// 	c = x.Data()
// 	for i := 0; i < b.N; i++ {
// 		c = x.Data()
// 	}
// 	x.data() = c
// }
