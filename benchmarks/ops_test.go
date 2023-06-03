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

// scalar op
// BenchmarkBigMul-8             32          34.973.966 ns/op          626015 B/op          2 allocs/op
// BenchmarkBigMul-8             31          40.353.242 ns/op          646199 B/op          2 allocs/op
// BenchmarkBigMul-8             39          37.115.072 ns/op          513648 B/op          2 allocs/op
// mul noasm
// BenchmarkBigMul-8            434           2.422.563 ns/op           46155 B/op          0 allocs/op
// BenchmarkBigMul-8            469           2.410.727 ns/op           42711 B/op          0 allocs/op
// BenchmarkBigMul-8            480           2.434.254 ns/op           41732 B/op          0 allocs/op
// avx2
// BenchmarkBigMul-8           1047           1.095.734 ns/op           19132 B/op          0 allocs/op
// BenchmarkBigMul-8           1172           1.150.072 ns/op           17090 B/op          0 allocs/op
// BenchmarkBigMul-8           1178           1.024.620 ns/op           17003 B/op          0 allocs/op
// shape 1000x1000
func BenchmarkBigMul(b *testing.B) {
	scalar := tensor.Scalar[float32](1000)
	a1 := tensor.Range[float32](1000000).Reshape(1000, 1000).Div(scalar, nil)
	a2 := tensor.Range[float32](1000000).Reshape(1000, 1000).Div(scalar, nil)
	out := tensor.CreateEmptyTensor[float32](1000, 1000)
	for i := 0; i < b.N; i++ {
		a1.Mul(a2, out)
	}
	fmt.Println(out.Get(999, 999))
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
