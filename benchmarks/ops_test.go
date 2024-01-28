package main

import (
	"fmt"
	"gograd/tensor"
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
// BenchmarkBigAdd-8			26	  		42.891.900 ns/op		4.314.271 B/op	       7 allocs/op
// Get_fast/range data
// BenchmarkBigAdd-8  			44     		23.705.093 ns/op   		4.188.202 B/op          7 allocs/op
// out Tensor prepared/Get_fast/range data
// BenchmarkBigAdd-8 			50     		23.525.360 ns/op	 	  160.285 B/op          1 allocs/op
// AVX
// BenchmarkBigAdd-8            456          2.198.060 ns/op           26.358 B/op          0 allocs/op
// BenchmarkBigAdd-8            328          3.064.911 ns/op           36.644 B/op          0 allocs/op
// BenchmarkBigAdd-8            489          2.165.072 ns/op           24.579 B/op          0 allocs/op

// same values tests
// Fill() unrolled 4
// 654    1.885.201 ns/op    4.018.343 B/op          6 allocs/op
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

func BenchmarkBigMulToConst(b *testing.B) {
	scalar := tensor.Scalar[float32](1000)
	a1 := tensor.Range[float32](1000000).Reshape(1000, 1000).Div(scalar, nil)
	a2 := scalar
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
	rng := tensor.NewRNG(0)
	a1 := rng.RandomFloat32(100, 100)
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
