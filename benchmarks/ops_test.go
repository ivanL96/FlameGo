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

// go test -benchmem -run=^$ -bench ^BenchmarkBigAdd$ gograd/benchmarks -benchmem -v -count=5
// goos: windows
// goarch: amd64
// pkg: gograd/benchmarks
// cpu: 11th Gen Intel(R) Core(TM) i5-11400H @ 2.70GHz
// scalar
// BenchmarkBigAdd-12           199           5888470 ns/op           60408 B/op          1 allocs/op
// vectorized
// BenchmarkBigAdd-12          1946            523187 ns/op            6175 B/op          0 allocs/op
// BenchmarkBigAdd-12          1957            529357 ns/op            6141 B/op          0 allocs/op
// vec + goroutines
// BenchmarkBigAdd-12          7246            158306 ns/op            3212 B/op         25 allocs/op
// BenchmarkBigAdd-12          6296            199704 ns/op            3460 B/op         25 allocs/op
func BenchmarkBigAdd(b *testing.B) {
	a1 := tensor.Range[int32](1000000).Reshape(1000, 1000) //.Fill(1)
	a2 := tensor.Range[int32](1000000).Reshape(1000, 1000) //.Fill(1)
	out := tensor.CreateEmptyTensor[int32](1000, 1000)
	for i := 0; i < b.N; i++ {
		a1.Add(a2, out)
	}
	// fmt.Println(out.Get(999, 999))
}

// goos: windows
// goarch: amd64
// pkg: gograd/benchmarks
// cpu: 11th Gen Intel(R) Core(TM) i5-11400H @ 2.70GHz
// scalar
// BenchmarkBigMul-12           186           6120142 ns/op          107705 B/op          1 allocs/op
// BenchmarkBigMul-12           198           5973748 ns/op          101178 B/op          1 allocs/op
// BenchmarkBigMul-12           198           6092643 ns/op          101178 B/op          1 allocs/op
// BenchmarkBigMul-12           198           5970509 ns/op          101179 B/op          1 allocs/op
// BenchmarkBigMul-12           196           6123761 ns/op          102210 B/op          1 allocs/op
// vec + goroutines
// BenchmarkBigMul-12          1845            548058 ns/op           10856 B/op          0 allocs/op
// BenchmarkBigMul-12          2286            514924 ns/op            8762 B/op          0 allocs/op
// BenchmarkBigMul-12          2344            519922 ns/op            8545 B/op          0 allocs/op
// BenchmarkBigMul-12          2221            638278 ns/op            9019 B/op          0 allocs/op
// BenchmarkBigMul-12          1401            796004 ns/op           14297 B/op          0 allocs/op
// AVX
// BenchmarkBigMul-12          3416            377943 ns/op            5863 B/op          0 allocs/op
// BenchmarkBigMul-12          3577            343849 ns/op            5599 B/op          0 allocs/op
// BenchmarkBigMul-12          3550            350165 ns/op            5642 B/op          0 allocs/op
// BenchmarkBigMul-12          3417            343890 ns/op            5861 B/op          0 allocs/op
// BenchmarkBigMul-12          3476            394101 ns/op            5762 B/op          0 allocs/op
// shape 1000x1000
func BenchmarkBigMul(b *testing.B) {
	scalar := tensor.Scalar[float32](1000)
	a1 := tensor.Range[float32](1000000).Reshape(1000, 1000).Div(scalar, nil)
	a2 := tensor.Range[float32](1000000).Reshape(1000, 1000).Div(scalar, nil)
	out := tensor.CreateEmptyTensor[float32](1000, 1000)
	for i := 0; i < b.N; i++ {
		a1.Mul(a2, out)
	}
	// fmt.Println(out.Get(999, 999))
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
