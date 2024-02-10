package main

import (
	_ "fmt"
	"gograd/tensor"
	"gograd/tensor/types"
	"testing"
)

// ======================input shape 100x100
// BenchmarkMatMulSplit-8              1068           1.057.919 ns/op          161623 B/op       7525 allocs/op
// BenchmarkMatMulSplit-8               985           1.056.939 ns/op          161626 B/op       7525 allocs/op
// ======================removed copying of indexes
// BenchmarkMatMulSplit-8              2262            445.965 ns/op           41602 B/op         25 allocs/op
// BenchmarkMatMulSplit-8              2362            528.200 ns/op           41601 B/op         25 allocs/op
// ======================using Get_fast logic
// BenchmarkMatMulSplit-8              4664            242.173 ns/op           41592 B/op         25 allocs/op
// BenchmarkMatMulSplit-8              5130            256.283 ns/op           41592 B/op         25 allocs/op
// ======================using preallocated out tensors
// BenchmarkMatMulSplit-8              4986            211.388 ns/op              40 B/op          2 allocs/op
// BenchmarkMatMulSplit-8              5218            241.416 ns/op              39 B/op          2 allocs/op
// ======================Split version 2: native loop & prealloc output
// BenchmarkMatMulSplit-8             43425             30.704 ns/op               9 B/op          1 allocs/op
// BenchmarkMatMulSplit-8             42649             27.424 ns/op               9 B/op          1 allocs/op
// ======================Split version 2: native loop & prealloc output & switch case
// BenchmarkMatMulSplit-8             43924             24.775 ns/op               9 B/op          1 allocs/op
// BenchmarkMatMulSplit-8             45511             24.464 ns/op               9 B/op          1 allocs/op
// ======================Split version 2: + copy instead of set data loop
// BenchmarkMatMulSplit-8            160952              7864 ns/op               8 B/op          1 allocs/op
// BenchmarkMatMulSplit-8            159236              7121 ns/op               8 B/op          1 allocs/op
func BenchmarkMatMulSplit(b *testing.B) {
	var dim types.Dim = 100
	X := tensor.Range[int32](int(dim*dim)).Reshape(dim, dim)
	a1 := tensor.CreateEmptyTensor[int32](dim/2, dim/2)
	b1 := tensor.CreateEmptyTensor[int32](dim/2, dim/2)
	c1 := tensor.CreateEmptyTensor[int32](dim/2, dim/2)
	d1 := tensor.CreateEmptyTensor[int32](dim/2, dim/2)
	for i := 0; i < b.N; i++ {
		tensor.SplitTensor(X, a1, b1, c1, d1)
		// tensor.SplitTensor(X, nil, nil, nil, nil)
	}
	t := tensor.AnyErrors(a1, b1, c1, d1)
	if t != nil {
		t.MustAssert()
	}
}

// goos: windows
// goarch: amd64
// pkg: gograd/benchmarks
// cpu: 11th Gen Intel(R) Core(TM) i5-11400H @ 2.70GHz
// 1000 x 1000
// BenchmarkMatMulUnite-12             6656            186.442 ns/op            1215 B/op          1 allocs/op
// BenchmarkMatMulUnite-12             6686            179.048 ns/op            1210 B/op          1 allocs/op
// BenchmarkMatMulUnite-12             5698            195.163 ns/op            1418 B/op          1 allocs/op
// BenchmarkMatMulUnite-12             6565            175.231 ns/op            1232 B/op          1 allocs/op
// BenchmarkMatMulUnite-12             6696            177.467 ns/op            1208 B/op          1 allocs/op
func BenchmarkMatMulUnite(b *testing.B) {
	var dim types.Dim = 1000
	X := tensor.Range[int32](int(dim*dim)).Reshape(dim, dim)
	a1 := tensor.CreateEmptyTensor[int32](dim/2, dim/2)
	b1 := tensor.CreateEmptyTensor[int32](dim/2, dim/2)
	c1 := tensor.CreateEmptyTensor[int32](dim/2, dim/2)
	d1 := tensor.CreateEmptyTensor[int32](dim/2, dim/2)
	tensor.SplitTensor(X, a1, b1, c1, d1)
	for i := 0; i < b.N; i++ {
		tensor.UniteTensors(a1, b1, c1, d1, X)
	}
}

// O(N^3) impl
// matmul for 1000x1000
// goos: windows
// goarch: amd64
// pkg: gograd/benchmarks
// cpu: 11th Gen Intel(R) Core(TM) i5-11400H @ 2.70GHz
// threads
// BenchmarkMatMul-12            12          94.282.383 ns/op        13506368 B/op       2020 allocs/op
// BenchmarkMatMul-12            12          93.556.358 ns/op        13506376 B/op       2021 allocs/op
// BenchmarkMatMul-12            12          92.763.142 ns/op        13506342 B/op       2020 allocs/op
// AVX + threads
// BenchmarkMatMul-12            60          18.911.752 ns/op        12437393 B/op       2019 allocs/op
// BenchmarkMatMul-12            62          18.864.219 ns/op        12428772 B/op       2019 allocs/op
// BenchmarkMatMul-12            58          18.832.110 ns/op        12446719 B/op       2019 allocs/op
// AVX + threads + thread affinity, moved some code outside the loop
// BenchmarkMatMul-12            62          18.690.979 ns/op        12428797 B/op       2019 allocs/op
// BenchmarkMatMul-12            63          18.725.068 ns/op        12424663 B/op       2019 allocs/op
// BenchmarkMatMul-12            63          18.659.789 ns/op        12424688 B/op       2019 allocs/op
// AVX512
// BenchmarkMatMul-12            70          16.607.281 ns/op        12399209 B/op       2019 allocs/op
// BenchmarkMatMul-12            70          16.702.821 ns/op        12399208 B/op       2019 allocs/op
// BenchmarkMatMul-12            70          16.621.721 ns/op        12399222 B/op       2019 allocs/op
// AVX512 + changed TrC() to TrC2D()
// BenchmarkMatMul-12            87          12.798.697 ns/op        12354380 B/op       2014 allocs/op
// BenchmarkMatMul-12            91          13.362.971 ns/op        12346251 B/op       2014 allocs/op
// BenchmarkMatMul-12            91          12.793.149 ns/op        12346250 B/op       2014 allocs/op
// AVX512 + multithreaded TrC2D()
// BenchmarkMatMul-12           100          11.420.477 ns/op        12450394 B/op       4015 allocs/op
// BenchmarkMatMul-12           102          11.561.105 ns/op        12447245 B/op       4015 allocs/op
// BenchmarkMatMul-12           102          11.648.135 ns/op        12447252 B/op       4015 allocs/op
// splitted by blocks, improved cache locality
// BenchmarkMatMul-12           106          10.039.453 ns/op        12329846 B/op        557 allocs/op
// BenchmarkMatMul-12           115           9.977.032 ns/op        12308991 B/op        557 allocs/op
// BenchmarkMatMul-12           112           9.891.651 ns/op        12315537 B/op        557 allocs/op
//
// numpy matmul ref                          10.645.914 ns/op
// 10000x10000
// mine 54.563.731.700
// numpy 7.661.700.820
//
// go test -benchmem -run=^$ -bench ^BenchmarkMatMul$ gograd/benchmarks -benchmem -v -count=5
func BenchmarkMatMul(b *testing.B) {
	rng := tensor.NewRNG(-1)
	var size types.Dim = 1000
	// a1 := tensor.Range[float32](int(size*size)).Reshape(size, size)
	// b1 := tensor.Range[float32](int(size*size)).Reshape(size, size)
	a1 := rng.RandomFloat32(size, size)
	b1 := rng.RandomFloat32(size, size)
	a1.MatMul(b1)
	for i := 0; i < b.N; i++ {
		a1.MatMul(b1)
	}
	a1.MustAssert()
	b1.MustAssert()
}

// goos: windows
// goarch: amd64
// pkg: gograd/benchmarks
// cpu: 11th Gen Intel(R) Core(TM) i5-11400H @ 2.70GHz
// 10 x 1000 x 1000
// BenchmarkDot-12                8         142.658.350 ns/op        223290256 B/op     40265 allocs/op
// BenchmarkDot-12                8         137.481.225 ns/op        223285272 B/op     40254 allocs/op
// BenchmarkDot-12                7         143.675.343 ns/op        226142722 B/op     40254 allocs/op
// BenchmarkDot-12                8         138.330.538 ns/op        223285312 B/op     40254 allocs/op
// BenchmarkDot-12                8         139.397.838 ns/op        223285272 B/op     40254 allocs/op
func BenchmarkDot(b *testing.B) {
	rng := tensor.NewRNG(-1)
	var size types.Dim = 1000
	var batch types.Dim = 10
	a1 := rng.RandomFloat32(batch, size, size)
	b1 := rng.RandomFloat32(batch, size, size)
	for i := 0; i < b.N; i++ {
		a1.Dot(b1)
	}
}
