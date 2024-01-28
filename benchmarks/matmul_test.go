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
}

// BenchmarkMatMulUnite-8             36697             30950 ns/op           41115 B/op          6 allocs/op
// prealloc out tensor
// BenchmarkMatMulUnite-8            138862              7659 ns/op               8 B/op          1 allocs/op
func BenchmarkMatMulUnite(b *testing.B) {
	var dim types.Dim = 100
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
// BenchmarkMatMul-12            10         115.477.200 ns/op        13773644 B/op       2022 allocs/op
// BenchmarkMatMul-12            12          94.282.383 ns/op        13506368 B/op       2020 allocs/op
// BenchmarkMatMul-12            12          93.556.358 ns/op        13506376 B/op       2021 allocs/op
// BenchmarkMatMul-12            12          92.763.142 ns/op        13506342 B/op       2020 allocs/op
// BenchmarkMatMul-12            12          96.534.575 ns/op        13506392 B/op       2021 allocs/op
// AVX + threads
// BenchmarkMatMul-12            60          18.911.752 ns/op        12437393 B/op       2019 allocs/op
// BenchmarkMatMul-12            62          18.853.710 ns/op        12428761 B/op       2019 allocs/op
// BenchmarkMatMul-12            62          18.864.219 ns/op        12428772 B/op       2019 allocs/op
// BenchmarkMatMul-12            58          18.832.110 ns/op        12446719 B/op       2019 allocs/op
// BenchmarkMatMul-12            62          19.135.018 ns/op        12428771 B/op       2019 allocs/op
// numpy matmul ref:
// 4.184.719.133 ~ 4.910.045.600 ns
// go test -benchmem -run=^$ -bench ^BenchmarkMatMul$ gograd/benchmarks -benchmem -v -count=5
func BenchmarkMatMul(b *testing.B) {
	rng := tensor.NewRNG(-1)
	var size types.Dim = 1000
	// a1 := tensor.Range[float32](int(size*size)).Reshape(size, size)
	// b1 := tensor.Range[float32](int(size*size)).Reshape(size, size)
	a1 := rng.RandomFloat32(size, size)
	b1 := rng.RandomFloat32(size, size)
	for i := 0; i < b.N; i++ {
		a1.MatMul(b1)
	}
}
