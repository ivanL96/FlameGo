package main

import (
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
	var dim types.Dim = 1000
	X := tensor.Range[int32](int(dim*dim)).Reshape(dim, dim)
	a1 := tensor.InitEmptyTensor[int32](dim/2, dim/2)
	b1 := tensor.InitEmptyTensor[int32](dim/2, dim/2)
	c1 := tensor.InitEmptyTensor[int32](dim/2, dim/2)
	d1 := tensor.InitEmptyTensor[int32](dim/2, dim/2)
	for i := 0; i < b.N; i++ {
		tensor.SplitTensor(X, a1, b1, c1, d1)
		// tensor.SplitTensor(X, nil, nil, nil, nil)
	}
}

// O(N^3) impl
// BenchmarkMatMul-8   	       1		  69.653.349.600 ns/op	   12020000 B/op	      31 allocs/op
// fast get
// BenchmarkMatMul-8              1       33.566.461.400 ns/op       12018176 B/op         24 allocs/op
// numpy matmul 4.910.045.600
func BenchmarkMatMul(b *testing.B) {
	a1 := tensor.Range[int32](1000*1000).Reshape(1000, 1000)
	b1 := tensor.Range[int32](1000*1000).Reshape(1000, 1000)
	for i := 0; i < b.N; i++ {
		a1.MatMul(b1)
	}
}
