package main

import (
	"gograd/tensor"
	"testing"
)

// ======================init
// BenchmarkMatMulSplit-8              1068           1.057.919 ns/op          161623 B/op       7525 allocs/op
// BenchmarkMatMulSplit-8               976           1.241.789 ns/op          161626 B/op       7525 allocs/op
// BenchmarkMatMulSplit-8               985           1.056.939 ns/op          161626 B/op       7525 allocs/op
// ======================removed copying of indexes
// BenchmarkMatMulSplit-8              1873            562.053 ns/op           41606 B/op         25 allocs/op
// BenchmarkMatMulSplit-8              2262            445.965 ns/op           41602 B/op         25 allocs/op
// BenchmarkMatMulSplit-8              2362            528.200 ns/op           41601 B/op         25 allocs/op
// ======================using get_fast logic
// BenchmarkMatMulSplit-8              4993            263.525 ns/op           41592 B/op         25 allocs/op
// BenchmarkMatMulSplit-8              4664            242.173 ns/op           41592 B/op         25 allocs/op
// BenchmarkMatMulSplit-8              5130            256.283 ns/op           41592 B/op         25 allocs/op
// ======================using preallocated out tensors
// BenchmarkMatMulSplit-8              4537            254.710 ns/op              42 B/op          2 allocs/op
// BenchmarkMatMulSplit-8              4986            211.388 ns/op              40 B/op          2 allocs/op
// BenchmarkMatMulSplit-8              5218            241.416 ns/op              39 B/op          2 allocs/op
// ======================Split version 2: native loop & prealloc output
// BenchmarkMatMulSplit-8             41451             28753 ns/op               9 B/op          1 allocs/op
// BenchmarkMatMulSplit-8             38742             28343 ns/op              10 B/op          1 allocs/op
// BenchmarkMatMulSplit-8             38750             31582 ns/op              10 B/op          1 allocs/op
func BenchmarkMatMulSplit(b *testing.B) {
	X := tensor.Range[int32](100*100).Reshape(100, 100)
	a1 := tensor.InitEmptyTensor[int32](50, 50)
	b1 := tensor.InitEmptyTensor[int32](50, 50)
	c1 := tensor.InitEmptyTensor[int32](50, 50)
	d1 := tensor.InitEmptyTensor[int32](50, 50)
	for i := 0; i < b.N; i++ {
		tensor.SplitTensor2(X, a1, b1, c1, d1)
		// tensor.SplitTensor2(X, nil, nil, nil, nil)
	}
}
