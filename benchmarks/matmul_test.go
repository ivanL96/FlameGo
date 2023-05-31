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
// BenchmarkMatMul-8   	       1		  69.653.349.600 ns/op	   12.020.000 B/op	      31 allocs/op
// fast get
// BenchmarkMatMul-8              1       33.566.461.400 ns/op       12.018.176 B/op         24 allocs/op
// calculate index inplace
// BenchmarkMatMul-8              1       17.409.256.100 ns/op       12.018.264 B/op         26 allocs/op
// removed extra computations, minor loop unrolling
// BenchmarkMatMul-8              1        6.891.281.300 ns/op        12.019.960 B/op         29 allocs/op
// BenchmarkMatMul-8              1        7.084.445.400 ns/op        12.018.168 B/op         25 allocs/op
//
//	Using noasm Dot()
//
// BenchmarkMatMul-8              1        3.074.711.000 ns/op        20.032.064 B/op         40 allocs/op
// GOASM AVX !!!!!!
// BenchmarkMatMul-8              2          509.863.150 ns/op        20.024.276 B/op    1000028 allocs/op
// BenchmarkMatMul-8              2          555.004.700 ns/op        20.024.208 B/op    1000027 allocs/op
// numpy matmul:
// 4.184.719.133 ~ 4.910.045.600 ns
func BenchmarkMatMul(b *testing.B) {
	var size types.Dim = 1000
	// c := tensor.Scalar(float32(size))
	a1 := tensor.Range[float32](int(size*size)).Reshape(size, size)
	b1 := tensor.Range[float32](int(size*size)).Reshape(size, size)
	for i := 0; i < b.N; i++ {
		a1.MatMul(b1)
	}
	// c1 := a1.MatMul(b1)
	// fmt.Println(tensor.AsType[float32, int32](c1).Hash())
}
