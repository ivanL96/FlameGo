package main

import (
	"gograd/tensor"
	"testing"
)

// vec
// BenchmarkSum-12             1917            593.707 ns/op            4005 B/op         32 allocs/op
// BenchmarkSum-12             2019            590.547 ns/op            3868 B/op         32 allocs/op
// BenchmarkSum-12             1850            609.871 ns/op            4047 B/op         32 allocs/op
// BenchmarkSum-12             1996            596.353 ns/op            3892 B/op         32 allocs/op
// BenchmarkSum-12             1956            590.290 ns/op            3928 B/op         32 allocs/op
// fixed sum impl
// BenchmarkSum-12            10000            103.768 ns/op            2288 B/op         32 allocs/op
// BenchmarkSum-12            10000            101.975 ns/op            2280 B/op         32 allocs/op
// BenchmarkSum-12             9901            102.449 ns/op            2284 B/op         32 allocs/op
// BenchmarkSum-12            10000            102.381 ns/op            2280 B/op         32 allocs/op
// BenchmarkSum-12             9975            102.502 ns/op            2281 B/op         32 allocs/op
func BenchmarkSum(b *testing.B) {
	a1 := tensor.Range[int32](1000000).Reshape(1000, 1000) //.Fill(1)
	for i := 0; i < b.N; i++ {
		a1.Sum(false)
	}
}

// goos: windows
// goarch: amd64
// pkg: gograd/benchmarks
// cpu: 11th Gen Intel(R) Core(TM) i5-11400H @ 2.70GHz
// shape 1000 x 1000
// default SumAlongAxis(0, false)
// BenchmarkSumAxis-12          153           7.244.017 ns/op         6.120.516 B/op      38874 allocs/op
// BenchmarkSumAxis-12          162           7.287.277 ns/op         6.119.115 B/op      38874 allocs/op
// BenchmarkSumAxis-12          154           8.152.823 ns/op         6.120.345 B/op      38874 allocs/op
// using IndexAdv_()
// BenchmarkSumAxis-12          174           7.007.597 ns/op         6.045.399 B/op      33976 allocs/op
// BenchmarkSumAxis-12          174           7.006.540 ns/op         6.045.464 B/op      33976 allocs/op
// BenchmarkSumAxis-12          171           7.259.490 ns/op         6.045.796 B/op      33976 allocs/op
// 2D impl
// BenchmarkSumAxis-12          750           1.520.501 ns/op            9.613 B/op          9 allocs/op
// BenchmarkSumAxis-12          739           1.544.970 ns/op            9.692 B/op          9 allocs/op
// BenchmarkSumAxis-12          720           1.521.726 ns/op            9.835 B/op          9 allocs/op
// parallel
// BenchmarkSumAxis-12         2139            508.514 ns/op          142.328 B/op       2013 allocs/op
// BenchmarkSumAxis-12         2271            504.863 ns/op          142.108 B/op       2013 allocs/op
// BenchmarkSumAxis-12         2374            507.044 ns/op          142.026 B/op       2013 allocs/op
// goroutine per core
// BenchmarkSumAxis-12         4311            235.806 ns/op            7.001 B/op         36 allocs/op
// BenchmarkSumAxis-12         4759            225.101 ns/op            6.914 B/op         36 allocs/op
// BenchmarkSumAxis-12         5380            221.528 ns/op            6.816 B/op         36 allocs/op
func BenchmarkSumAxis(b *testing.B) {
	a1 := tensor.Range[int32](1000000).Reshape(1000, 1000)
	for i := 0; i < b.N; i++ {
		a1.SumAlongAxis(0, false)
	}
}

// goos: windows
// goarch: amd64
// pkg: gograd/benchmarks
// cpu: 11th Gen Intel(R) Core(TM) i5-11400H @ 2.70GHz
// vec 1000x1000
// BenchmarkMean-12            2976            363.799 ns/op            1466 B/op          5 allocs/op
// BenchmarkMean-12            3102            360.796 ns/op            1411 B/op          5 allocs/op
// BenchmarkMean-12            3114            359.069 ns/op            1406 B/op          5 allocs/op
// BenchmarkMean-12            3086            361.250 ns/op            1418 B/op          5 allocs/op
// BenchmarkMean-12            2866            361.388 ns/op            1517 B/op          5 allocs/op
// fixed vectorized impl
// BenchmarkMean-12           11535            102.478 ns/op            2229 B/op         32 allocs/op
// BenchmarkMean-12           10000            102.221 ns/op            2280 B/op         32 allocs/op
// BenchmarkMean-12           10000            102.291 ns/op            2280 B/op         32 allocs/op
func BenchmarkMean(b *testing.B) {
	a1 := tensor.Range[int32](1000000).Reshape(1000, 1000) //.Fill(1)
	for i := 0; i < b.N; i++ {
		a1.Mean(false)
	}
}
