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

func BenchmarkSumAxis(b *testing.B) {
	a1 := tensor.Range[int32](1000000).Reshape(1000, 1000) //.Fill(1)
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
