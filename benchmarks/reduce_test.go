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
