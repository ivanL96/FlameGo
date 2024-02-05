package main

import (
	"gograd/tensor"
	"gograd/tensor/types"
	"testing"
)

func TestSumAlongAxis(t *testing.T) {
	a := tensor.Range[int32](2*3*4).Reshape(2, 3, 4)
	b := a.SumAlongAxis(0, true)
	assertEqualSlices(t, b.Shape(), types.Shape{1, 3, 4})
	assertEqualSlices(t, b.Data(), []int32{12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34})
	b = a.SumAlongAxis(0, false)
	assertEqualSlices(t, b.Shape(), types.Shape{3, 4})
	assertEqualSlices(t, b.Data(), []int32{12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34})
}

func TestSumAlongAxis2(t *testing.T) {
	axis := 0
	a := tensor.Range[float32](6).Reshape(2, 3)
	out := a.SumAlongAxis(uint(axis), true).MustAssert()
	assertEqualSlices(t, out.Data(), []float32{3, 5, 7})
	assertEqualSlices(t, out.Shape(), types.Shape{1, 3})
}

func TestSum2(t *testing.T) {
	a := tensor.Range[int32](2*3*4*5).Reshape(5, 4, 3, 2, 1)
	b := a.IndexAdv("3,2,1").SumAlongAxis(0, true).MustAssert()
	assertEqualSlices(t, b.Shape(), types.Shape{1, 1})
	assertEqualSlices(t, b.Data(), []int32{173})
}

func TestSum3(t *testing.T) {
	a := tensor.Range[int32](3*2).Reshape(1, 2, 3)
	b := a.Sum(true)
	assertEqualSlices(t, b.Shape(), types.Shape{1, 1, 1})
	assertEqualSlices(t, b.Data(), []int32{1 + 2 + 3 + 4 + 5})
}

func TestMean(t *testing.T) {
	a := tensor.Range[float32](1000).Reshape(100, 2, 5)
	b := a.Mean(true)
	assertEqualSlices(t, b.Shape(), types.Shape{1, 1, 1})
	assertEqualSlices(t, b.Data(), []float32{499.5})
}

func TestMax(t *testing.T) {
	a := tensor.Range[float32](100).Reshape(10, 10)
	b := a.Max(true)
	tensor.MustAssertAll(a, b)
	assertEqualSlices(t, b.Shape(), types.Shape{1, 1})
	assertEqualSlices(t, b.Data(), []float32{99})
}

func TestMin(t *testing.T) {
	a := tensor.Range[float32](100).Reshape(10, 10)
	b := a.Min(true)
	tensor.MustAssertAll(a, b)
	assertEqualSlices(t, b.Shape(), types.Shape{1, 1})
	assertEqualSlices(t, b.Data(), []float32{0})
}
