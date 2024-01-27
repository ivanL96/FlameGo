package main

import (
	"gograd/tensor"
	"gograd/tensor/types"
	"testing"
)

func TestSum(t *testing.T) {
	a := tensor.Range[int32](2*3*4).Reshape(2, 3, 4)
	b := a.SumAlongAxis(0, true)
	assertEqualSlices(t, b.Shape(), types.Shape{1, 3, 4})
	assertEqualSlices(t, b.Data(), []int32{12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34})
	b = a.SumAlongAxis(0, false)
	assertEqualSlices(t, b.Shape(), types.Shape{3, 4})
	assertEqualSlices(t, b.Data(), []int32{12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34})
}

func TestSum2(t *testing.T) {
	a := tensor.Range[int32](2*3*4*5).Reshape(5, 4, 3, 2, 1)
	b := a.IndexAdv("3,2,1").SumAlongAxis(0, true)
	assertEqualSlices(t, b.Shape(), types.Shape{1, 1})
	assertEqualSlices(t, b.Data(), []int32{173})
}
