package main

import (
	"gograd/tensor"
	"testing"
)

func TestSort(t *testing.T) {
	a := tensor.InitTensor(
		[]int32{3, 2, 5, 0, 6, 1, 0, 5}, tensor.Shape{2, 2, 2})
	sort_a := a.Sort()
	assertEqualSlices(t,
		sort_a.Data(), []int32{0, 0, 1, 2, 3, 5, 5, 6})
}
