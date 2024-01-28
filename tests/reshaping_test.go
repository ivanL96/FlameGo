package main

import (
	"gograd/tensor"
	types "gograd/tensor/types"
	"testing"
)

// RESHAPING

func TestFlatten(t *testing.T) {
	super_nested_arr := tensor.CreateTensor([]int32{1, 2, 3, 4, 5, 6, 7, 8, 9}, types.Shape{3, 1, 3, 1, 1})
	assertEqualSlices(t, super_nested_arr.Flatten(nil).Shape(), types.Shape{9})
	assertEqualSlices(t, super_nested_arr.Flatten(nil).Data(), super_nested_arr.Data())
}

func TestReshape(t *testing.T) {
	a := tensor.CreateTensor([]int32{1, 2, 3, 4, 5, 6}, types.Shape{2, 3})
	assertEqualSlices(t, a.Reshape(3, 2).Shape(), types.Shape{3, 2})
	assertEqualSlices(t, a.Reshape(1, 1, 1, 3, 2).Shape(), types.Shape{1, 1, 1, 3, 2})
	assertEqualSlices(t, a.Reshape(1, 1, 3, 2).Data(), a.Data())
}
