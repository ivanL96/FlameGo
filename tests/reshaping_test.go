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

func TestUnsqueeze(t *testing.T) {
	a := tensor.CreateTensor([]int32{1, 2, 3, 4, 5, 6}, types.Shape{6})
	a.Unsqueeze(1)
	a.MustAssert()
	assertEqualSlices(t, a.Shape(), types.Shape{6, 1})
}
func TestSqueeze(t *testing.T) {
	a := tensor.CreateTensor([]int32{1, 2, 3, 4, 5, 6}, types.Shape{1, 6, 1})
	a = a.Squeeze()
	a.MustAssert()
	assertEqualSlices(t, a.Shape(), types.Shape{6})
}

func TestTensorList(t *testing.T) {
	a := tensor.CreateTensor([]int32{1, 2, 3, 4, 5, 6}, types.Shape{1, 6})
	b := tensor.CreateTensor([]int32{7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9}, types.Shape{2, 6})
	c := tensor.CreateTensor([]int32{0, 1, 0, 2, 0, 3}, types.Shape{1, 6})
	tensor.MustAssertAll(a, b, c)
	tlist := &tensor.TensorList[int32]{}
	tlist.Append(a)
	tensor.MustAssertAll(tlist.StackedTensors, a)
	tlist.Append(b)
	tensor.MustAssertAll(tlist.StackedTensors, b)
	tlist.Append(c)
	tensor.MustAssertAll(tlist.StackedTensors, c)
	// fmt.Println(tlist.StackedTensors.ToString())
	assertEqualSlices(t, tlist.StackedTensors.Shape(), types.Shape{4, 6})
}

func TestShapeStack(t *testing.T) {
	a := types.Shape{1, 2, 3}
	b := types.Shape{3, 2, 3}
	c := types.Shape{2, 2, 3}
	stacked, err := types.StackShapes(0, a, b, c)
	assert(t, err == nil)
	assertEqualSlices(t, stacked, types.Shape{6, 2, 3})
}
