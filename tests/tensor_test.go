package main

import (
	"fmt"
	"gograd/tensor"
	"testing"
)

func assertEqualSlices[DT tensor.Number](slice1 []DT, slice2 []DT, t *testing.T) {
	if !tensor.Compare_slices(slice1, slice2) {
		t.Errorf("Slices must be equal. Got %v and %v", slice1, slice2)
	}
}

func TestAdd(t *testing.T) {
	a := tensor.InitEmptyTensor[int32](3, 2).Fill(2)
	b := tensor.InitEmptyTensor[int32](3, 1).Fill(3)
	assertEqualSlices(a.Add(b).Data(), []int32{5, 5, 5, 5, 5, 5}, t)

	c := tensor.InitEmptyTensor[int32](1, 2).Fill(2)
	d := tensor.InitEmptyTensor[int32](3, 1).Fill(3)
	fmt.Println(c.Add(d).Shape())
	assertEqualSlices(c.Add(d).Data(), []int32{5, 5, 5, 5, 5, 5}, t)
}

func TestBroadcast(t *testing.T) {
	a := tensor.InitEmptyTensor[int32](3, 2)
	a.Fill(7)
	a.Broadcast(3, 1, 1)
	want := tensor.Shape{3, 3, 2}
	assertEqualSlices(a.Shape(), want, t)

	b := tensor.InitEmptyTensor[int32](1, 1)
	b.Fill(7)
	b.Broadcast(6)
	assertEqualSlices(b.Shape(), tensor.Shape{1, 6}, t)
}

func TestFlatten(t *testing.T) {
	super_nested_arr := tensor.InitTensor([]int32{1, 2, 3, 4, 5, 6, 7, 8, 9}, tensor.Shape{3, 1, 3, 1, 1})
	super_nested_arr.Flatten()
	assertEqualSlices(super_nested_arr.Shape(), tensor.Shape{9}, t)
}
