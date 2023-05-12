package main

import (
	"fmt"
	"gograd/tensor"
	"testing"
)

// RESHAPING
func TestBroadcast(t *testing.T) {
	a := tensor.InitEmptyTensor[int32](3, 2)
	a.Fill(7)
	a.Broadcast(3, 1, 1)
	want := tensor.Shape{3, 3, 2}
	assertEqualSlices(t, a.Shape(), want)

	b := tensor.InitEmptyTensor[int32](1, 1)
	b.Fill(7)
	b.Broadcast(6)
	assertEqualSlices(t, b.Shape(), tensor.Shape{1, 6})
}

func TestFlatten(t *testing.T) {
	super_nested_arr := tensor.InitTensor([]int32{1, 2, 3, 4, 5, 6, 7, 8, 9}, tensor.Shape{3, 1, 3, 1, 1})
	super_nested_arr.Flatten()
	assertEqualSlices(t, super_nested_arr.Shape(), tensor.Shape{9})
}

func TestReshape(t *testing.T) {
	a := tensor.InitTensor([]int32{1, 2, 3, 4, 5, 6}, tensor.Shape{2, 3})
	assertEqualSlices(t, a.Reshape(3, 2).Shape(), tensor.Shape{3, 2})
	assertEqualSlices(t, a.Reshape(1, 1, 1, 3, 2).Shape(), tensor.Shape{1, 1, 1, 3, 2})
}

func TestIndex(t *testing.T) {
	a := tensor.InitTensor([]int32{0, 1, 2, 3, 4, 5, 6, 7}, tensor.Shape{2, 2, 2})
	sub1 := a.Index(1)
	assertEqualSlices(t, sub1.Data(), []int32{4, 5, 6, 7})
	assertEqualSlices(t, sub1.Shape(), tensor.Shape{2, 2})

	sub2 := sub1.Index(1)
	assertEqualSlices(t, sub2.Data(), []int32{6, 7})
	fmt.Println(sub2.Shape())
	assertEqualSlices(t, sub2.Shape(), tensor.Shape{2})

}
