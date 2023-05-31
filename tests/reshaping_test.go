package main

import (
	"fmt"
	"gograd/tensor"
	types "gograd/tensor/types"
	"testing"
)

// RESHAPING
func TestBroadcast(t *testing.T) {
	a := tensor.Range[int32](1, 4).Reshape(3, 1, 1)
	br_a := a.Broadcast(2, 3, 2, 3)
	fmt.Println(a.ToString())
	fmt.Println(br_a.ToString())
	assertEqualSlices(t, br_a.Shape(), types.Shape{2, 3, 2, 3})

	b := tensor.Range[int32](1, 2).Reshape(1, 1)
	br_b := b.Broadcast(6)
	assertEqualSlices(t, br_b.Shape(), types.Shape{1, 6})
	fmt.Println(b.ToString())
	fmt.Println(br_b.ToString())

	c := tensor.Range[int32](1, 2).Reshape(1)
	br_c := c.Broadcast(3)
	assertEqualSlices(t, br_c.Shape(), types.Shape{3})
	fmt.Println(br_c.ToString())

	d := tensor.Range[int32](12).Reshape(3, 2, 2)
	fmt.Println(d.ToString())
	br_d := d.Broadcast(3, 3, 1, 2)
	assertEqualSlices(t, br_d.Shape(), types.Shape{3, 3, 2, 2})
	fmt.Println(br_d.ToString())
}

func TestFlatten(t *testing.T) {
	super_nested_arr := tensor.CreateTensor([]int32{1, 2, 3, 4, 5, 6, 7, 8, 9}, types.Shape{3, 1, 3, 1, 1})
	assertEqualSlices(t, super_nested_arr.Flatten(nil).Shape(), types.Shape{9})
}

func TestReshape(t *testing.T) {
	a := tensor.CreateTensor([]int32{1, 2, 3, 4, 5, 6}, types.Shape{2, 3})
	assertEqualSlices(t, a.Reshape(3, 2).Shape(), types.Shape{3, 2})
	assertEqualSlices(t, a.Reshape(1, 1, 1, 3, 2).Shape(), types.Shape{1, 1, 1, 3, 2})
}
