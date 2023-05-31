package main

import (
	"fmt"
	"gograd/tensor"
	types "gograd/tensor/types"
	"testing"
)

func TestBroadcast1(t *testing.T) {
	a := tensor.Range[int32](1, 4).Reshape(3, 1, 1)
	br_a := a.Broadcast(2, 3, 2, 3)
	fmt.Println(a.ToString())
	fmt.Println(br_a.ToString())
	// correct
	data := []int32{
		1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
		1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3}
	assertEqualSlices(t, br_a.Data(), data)
	assertEqualSlices(t, br_a.Shape(), types.Shape{2, 3, 2, 3})
}

func TestBroadcast2(t *testing.T) {
	// broadcast inner dim
	b := tensor.Range[int32](1, 2).Reshape(1, 1)
	br_b := b.Broadcast(6)
	fmt.Println(b.ToString())
	fmt.Println(br_b.ToString())
	assertEqualSlices(t, br_b.Data(), []int32{1, 1, 1, 1, 1, 1})
	assertEqualSlices(t, br_b.Shape(), types.Shape{1, 6})
}

func TestBroadcast3(t *testing.T) {
	// broadcast outer dim
	c := tensor.Range[int32](6).Reshape(3, 2)
	br_c := c.Broadcast(2, 3, 2)
	data := []int32{0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5}
	assertEqualSlices(t, br_c.Data(), data)
	assertEqualSlices(t, br_c.Shape(), types.Shape{2, 3, 2})
	fmt.Println(br_c.ToString())
}

func TestBroadcast4(t *testing.T) {
	d := tensor.Range[int32](12).Reshape(3, 2, 2)
	fmt.Println(d.ToString())
	br_d := d.Broadcast(3, 3, 1, 2)
	data := []int32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
	assertEqualSlices(t, br_d.Data(), data)
	assertEqualSlices(t, br_d.Shape(), types.Shape{3, 3, 2, 2})
	fmt.Println(br_d.ToString())
}

// Reshape(3, 2, 1)
//
//	a.Broadcast(3, 4, 1, 1, 3)
func TestBroadcast5(t *testing.T) {
	d := tensor.Range[int32](6).Reshape(3, 2, 1)
	fmt.Println(d.ToString())
	br_d := d.Broadcast(3, 4, 1, 1, 3)
	data := []int32{
		0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 0,
		0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 0, 0,
		0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 0, 0, 0,
		1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 0, 0, 0, 1,
		1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 0, 0, 0, 1, 1,
		1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5,
	}
	assertEqualSlices(t, br_d.Data(), data)
	assertEqualSlices(t, br_d.Shape(), types.Shape{3, 4, 3, 2, 3})
	fmt.Println(br_d.ToString())
}
