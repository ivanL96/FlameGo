package main

import (
	"gograd/tensor"
	"testing"
)

func TestAdd(t *testing.T) {
	// not finished. need to complete Add()
	a := tensor.InitEmptyTensor[int32](3, 2)
	a.Fill(2)
	b := tensor.InitEmptyTensor[int32](3, 1)
	b.Fill(3)
	// fmt.Println("a", a.ToString(), "b", b.ToString())

	c := a.Add(b)
	want := []int32{5, 5, 5, 5, 5, 5}
	if !tensor.Compare_slices(c.Data(), want) {
		t.Errorf("Have %v; want %v", c.Data(), want)
	}
}

func TestBroadcast(t *testing.T) {
	a := tensor.InitEmptyTensor[int32](3, 2)
	a.Fill(7)
	a.Broadcast(3, 1, 1)
	want := tensor.Shape{3, 3, 2}
	if !tensor.Compare_slices(a.Shape(), want) {
		t.Errorf("Have %v; want %v", a.Shape(), want)
	}
	b := tensor.InitEmptyTensor[int32](1, 1)
	b.Fill(7)
	b.Broadcast(6)
	if !tensor.Compare_slices(b.Shape(), tensor.Shape{1, 6}) {
		t.Errorf("Have %v; want %v", a.Shape(), want)
	}
}

func TestFlatten(t *testing.T) {
	super_nested_arr := tensor.InitTensor([]int32{1, 2, 3, 4, 5, 6, 7, 8, 9}, tensor.Shape{3, 1, 3, 1, 1})
	super_nested_arr.Flatten()
	assert(tensor.Compare_slices(super_nested_arr.Shape(), tensor.Shape{9}))
}
