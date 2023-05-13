package main

import (
	"gograd/tensor"
	"testing"
)

func TestAsType(t *testing.T) {
	a := tensor.InitTensor([]int32{1, 2, 3}, tensor.Shape{3, 1})
	b := tensor.AsType[int32, int32](a)
	assert(t, a.DType() == b.DType())
}

func TestCopy(t *testing.T) {
	a := tensor.InitTensor([]int32{1, 2, 3}, tensor.Shape{3, 1})
	b := a.Copy().Set([]int32{7, 8, 9})
	a_data := a.Data()
	b_data := b.Data()
	assertNotEqualSlices(t, a_data, b_data)
	assert(t, &a != &b)
	assert(t, &(a_data) != &(b_data))
}

func TestCompare(t *testing.T) {
	a := tensor.InitTensor([]int32{1, 2, 3}, tensor.Shape{3, 1})
	b := tensor.InitTensor([]int32{1, 2, 3}, tensor.Shape{1, 3})
	c := tensor.InitTensor([]int32{1, 2, 3, 4}, tensor.Shape{1, 4})
	assert(t, a.IsEqual(b))
	assert(t, !a.IsEqual(c))
}

func TestRange(t *testing.T) {
	var a *tensor.Tensor[int32] = nil
	var b *tensor.Tensor[int32] = nil

	a = tensor.Range[int32](6)
	b = tensor.InitTensor([]int32{0, 1, 2, 3, 4, 5}, tensor.Shape{6})
	assertEqualSlices(t, a.Data(), b.Data())
	assertEqualSlices(t, a.Shape(), b.Shape())

	a = tensor.Range[int32](2, 6, 1)
	b = tensor.InitTensor([]int32{2, 3, 4, 5}, tensor.Shape{4})
	assertEqualSlices(t, a.Data(), b.Data())
	assertEqualSlices(t, a.Shape(), b.Shape())

	a = tensor.Range[int32](1, 6, 2)
	b = tensor.InitTensor([]int32{1, 3, 5}, tensor.Shape{3})
	assertEqualSlices(t, a.Data(), b.Data())
	assertEqualSlices(t, a.Shape(), b.Shape())
}
