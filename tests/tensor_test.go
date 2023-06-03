package main

import (
	"flamego/tensor"
	types "flamego/tensor/types"
	"fmt"
	"reflect"
	"testing"
)

func TestAsType(t *testing.T) {
	a := tensor.CreateTensor([]int32{1, 2, 3}, types.Shape{3, 1})
	b := tensor.AsType[int32, int32](a)
	assert(t, a.DType() == b.DType())

	c := tensor.AsType[int32, float32](a)
	assertStatement(t, a.DType().Kind(), NotEquals, c.DType().Kind())
	assertStatement(t, a.DType().Kind(), Equals, reflect.Int32)
	assertStatement(t, c.DType().Kind(), Equals, reflect.Float32)
}

func TestCopy(t *testing.T) {
	a := tensor.CreateTensor([]int32{1, 2, 3}, types.Shape{3, 1})
	b := a.Copy().SetData([]int32{7, 8, 9})
	a_data := a.Data()
	b_data := b.Data()
	assertNotEqualSlices(t, a_data, b_data)
	assert(t, &a != &b)
	assert(t, &(a_data) != &(b_data))
}

func TestCompare(t *testing.T) {
	a := tensor.CreateTensor([]int32{1, 2, 3}, types.Shape{3, 1})
	b := tensor.CreateTensor([]int32{1, 2, 3}, types.Shape{1, 3})
	c := tensor.CreateTensor([]int32{1, 2, 3, 4}, types.Shape{1, 4})
	d := tensor.CreateTensor([]int32{1, 2, 3}, types.Shape{1, 3})
	assert(t, !a.IsEqual(b))
	assert(t, !a.IsEqual(c))
	assert(t, b.IsEqual(d))

	// IsEqual is dim order aware
	a1 := tensor.Range[int32](4).Reshape(2, 2).Transpose()
	a2 := a1.AsContinuous(nil)
	assert(t, a1.IsEqual(a2))
}

func TestRange(t *testing.T) {
	var a *tensor.Tensor[int32] = nil
	var b *tensor.Tensor[int32] = nil

	a = tensor.Range[int32](6)
	b = tensor.CreateTensor([]int32{0, 1, 2, 3, 4, 5}, types.Shape{6})
	assertEqualSlices(t, a.Data(), b.Data())
	assertEqualSlices(t, a.Shape(), b.Shape())

	a = tensor.Range[int32](2, 6, 1)
	b = tensor.CreateTensor([]int32{2, 3, 4, 5}, types.Shape{4})
	assertEqualSlices(t, a.Data(), b.Data())
	assertEqualSlices(t, a.Shape(), b.Shape())

	a = tensor.Range[int32](1, 6, 2)
	b = tensor.CreateTensor([]int32{1, 3, 5}, types.Shape{3})
	assertEqualSlices(t, a.Data(), b.Data())
	assertEqualSlices(t, a.Shape(), b.Shape())
}

// test -timeout 30s -run ^TestFill$ flamego/tests
func TestFill(t *testing.T) {
	a := tensor.CreateEmptyTensor[int32](12)
	b := a.Copy()
	a.Fill(2)
	fmt.Println(b.ToString(), a.ToString())
	assertEqualSlices(t, a.Shape(), b.Shape())
	assertNotEqualSlices(t, a.Data(), b.Data())
}
