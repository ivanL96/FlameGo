package main

import (
	"gograd/tensor"
	"testing"
)

func TestAdd(t *testing.T) {
	a := tensor.InitEmptyTensor[int32](3, 2).Fill(2)
	b := tensor.InitEmptyTensor[int32](3, 1).Fill(3)
	ab := a.Add(b)
	assertEqualSlices(t, ab.Data(), []int32{5, 5, 5, 5, 5, 5})
	assertEqualSlices(t, ab.Shape(), tensor.Shape{3, 2})

	c := tensor.InitEmptyTensor[int32](3, 3).Fill(4)
	d := tensor.InitEmptyTensor[int32](1).Fill(1)
	cd := c.Add(d)
	assertEqualSlices(t, cd.Data(), []int32{5, 5, 5, 5, 5, 5, 5, 5, 5})
	assertEqualSlices(t, cd.Shape(), tensor.Shape{3, 3})

	e := tensor.InitEmptyTensor[int32](1).Fill(4)
	f := tensor.InitEmptyTensor[int32](1).Fill(1)
	ef := e.Add(f)
	assertEqualSlices(t, ef.Data(), []int32{5})
	assertEqualSlices(t, ef.Shape(), tensor.Shape{1})
}

func TestMul(t *testing.T) {
	a := tensor.InitEmptyTensor[int32](3, 2).Fill(2)
	b := tensor.InitEmptyTensor[int32](3, 1).Fill(3)
	ab := a.Mul(b)
	assertEqualSlices(t, ab.Data(), []int32{6, 6, 6, 6, 6, 6})
	assertEqualSlices(t, ab.Shape(), tensor.Shape{3, 2})
}
