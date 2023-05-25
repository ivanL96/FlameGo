package main

import (
	"gograd/tensor"
	types "gograd/tensor/types"
	"testing"
)

func TestAdd(t *testing.T) {
	a := tensor.InitEmptyTensor[int32](3, 2).Fill(2)
	b := tensor.InitEmptyTensor[int32](3, 1).Fill(3)
	ab := a.Add(b)
	assertEqualSlices(t, ab.Data(), []int32{5, 5, 5, 5, 5, 5})
	assertEqualSlices(t, ab.Shape(), types.Shape{3, 2})

	c := tensor.InitEmptyTensor[int32](3, 3).Fill(4)
	d := tensor.InitEmptyTensor[int32](1).Fill(1)
	cd := c.Add(d)
	assertEqualSlices(t, cd.Data(), []int32{5, 5, 5, 5, 5, 5, 5, 5, 5})
	assertEqualSlices(t, cd.Shape(), types.Shape{3, 3})

	e := tensor.InitEmptyTensor[int32](1).Fill(4)
	f := tensor.InitEmptyTensor[int32](1).Fill(1)
	ef := e.Add(f)
	assertEqualSlices(t, ef.Data(), []int32{5})
	assertEqualSlices(t, ef.Shape(), types.Shape{1})

	e1 := tensor.Range[int32](10).Reshape(5, 2).Transpose()
	e2 := tensor.Range[int32](10).Reshape(5, 2).Transpose().AsContinuous()
	f1 := tensor.Range[int32](10).Reshape(2, 5)
	ef1 := e1.Add(f1)
	ef2 := e2.Add(f1)
	assertEqualSlices(t, ef1.Data(), ef2.Data())
	assertEqualSlices(t, ef1.Shape(), types.Shape{2, 5})
	// fmt.Println(ef1.ToString())

	g1 := tensor.Range[int32](5).Reshape(5, 1).Transpose()
	g2 := tensor.Range[int32](15).Reshape(5, 3).Transpose()
	g22 := tensor.Range[int32](15).Reshape(5, 3).Transpose().AsContinuous()
	g3 := g1.Add(g2)
	g4 := g1.Add(g22)
	assertEqualSlices(t, g3.Data(), g4.Data())
}

func TestMul(t *testing.T) {
	a := tensor.InitEmptyTensor[int32](3, 2).Fill(2)
	b := tensor.InitEmptyTensor[int32](3, 1).Fill(3)
	ab := a.Mul(b)
	assertEqualSlices(t, ab.Data(), []int32{6, 6, 6, 6, 6, 6})
	assertEqualSlices(t, ab.Shape(), types.Shape{3, 2})
}

func TestMatMul(t *testing.T) {
	a := tensor.Range[int32](9).Reshape(3, 3)
	b := tensor.Range[int32](9).Reshape(3, 3)
	c := a.MatMul(b)
	assertEqualSlices(t, c.Data(), []int32{15, 18, 21, 42, 54, 66, 69, 90, 111})
	assertEqualSlices(t, c.Shape(), types.Shape{3, 3})

	a1 := tensor.Range[int32](3).Reshape(1, 3)
	b1 := tensor.Range[int32](3).Reshape(3, 1)
	c1 := a1.MatMul(b1)
	assertEqualSlices(t, c1.Data(), []int32{5})
	assertEqualSlices(t, c1.Shape(), types.Shape{1, 1})

	a2 := tensor.Range[int32](8).Reshape(2, 4)
	b2 := tensor.Range[int32](8).Reshape(4, 2)
	c2 := a2.MatMul(b2)
	assertEqualSlices(t, c2.Data(), []int32{28, 34, 76, 98})
	assertEqualSlices(t, c2.Shape(), types.Shape{2, 2})
}
