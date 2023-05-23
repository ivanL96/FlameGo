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

	e1 := tensor.Range[int32](10).Reshape(5, 2).Transpose()
	e2 := tensor.Range[int32](10).Reshape(5, 2).Transpose().AsContinuous()
	f1 := tensor.Range[int32](10).Reshape(2, 5)
	ef1 := e1.Add(f1)
	ef2 := e2.Add(f1)
	assertEqualSlices(t, ef1.Data(), ef2.Data())
	assertEqualSlices(t, ef1.Shape(), tensor.Shape{2, 5})
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
	assertEqualSlices(t, ab.Shape(), tensor.Shape{3, 2})
}
