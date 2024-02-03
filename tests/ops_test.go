package main

import (
	"gograd/tensor"
	types "gograd/tensor/types"
	"testing"
)

func TestAdd(t *testing.T) {
	a := tensor.CreateEmptyTensor[float32](3, 2).Fill(2)
	b := tensor.CreateEmptyTensor[float32](3, 1).Fill(3)
	ab := a.Add(b, nil)
	assertEqualSlices(t, ab.Data(), []float32{5, 5, 5, 5, 5, 5})
	assertEqualSlices(t, ab.Shape(), types.Shape{3, 2})
	a.MustAssert()
	b.MustAssert()
	ab.MustAssert()
}

func TestAdd1(t *testing.T) {
	c := tensor.CreateEmptyTensor[int32](3, 3).Fill(4)
	d := tensor.CreateEmptyTensor[int32](1).Fill(1)
	cd := c.Add(d, nil)
	assertEqualSlices(t, cd.Data(), []int32{5, 5, 5, 5, 5, 5, 5, 5, 5})
	assertEqualSlices(t, cd.Shape(), types.Shape{3, 3})
	c.MustAssert()
	d.MustAssert()
	cd.MustAssert()
}

func TestAdd2(t *testing.T) {
	e := tensor.CreateEmptyTensor[int32](1).Fill(4)
	f := tensor.CreateEmptyTensor[int32](1).Fill(1)
	ef := e.Add(f, nil)
	assertEqualSlices(t, ef.Data(), []int32{5})
	assertEqualSlices(t, ef.Shape(), types.Shape{1})
	e.MustAssert()
	f.MustAssert()
	ef.MustAssert()
}

func TestAdd3(t *testing.T) {
	e1 := tensor.Range[int32](10).Reshape(5, 2).T()
	e2 := tensor.Range[int32](10).Reshape(5, 2).T().AsContinuous()
	f1 := tensor.Range[int32](10).Reshape(2, 5)
	ef1 := e1.Add(f1, nil)
	ef2 := e2.Add(f1, nil)
	assertEqualSlices(t, ef1.Data(), ef2.Data())
	assertEqualSlices(t, ef1.Shape(), types.Shape{2, 5})
	tensor.MustAssertAll(e1, e2, f1, ef1, ef2)
}

func TestAdd4(t *testing.T) {
	g1 := tensor.Range[int32](5).Reshape(5, 1).T()
	g2 := tensor.Range[int32](15).Reshape(5, 3).T()
	g22 := tensor.Range[int32](15).Reshape(5, 3).T().AsContinuous()
	g3 := g1.Add(g2, nil)
	g4 := g1.Add(g22, nil)
	assertEqualSlices(t, g3.Data(), g4.Data())
	tensor.MustAssertAll(g1, g2, g22, g3, g4)
}

func TestAddInplace(t *testing.T) {
	a1 := tensor.Range[float32](4).Reshape(2, 2)
	b1 := tensor.Range[float32](4).Reshape(2, 2)
	a1.Add(b1, a1)
	assertEqualSlices(t, a1.Data(), []float32{0, 2, 4, 6})

	a2 := tensor.Range[float32](4).Reshape(2, 2)
	b2 := tensor.Range[float32](4).Reshape(2, 2)
	a2.Add(b2, b2)
	assertEqualSlices(t, b2.Data(), []float32{0, 2, 4, 6})
}

func TestMul(t *testing.T) {
	a := tensor.CreateEmptyTensor[float32](3, 2).Fill(2)
	b := tensor.CreateEmptyTensor[float32](3, 1).Fill(3)
	ab := a.Mul(b, nil)
	assertEqualSlices(t, ab.Data(), []float32{6, 6, 6, 6, 6, 6})
	assertEqualSlices(t, ab.Shape(), types.Shape{3, 2})
}

func TestMul2(t *testing.T) {
	a := tensor.CreateEmptyTensor[int32](3, 2).Fill(2)
	b := tensor.CreateEmptyTensor[int32](3, 1).Fill(3)
	ab := a.Mul(b, nil)
	assertEqualSlices(t, ab.Data(), []int32{6, 6, 6, 6, 6, 6})
	assertEqualSlices(t, ab.Shape(), types.Shape{3, 2})
}

func TestMulToConst(t *testing.T) {
	a1 := tensor.CreateTensor([]float32{1, 2, 3, 4, 5, 6}, types.Shape{2, 3})
	a2 := tensor.Scalar[float32](2)
	out := a1.Mul(a2)
	// fmt.Println(out.Get(999, 999))
	assertEqualSlices(t, out.Data(), []float32{2, 4, 6, 8, 10, 12})
}

func TestMatMul1(t *testing.T) {
	a := tensor.Range[float32](9).Reshape(3, 3)
	b := tensor.Range[float32](9).Reshape(3, 3)
	c := a.MatMul(b)
	assertEqualSlices(t, c.Data(), []float32{15, 18, 21, 42, 54, 66, 69, 90, 111})
	assertEqualSlices(t, c.Shape(), types.Shape{3, 3})
}

func TestMatMul2(t *testing.T) {
	a1 := tensor.Range[float32](3).Reshape(1, 3)
	b1 := tensor.Range[float32](3).Reshape(3, 1)
	c1 := a1.MatMul(b1)
	assertEqualSlices(t, c1.Data(), []float32{5})
	assertEqualSlices(t, c1.Shape(), types.Shape{1, 1})
}

func TestMatMul3(t *testing.T) {
	a2 := tensor.Range[float32](8).Reshape(2, 4)
	b2 := tensor.Range[float32](8).Reshape(4, 2)
	// fmt.Println(a2.ToString(), b2.ToString())
	c2 := a2.MatMul(b2)
	assertEqualSlices(t, c2.Data(), []float32{28, 34, 76, 98})
	assertEqualSlices(t, c2.Shape(), types.Shape{2, 2})
}

func TestMatMul4Vec(t *testing.T) {
	a2 := tensor.Range[float32](10).Reshape(1, 10)
	b2 := tensor.Range[float32](10).Reshape(10, 1)
	c2 := a2.MatMul(b2)
	// fmt.Println(c2.ToString())
	assertEqualSlices(t, c2.Data(), []float32{285})
	assertEqualSlices(t, c2.Shape(), types.Shape{1, 1})
}

func TestMatMul5(t *testing.T) {
	a2 := tensor.Range[float32](12).Reshape(3, 4)
	b2 := tensor.Range[float32](8).Reshape(4, 2)
	c2 := a2.MatMul(b2)
	// fmt.Println(c2.ToString())
	assertEqualSlices(t, c2.Data(), []float32{28, 34, 76, 98, 124, 162})
	assertEqualSlices(t, c2.Shape(), types.Shape{3, 2})
}

func TestMatMul6(t *testing.T) {
	a2 := tensor.Range[float32](12).Reshape(3, 4)
	b2 := tensor.Range[float32](4).Reshape(4, 1)
	c2 := a2.MatMul(b2)
	// fmt.Println(c2.ToString())
	assertEqualSlices(t, c2.Data(), []float32{14, 38, 62})
	assertEqualSlices(t, c2.Shape(), types.Shape{3, 1})
}

func TestDot(t *testing.T) {
	a2 := tensor.Range[float32](12).Reshape(3, 2, 2)
	b2 := tensor.Range[float32](12).Reshape(3, 2, 2)
	c2 := a2.Dot(b2)
	assertEqualSlices(t, c2.Data(), []float32{2, 3, 6, 11, 46, 55, 66, 79, 154, 171, 190, 211})
	assertEqualSlices(t, c2.Shape(), types.Shape{3, 2, 2})
}

func TestDot2(t *testing.T) {
	a2 := tensor.Range[float32](18).Reshape(3, 2, 3)
	b2 := tensor.Range[float32](18).Reshape(3, 3, 2)
	c2 := a2.Dot(b2)
	assertEqualSlices(t, c2.Data(), []float32{
		10, 13, 28, 40, 172, 193, 244, 274, 550, 589, 676, 724,
	})
	assertEqualSlices(t, c2.Shape(), types.Shape{3, 2, 2})
}

func TestPow(t *testing.T) {
	a2 := tensor.Range[float32](1, 4)
	b2 := tensor.Range[float32](1, 4)
	c2 := a2.Pow(b2)
	assertEqualSlices(t, c2.Data(), []float32{1, 4, 27})
	assertEqualSlices(t, c2.Shape(), types.Shape{3})
}

func TestDiv(t *testing.T) {
	a2 := tensor.Range[float32](1, 4)
	b2 := tensor.CreateTensor[float32]([]float32{2, 2, 2}, types.Shape{3})
	c2 := a2.Div(b2)
	assertEqualSlices(t, c2.Data(), []float32{0.5, 1, 1.5})
	assertEqualSlices(t, c2.Shape(), types.Shape{3})
}

func TestDivInplace(t *testing.T) {
	a2 := tensor.Range[float32](1, 4)
	b2 := tensor.CreateTensor[float32]([]float32{2, 2, 2}, types.Shape{3})
	a2.Div(b2, a2)
	assertEqualSlices(t, a2.Data(), []float32{0.5, 1, 1.5})
	assertEqualSlices(t, a2.Shape(), types.Shape{3})
}
