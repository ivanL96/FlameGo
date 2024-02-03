package main

import (
	"gograd/tensor"
	types "gograd/tensor/types"
	"testing"
)

func TestSigmoid(t *testing.T) {
	rng := tensor.NewRNG(0)
	a := rng.RandomFloat32(2, 3)
	s := a.Sigmoid()

	data := []float32{0.7201481, 0.56093687, 0.6583514, 0.5135826, 0.59087586, 0.57186896}
	assertEqualSlices(t, s.Data(), data)
	assertEqualSlices(t, s.Shape(), types.Shape{2, 3})
}

func TestNeg(t *testing.T) {
	a := tensor.Range[float32](10)
	s := a.Neg()
	assertEqualSlices(t, s.Data(), []float32{0, -1, -2, -3, -4, -5, -6, -7, -8, -9})
	assertEqualSlices(t, s.Shape(), types.Shape{10})
}

func TestRelu(t *testing.T) {
	a := tensor.Range[float32](-3, 7)
	s := a.Relu()
	assertEqualSlices(t, s.Data(), []float32{0, 0, 0, 0, 1, 2, 3, 4, 5, 6})
	assertEqualSlices(t, s.Shape(), types.Shape{10})
}

func TestApplyFunc1(t *testing.T) {
	a := tensor.Range[float32](6).Reshape(2, 3)
	expr := func(a float32) float32 {
		return -a
	}
	s := a.ApplyFunc(expr)
	assertEqualSlices(t, s.Shape(), types.Shape{2, 3})
	assertEqualSlices(t, s.Data(), []float32{0, -1, -2, -3, -4, -5})
}

func TestApplyFunc2(t *testing.T) {
	a := tensor.Range[float32](-3, 3).Reshape(2, 3)
	expr := func(a float32) float32 {
		if a > 0 {
			return 1
		}
		return 0
	}
	s := a.ApplyFunc(expr)
	assertEqualSlices(t, s.Shape(), types.Shape{2, 3})
	assertEqualSlices(t, s.Data(), []float32{0, 0, 0, 0, 1, 1})
}

func TestExp(t *testing.T) {
	a := tensor.Range[float32](10)
	s := a.Exp()
	assertEqualSlices(t, s.Data(), []float32{1.0, 2.71828183, 7.38905610, 2.00855369e+01,
		5.45981500e+01, 1.48413159e+02, 4.03428793e+02, 1.09663316e+03,
		2.98095799e+03, 8.10308393e+03})
	assertEqualSlices(t, s.Shape(), types.Shape{10})
	a.MustAssert()
	s.MustAssert()
}
