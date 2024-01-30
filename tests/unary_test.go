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
