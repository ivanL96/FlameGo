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
