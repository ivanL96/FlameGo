package main

import (
	"gograd/tensor"
	types "gograd/tensor/types"
	"testing"
)

func TestError(t *testing.T) {
	a2 := tensor.Range[float32](1, 4)
	b2 := tensor.CreateTensor[float32]([]float32{2, 2, 2}, types.Shape{4}) // incorrect shape
	a2.Div(b2, a2).T().Copy().Unsqueeze(0).Fill(4).Sum(false)
	assert(t, a2.Err == nil)
	assert(t, b2.Err != nil) // should generate an error
}

func TestGetError(t *testing.T) {
	a2 := tensor.Range[float32](1, 4)
	data, err := a2.Get(666)
	assert(t, err != nil)
	assert(t, data == 0)
}
