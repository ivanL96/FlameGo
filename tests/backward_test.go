package main

import (
	"gograd/grad"
	"gograd/tensor"
	types "gograd/tensor/types"
	"math"
	"testing"
)

// go test -run TestNewPerson -v
// go test '-run=^TestMyTest$'
func TestGradAdd(t *testing.T) {
	a := grad.Variable[float32](tensor.CreateTensor[float32](
		[]float32{4}, types.Shape{1, 1}))
	b := grad.Variable[float32](tensor.CreateTensor[float32](
		[]float32{5}, types.Shape{1, 1}))
	z := a.Add(b)
	assertEqualSlices(t, z.Value.Data(), []float32{9})
	z.Backward(nil)
	assertEqualSlices(t, a.Grad.Data(), []float32{1})
	assertEqualSlices(t, b.Grad.Data(), []float32{1})

	_add := func(x float64) float64 {
		return x + 5
	}
	deriv := grad.NumericDeriv(grad.EPSILON, 4, _add)
	assert(t, math.Abs(1-deriv) <= 0.0001)
}

func TestGradSub(t *testing.T) {
	a := grad.Variable[float32](tensor.CreateTensor[float32](
		[]float32{4}, types.Shape{1, 1}))
	b := grad.Variable[float32](tensor.CreateTensor[float32](
		[]float32{5}, types.Shape{1, 1}))
	z := a.Sub(b)
	assertEqualSlices(t, z.Value.Data(), []float32{-1})
	z.Backward(nil)
	assertEqualSlices(t, a.Grad.Data(), []float32{1})
	assertEqualSlices(t, b.Grad.Data(), []float32{-1})

	_sub := func(x float64) float64 {
		return x - 5
	}
	deriv := grad.NumericDeriv(grad.EPSILON, 4, _sub)
	assert(t, math.Abs(1-deriv) <= 0.0001)
}

func TestGradMul(t *testing.T) {
	a := grad.Variable[float32](tensor.CreateTensor[float32](
		[]float32{4}, types.Shape{1, 1}))
	b := grad.Variable[float32](tensor.CreateTensor[float32](
		[]float32{5}, types.Shape{1, 1}))
	z := a.Mul(b)
	assertEqualSlices(t, z.Value.Data(), []float32{20})
	z.Backward(nil)
	assertEqualSlices(t, a.Grad.Data(), []float32{5})
	assertEqualSlices(t, b.Grad.Data(), []float32{4})

	_mul := func(x float64) float64 {
		return x * 5
	}
	deriv := grad.NumericDeriv(grad.EPSILON, 4, _mul)
	assert(t, math.Abs(5-deriv) <= 0.0001)

	_mul2 := func(x float64) float64 {
		return x * 4
	}
	deriv = grad.NumericDeriv(grad.EPSILON, 5, _mul2)
	assert(t, math.Abs(4-deriv) <= 0.0001)
}
