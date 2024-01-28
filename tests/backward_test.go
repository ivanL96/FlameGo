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

func TestGradMatMul(t *testing.T) {
	a := grad.Variable[float32](tensor.Range[float32](5).Reshape(1, 5))
	b := grad.Variable[float32](tensor.Range[float32](5).Reshape(5, 1))
	z := a.MatMul(b)
	z.Backward(nil)
	assertEqualSlices(t, z.Value.Shape(), types.Shape{1, 1})
	assertEqualSlices(t, z.Value.Data(), []float32{30})
	assertEqualSlices(t, a.Grad.Data(), []float32{0, 1, 2, 3, 4})
	assertEqualSlices(t, a.Grad.Shape(), types.Shape{1, 5})
	assertEqualSlices(t, b.Grad.Data(), []float32{0, 1, 2, 3, 4})
	assertEqualSlices(t, b.Grad.Shape(), types.Shape{5, 1})
}

func TestGradMatMulMean(t *testing.T) {
	a := grad.Variable[float32](tensor.Range[float32](8).Reshape(4, 2))
	b := grad.Variable[float32](tensor.Range[float32](10).Reshape(2, 5))
	z := a.MatMul(b).Mean()
	z.Backward(nil)
	assertEqualSlices(t, z.Value.Shape(), types.Shape{1})
	assertEqualSlices(t, z.Value.Data(), []float32{34})
	assertEqualSlices(t, a.Grad.Data(), []float32{
		0.5, 1.75, 0.50, 1.75, 0.5, 1.75, 0.5, 1.75})
	assertEqualSlices(t, a.Grad.Shape(), types.Shape{4, 2})
	assert(t, b.Grad.IsAllClose(
		tensor.CreateTensor[float32]([]float32{0.6, 0.6, 0.6, 0.6, 0.6, 0.8, 0.8, 0.8, 0.8, 0.8}, types.Shape{2, 5}),
		0.00001,
	))
	assertEqualSlices(t, b.Grad.Shape(), types.Shape{2, 5})
}

func TestGradMatMulMSE(t *testing.T) {
	x := grad.Constant[float32](tensor.Range[float32](10).Div(tensor.Scalar[float32](10)).Reshape(10, 1))
	y := grad.Constant[float32](tensor.Range[float32](10).Reshape(10, 1))
	w := grad.Variable[float32](tensor.Range[float32](1).Reshape(1, 1))
	b := grad.Variable[float32](tensor.Range[float32](1).Reshape(1, 1))
	yhat := x.MatMul(w).Add(b)
	loss := yhat.MSE(y)
	loss.Backward(nil)
	assertEqualSlices(t, loss.Value.Data(), []float32{28.5})
	assertEqualSlices(t, w.Grad.Data(), []float32{-5.7})
	assertEqualSlices(t, b.Grad.Data(), []float32{-9})
}
