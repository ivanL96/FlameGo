package main

import (
	"fmt"
	"gograd/grad"
	"gograd/tensor"
	types "gograd/tensor/types"
	"math"
	"testing"
)

func TestLinear(t *testing.T) {
	rng := tensor.NewRNG(0)

	var batch types.Dim = 10
	// y_true := grad.Constant[float32](tensor.Ones[float32](batch, 3))
	// X_batch = 2 * torch.rand(32, 1)
	// y_batch = 4 + 3*X_batch + 0.1*torch.randn(32, 1)
	x := grad.Constant(
		rng.RandomFloat32(batch, 1).Mul(tensor.Scalar[float32](2)),
	)
	four := tensor.Scalar[float32](4)
	three := tensor.Scalar[float32](3)
	_y := three.Mul(x.Value.Copy())
	_y = four.Add(_y)
	_y = _y.Add(rng.RandomFloat32(batch, 1))
	y := grad.Constant(_y)

	optim := grad.SGD[float32](0.001)

	w := grad.Variable(rng.RandomFloat32(1, 1))
	w.Alias = "W"
	b := grad.Variable(rng.RandomFloat32(1))
	b.Alias = "B"

	epochs := 1000
	var history float32
	for i := 0; i < epochs; i++ {
		y_pred := (x.MatMul(w)).Add(b)
		loss := y_pred.MSE(y).MustAssert()
		loss.Backward(nil)
		if (i+1)%100 == 0 {
			history = loss.Value.Item()
		}
		optim.Step(w, b)
		optim.ZeroGrads()
	}

	fmt.Println(history, 0.0361432)
	assert(t, math.Abs(float64(history-0.0361432)) <= 0.0001)
}
