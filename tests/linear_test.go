package main

import (
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
	history := make([]float32, 0)
	for i := 0; i < epochs; i++ {
		y_pred := (x.MatMul(w)).Add(b)
		loss := y_pred.MSE(y)
		loss.Backward(nil)
		if (i+1)%100 == 0 {
			history = append(history, loss.Value.Data()...)
		}
		optim.Step(w, b)
		optim.ZeroGrads()
	}
	loss_history := []float32{
		14.10375,
		6.4046683,
		2.919373,
		1.3408439,
		0.6251891,
		0.30006462,
		0.15173683,
		0.08349221,
		0.051564693,
		0.0361432}

	for i := 0; i < len(history); i++ {
		a := history[i]
		b := loss_history[i]
		assert(t, math.Abs(float64(a-b)) <= 0.01)
	}
}
