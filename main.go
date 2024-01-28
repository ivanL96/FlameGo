package main

import (
	"fmt"
	"gograd/grad"
	"gograd/tensor"
	"gograd/tensor/types"
)

func linear_regression() {
	var batch types.Dim = 10
	// y_true := grad.Constant[float32](tensor.Ones[float32](batch, 3))
	// X_batch = 2 * torch.rand(32, 1)
	// y_batch = 4 + 3*X_batch + 0.1*torch.randn(32, 1)
	rng := tensor.NewRNG(0)
	x := grad.Constant[float32](
		rng.RandomFloat32(batch, 1).Mul(tensor.Scalar[float32](2)),
	)
	four := tensor.Scalar[float32](4)
	three := tensor.Scalar[float32](3)
	_y := three.Mul(x.Value.Copy())
	_y = four.Add(_y)
	_y = _y.Add(rng.RandomFloat32(batch, 1))
	y := grad.Constant[float32](_y)

	optim := grad.SGD[float32](0.001)

	w := grad.Variable[float32](rng.RandomFloat32(1, 1))
	w.Alias = "W"
	b := grad.Variable[float32](rng.RandomFloat32(1))
	b.Alias = "B"
	// fmt.Println("x, y", x.Value.Shape(), y.Value.Shape())
	epochs := 1000
	for i := 0; i < epochs; i++ {
		y_pred := (x.MatMul(w)).Add(b)
		loss := y_pred.MSE(y)
		loss.Backward(nil)
		if (i+1)%100 == 0 {
			fmt.Println("Epoch", i, "loss", loss.Value.Data())
		}
		optim.Step(w, b)
		optim.ZeroGrads()
	}

	// inference
	// y_pred := (x.Mul(w)).Add(b)
	// fmt.Println(x.Value.Data())
	// fmt.Println(y.Value.Data())
	// fmt.Println(y_pred.Value.Data())
}

func mse() {
	rng := tensor.NewRNG(0)
	a := grad.Variable[float32](tensor.CreateTensor[float32](
		[]float32{3., 4., 5.}, types.Shape{1, 3}))
	b := grad.Variable[float32](rng.RandomFloat32(1))
	z := a.Mul(b)
	fmt.Println("z", z.ToString())
	z.Backward(nil)
	fmt.Println("dz/da", a.Grad.ToString())
	fmt.Println("dz/db", b.Grad.ToString())
}

func mean() {
	a := grad.Variable[float32](tensor.Range[float32](6).Reshape(2, 3))
	b := grad.Variable[float32](tensor.Range[float32](6).Reshape(3, 2))
	c := a.MatMul(b)
	z := c.Mean()
	z.Backward(nil)
	fmt.Println(z.ToString())
	fmt.Println("dz/da", a.Grad.ToString())
	fmt.Println("dz/db", b.Grad.ToString())
}

func indexing4d() {
	a := tensor.Range[int32](2*2*2*2).Reshape(2, 2, 2, 2)
	b := a.IndexAdv(":,0")
	fmt.Println(b.ToString())
	// fmt.Println(a.TrC(3, 0, 1, 2).Index(0).ToString()) // arr[:,:,:,0]
	// c := a.TrC(2, 0, 1, 3).Index(0)
	// fmt.Println(c.ToString()) // arr[:,:,0,:]
	// c = c.TrC(2, 0, 1).Index(0)
	// fmt.Println(c.ToString()) // arr[:,:,0,0]
	// c = a.TrC(1, 0, 2, 3).Index(0)
	// fmt.Println(c.ToString()) // arr[:,0,:,:]
}

func indexing5d() {
	a := tensor.Range[int32](2*2*2*2*2).Reshape(2, 2, 2, 2, 2)
	// 16,8,4,2,1
	fmt.Println(a.TrC(2, 0, 1, 3, 4).Index(0).ToString()) // arr[:,:,0,:,:]
	// 2,16,8,4,1
	fmt.Println(a.TrC(3, 0, 1, 2, 4).Index(0).ToString()) // arr[:,:,:,0,:]
}

func sum_axis() {
	a := tensor.Range[int32](2*3*4).Reshape(2, 3, 4)
	// b := a.IndexAdv(":,0")
	b := a.SumAlongAxis(0, true)
	fmt.Println(b.ToString())
	b = a.SumAlongAxis(1, true)
	fmt.Println(b.ToString())
}

func main() {
	// fmt.Println(addLeftPadding([]int{2, 3, 4, 5}, 2, 1))
	// linear_regression()
	// mse()
	// mean()
	// indexing()
	// sum_axis()
	// indexing4d()
	// indexing5d()
}
