package main

import (
	"fmt"
	"gograd/grad"
	"gograd/tensor"
	"gograd/tensor/types"
)

func TestMatMul2() {
	a1 := tensor.Range[float32](3).Reshape(1, 3).Add(tensor.Scalar[float32](1))
	b1 := tensor.Range[float32](3).Reshape(3, 1).Add(tensor.Scalar[float32](1))
	a1.MatMul(b1)

}

func lin() {
	var batch types.Dim = 32
	// y_true := grad.Constant[float32](tensor.Ones[float32](batch, 3))
	// X_batch = 2 * torch.rand(32, 1)
	// y_batch = 4 + 3*X_batch + 0.1*torch.randn(32, 1)

	x := grad.Constant[float32](
		tensor.RandomFloat32(types.Shape{batch, 1}, 0).Mul(tensor.Scalar[float32](2)),
	)
	y_batch := grad.Constant[float32](
		tensor.Scalar[float32](4).Add(tensor.Scalar[float32](3).Mul(x.Value)).Add(tensor.Scalar[float32](0.1).Mul(tensor.RandomFloat32(types.Shape{batch, 1}, 0))),
	)
	w := grad.Variable[float32](tensor.RandomFloat32(types.Shape{1, 1}, 0))
	b := grad.Variable[float32](tensor.RandomFloat32(types.Shape{1}, 0))
	fmt.Println("x, y", x.Value.Shape(), y_batch.Value.Shape())
	y_pred := x.MatMul(w).Add(b) //.Sigmoid()
	loss := y_pred.MSE(y_batch)
	loss.Backward(nil)
	// fmt.Println("y_pred shape", y_pred.Value.Shape())
	// fmt.Println("y_pred", y_pred.Grad)
	// fmt.Println("loss", loss.ToString())
	fmt.Println("dL/dw", w.Grad.ToString())
	fmt.Println("dL/db", b.Grad.Shape())
}

func mse() {
	a := grad.Variable[float32](tensor.CreateTensor[float32](
		[]float32{3., 4., 5.}, types.Shape{1, 3}))
	b := grad.Variable[float32](tensor.RandomFloat32(types.Shape{1}, 0))
	// c := grad.Constant[float32](tensor.CreateTensor[float32](
	// 	[]float32{2., 1., 1.}, types.Shape{1, 3}))
	z := a.Mul(b)
	fmt.Println("z", z.ToString())
	z.Backward(nil)
	fmt.Println("dz/da", a.Grad.ToString())
	fmt.Println("dz/db", b.Grad.ToString())
}

func mean() {
	// a := grad.Variable[float32](tensor.CreateTensor[float32](
	// 	[]float32{3., 4., 5.}, types.Shape{1, 3}))
	// b := grad.Variable[float32](tensor.CreateTensor[float32](
	// 	[]float32{3.}, types.Shape{1}))

	a := grad.Variable[float32](tensor.CreateTensor[float32](
		[]float32{3, 4, 5, 6, 7, 8}, types.Shape{2, 3}))
	b := grad.Variable[float32](tensor.CreateTensor[float32](
		[]float32{3, 2, 1}, types.Shape{1, 3}))

	c := a.Mul(b)
	z := c.Mean()
	z.Backward(nil)
	fmt.Println(z.ToString())
	fmt.Println(a.ToString())
	fmt.Println("dz/da", a.Grad.ToString())
	fmt.Println("dz/db", b.Grad.ToString())
}

func indexing() {
	a := tensor.CreateTensor[int32](
		[]int32{1, 10, 2, 20, 3, 30, 4, 40, 5, 50, 6, 60},
		types.Shape{2, 3, 2})
	fmt.Println(a.Index(0).ToString())             // numpy: arr[0,:,:]
	b := a.Transpose().Index(0).Transpose()        //.AsContinuous(nil)
	fmt.Println(b.ToString())                      // arr[:,:,0]
	c := a.Transpose(1, 2, 0).Index(0).Transpose() // arr[:,0,:]
	fmt.Println(c.ToString())
}

func indexing4d() {
	a := tensor.Range[int32](2*2*2*2).Reshape(2, 2, 2, 2)
	fmt.Println(a.Index(0).ToString()) // numpy: arr[0,:,:]
	b := a.Transpose().AsContinuous(nil).Index(0)
	// .Transpose() //.AsContinuous(nil)
	fmt.Println(b.ToString()) // arr[:,:,0]
	// c := a.Transpose(1, 2, 0).Index(0).Transpose() // arr[:,0,:]
	// fmt.Println(c.ToString())
}
func main() {
	// lin()
	// mse()
	// mean()
	indexing4d()
}
