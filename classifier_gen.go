package main

import (
	"fmt"
	"gograd/datasets"
	"gograd/grad"
	"gograd/tensor"
	"gograd/tensor/types"
	"time"
)

func classifier_gen() {

	rng := tensor.NewRNG(69)

	// x, y, xtest, ytest := datasets.LoadIris("datasets/iris.csv", rng, 0.2)
	f := func(a, b float32) float32 {
		if a == b {
			return 1
		}
		return 0
	}
	n := 1000
	dset := datasets.GenerateFunction(f, 0, 10, n, rng.Seed)
	x, y, xtest, ytest := dset.Split(0.2)
	fmt.Println(len(x), len(y))

	X := grad.Constant(tensor.CreateTensor(x, types.Shape{types.Dim(800), 2})).MustAssert()
	Y := grad.Constant(tensor.CreateTensor(y, types.Shape{types.Dim(800), 1})).MustAssert()
	fmt.Println("x", X.Value.Shape())
	fmt.Println("y", Y.Value.Shape())

	inner := types.Dim(10)
	classes := types.Dim(2)
	W1 := grad.Variable(rng.RandomFloat32(2, inner))
	B1 := grad.Variable(rng.RandomFloat32(1, inner))
	W2 := grad.Variable(rng.RandomFloat32(inner, classes))
	B2 := grad.Variable(rng.RandomFloat32(1, classes))

	optim := grad.SGD[float32](0.01)
	start := time.Now()
	epochs := 1000

	for i := 0; i < epochs; i++ {
		logits := model(X, W1, B1, W2, B2)
		loss := logits.SoftmaxCrossEntropy(Y).Mean().MustAssert()
		loss.Backward(nil)
		if i%100 == 0 {
			fmt.Println(i, loss.Value.Item())
		}
		optim.Step(W1, B1, W2, B2)
		optim.ZeroGrads()
	}
	ela := time.Since(start)
	fmt.Println("ela", ela)

	Xtest := tensor.CreateTensor(xtest, types.Shape{200, 2}).MustAssert()
	Ytest := tensor.CreateTensor(ytest, types.Shape{200, 1}).MustAssert()
	var correct float32 = 0
	for i := 0; i < 30; i++ {
		x := grad.Constant(Xtest.Index(i).Reshape(1, 2))
		y := grad.Constant(Ytest.Index(i).Reshape(1, 1))
		pred := model(x, W1, B1, W2, B2).Value.Softmax(nil).MustAssert()

		fmt.Println("pred", pred.ToString(), "true", y.Value.Item())
		argmax, _ := pred.Find(pred.Max(false).Item())
		if argmax[1] == int(y.Value.Item()) {
			correct += 1
		}
	}
	fmt.Println("inference", correct/30)
}
