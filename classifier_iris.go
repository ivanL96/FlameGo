package main

import (
	"fmt"
	"gograd/datasets"
	"gograd/grad"
	"gograd/tensor"
	"gograd/tensor/types"
	"time"
)

func model[T types.TensorType](X, W1, B1, W2, B2 *grad.Var[T]) *grad.Var[T] {
	OUT1 := X.MatMul(W1).Add(B1).Relu()
	OUT2 := OUT1.MatMul(W2).Add(B2)
	return OUT2
}

func classifier_iris() {
	rng := tensor.NewRNG(69)

	split := float32(0.5)
	x, y, xtest, ytest := datasets.LoadIris("datasets/iris.csv").Shuffle(rng).Split(split)

	train_size := types.Dim(150 * (1 - split))
	test_size := types.Dim(150 * split)

	features := types.Dim(4)
	classes := types.Dim(3)
	inner := types.Dim(10)

	X := grad.Constant(tensor.CreateTensor(x, types.Shape{train_size, features})).MustAssert()
	Y := grad.Constant(tensor.CreateTensor(y, types.Shape{train_size, 1})).MustAssert()
	fmt.Println("x", X.Value.Shape())
	fmt.Println("y", Y.Value.Shape())

	W1 := grad.Variable(rng.RandomFloat32(features, inner))
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
			// Xtest := grad.Constant(tensor.CreateTensor(xtest, types.Shape{30, 4})).MustAssert()
			// Ytest := grad.Constant(tensor.CreateTensor(ytest, types.Shape{30, 1})).MustAssert()
			// infloss := model(Xtest, W1, B1, W2, B2, Ytest)
			// fmt.Println("inference", infloss.Value.Item())
			fmt.Println(i, loss.Value.Item())
		}
		optim.Step(W1, B1, W2, B2)
		optim.ZeroGrads()
	}
	ela := time.Since(start)
	fmt.Println("ela", ela)
	// 900 0.11521079
	// ela 3.489887s
	// better sum()
	// 900 0.11521079
	// ela 1.8048005s

	// 900 0.18752326
	// ela 1.3845107s
	// inference 0.96666664

	Xtest := tensor.CreateTensor(xtest, types.Shape{test_size, features}).MustAssert()
	Ytest := tensor.CreateTensor(ytest, types.Shape{test_size, 1}).MustAssert()
	var correct float32 = 0
	for i := 0; i < int(test_size); i++ {
		x := grad.Constant(Xtest.Index(i).Reshape(1, features))
		y := grad.Constant(Ytest.Index(i).Reshape(1, 1))
		pred := model(x, W1, B1, W2, B2).Value.Softmax(nil).MustAssert()

		argmax, _ := pred.Find(pred.Max(false).Item())
		if argmax[1] == int(y.Value.Item()) {
			fmt.Println(i, argmax[1], y.Value.Item())
			correct += 1
		} else {
			fmt.Println(i, argmax[1], y.Value.Item(), "mismatch")
		}
	}
	fmt.Println("inference: corrects", correct, "of", test_size)
}
