package main

import (
	"gograd/grad"
	"gograd/tensor"
	types "gograd/tensor/types"
	"testing"
)

func TestToOneHot(t *testing.T) {

	a := tensor.CreateTensor[int](
		[]int{1, 2, 0, 0, 1, 2, 1}, types.Shape{7},
	)
	oneh := grad.ToOneHot[int](a, 3)
	// fmt.Println(oneh.ToString())
	// onehf := tensor.AsType[int, float32](oneh)
	// fmt.Println(onehf.ToString())
	assertEqualSlices(t, oneh.Shape(), types.Shape{7, 3})
	assertEqualSlices(t, oneh.Data(), []int{
		0, 1, 0,
		0, 0, 1,
		1, 0, 0,
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
		0, 1, 0,
	})
}
