package main

import (
	"fmt"
	"gograd/grad"
	"gograd/tensor"
	"gograd/tensor/types"
)

func main() {
	// a := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	// b := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	// var c float32
	// result_noasm := noasm.Dot(a, b, c)
	// result := cpu.AVX.Dot(a, b)
	// fmt.Println("asm", result_noasm, result)

	// 4x4:
	// 	# [[56, 62, 68, 74]
	// #  [152, 174, 196, 218]
	// #  [248, 286, 324, 362]
	// #  [344, 398, 452, 506]],

	// var dim types.Dim = 10
	// ten := tensor.Scalar[float32](10)
	// aa := tensor.Range[float32](int(dim*dim)).Reshape(dim, dim).Div(ten)
	// bb := tensor.Range[float32](int(dim*dim)).Reshape(dim, dim).Div(ten)
	// fmt.Println(aa.ToString())
	// fmt.Println(bb.ToString())
	// cc := aa.MatMul(bb)
	// res := tensor.AsType[float32, int64](cc)
	// fmt.Println(cc.ToString())
	a := grad.Variable[float32](tensor.CreateTensor[float32](
		[]float32{4, 5, 6}, types.Shape{1, 3}))
	// []float32{4, 5, 6}, types.Shape{3, 1}))
	b := grad.Variable[float32](tensor.CreateTensor[float32](
		[]float32{1, 2, 3}, types.Shape{3, 1}))
	// b.Requires_grad = false
	z := a.MatMul(b)
	fmt.Println("z", z.ToString())
	z.Backward(nil)
	fmt.Println("dz/da", a.Grad.ToString())
	fmt.Println("dz/db", b.Grad.ToString())
}
