package main

import (
	"flamego/tensor"
	"fmt"
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
	var dim types.Dim = 10
	ten := tensor.Scalar[float32](10)
	aa := tensor.Range[float32](int(dim*dim)).Reshape(dim, dim).Div(ten, nil)
	bb := tensor.Range[float32](int(dim*dim)).Reshape(dim, dim).Div(ten, nil)
	fmt.Println(aa.ToString())
	fmt.Println(bb.ToString())
	cc := aa.MatMul(bb)
	// res := tensor.AsType[float32, int64](cc)
	fmt.Println(cc.ToString())
}
