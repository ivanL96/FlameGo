package main

import (
	"fmt"
	"gograd/tensor"
)

func main() {
	a := tensor.Range[int32](4).Reshape(2, 2)
	// b := a.Index(1)
	// b.Fill(6)
	// fmt.Println(b.ToString())
	fmt.Println(a.ToString())
	fmt.Println(a.Transpose().ToString())
	// a.Transpose()
	// a := tensor.Range[int32](16)
	// fmt.Println(a.Index(0).ToString())
	// fmt.Println(a.Index(1).ToString())

	// aa := tensor.AsType[int32, int64](a)
	// fmt.Printf("%v\n", a.ToString())
	// fmt.Printf("%v\n", aa.ToString())
}
