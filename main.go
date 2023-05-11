package main

import (
	// "gograd/tests"
	// "fmt"
	"gograd/tensor"
)

func main() {
	// tests.TestAdd()

	a := tensor.InitTensor[int32](3, 2)
	a.Set([]int32{0, 1, 2, 3, 4, 5})
	a.Index(1, 1)
	// b := a.Broadcast(tensor.Shape{3, 2})
	// fmt.Println(b.ToString())
}
