package main

import "gograd/tensor"

func main() {
	// tests.TestBroadcast()
	// tests.TestFlatten()

	a := tensor.InitTensor(
		[]int32{0, 1, 2, 3, 4, 5, 6, 7, 8},
		tensor.Shape{3, 3, 1})
	a.Index(2, 2, 0)

	// b := a.Broadcast(tensor.Shape{3, 2})

	// aa := tensor.AsType[int32, int64](a)
	// fmt.Printf("%v\n", a.ToString())
	// fmt.Printf("%v\n", aa.ToString())

	// flat := tensor.InitTensor[int16](3, 1, 1, 1, 1)
	// flat.Set([]int16{1, 2, 3})
	// flat.Flatten()

	// fmt.Println(flat.ToString())
}
