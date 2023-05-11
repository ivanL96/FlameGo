package tests

import (
	"fmt"
	"gograd/tensor"
)

func TestAdd() {
	a := tensor.InitTensor[int32](3, 2)
	a.FillTensor(2)
	b := tensor.InitTensor[int32](3, 1)
	b.FillTensor(3)

	c := a.Add(b)
	fmt.Println(c.ToString())
}
