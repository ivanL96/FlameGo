package main

import (
	"fmt"
	"gograd/tensor"
	"testing"
)

func TestAdd(t *testing.T) {
	a := tensor.InitEmptyTensor[int32](3, 2).Fill(2)
	b := tensor.InitEmptyTensor[int32](3, 1).Fill(3)
	assertEqualSlices(t, a.Add(b).Data(), []int32{5, 5, 5, 5, 5, 5})

	c := tensor.InitEmptyTensor[int32](1, 2).Fill(2)
	d := tensor.InitEmptyTensor[int32](3, 1).Fill(3)
	fmt.Println(c.Add(d).Shape())
	assertEqualSlices(t, c.Add(d).Data(), []int32{5, 5, 5, 5, 5, 5})
}
