package tensor

import (
	"fmt"
	types "gograd/tensor/types"
)

func PrepareOutTensor[T types.TensorType](out *Tensor[T], shape types.Shape) *Tensor[T] {
	if out == nil {
		return CreateEmptyTensor[T](shape...)
	}
	// check if 'out' tensor has shape less than required.
	// however having bigger shape is OK since the 'out' tensor can serve as a buffer for different outputs
	if len(out.shape) != len(shape) {
		panic(fmt.Sprintf("Output tensor has different number of dims. Required %v, but got %v", shape, out.shape))
	}
	for i, dim := range shape {
		if dim > out.shape[i] {
			panic(fmt.Sprintf("Output tensor dim %v is less than required dim %v", out.shape[i], dim))
		}
	}
	return out
}
