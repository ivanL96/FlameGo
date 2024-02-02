package tensor

import (
	"fmt"
	types "gograd/tensor/types"
)

func PrepareOutTensor[T types.TensorType](out *Tensor[T], shape types.Shape) (*Tensor[T], error) {
	if out == nil {
		return CreateEmptyTensor[T](shape...), nil
	}
	// check if 'out' tensor has shape less than required.
	if len(out.shape) != len(shape) {
		return nil, fmt.Errorf("output tensor has different number of dims. Required %v, but got %v", shape, out.shape)
	}
	for i, dim := range shape {
		if dim != out.shape[i] {
			return nil, fmt.Errorf("output tensor dim %v must be equal to required dim %v", out.shape[i], dim)
		}
	}
	return out, nil
}
