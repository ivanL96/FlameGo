package tensor

import "fmt"

func (tensor *Tensor[T]) Get(indices ...int) T {
	if len(indices) != len(tensor.shape) {
		panic(fmt.Sprintf(
			"Incorrect number of indices. Must be %v got %v", len(tensor.shape), len(indices)))
	}
	flatIndex := 0
	for i, ind := range indices {
		flatIndex += tensor.strides[i] * ind
	}
	return tensor.data[flatIndex]
}

// returns sub data for given indices.
func (tensor *Tensor[T]) Index(indices ...int) *Tensor[T] {
	n_indices := len(indices)
	n_dims := len(tensor.shape)
	if n_indices == 0 {
		panic("At leat one index is required in View")
	}
	if n_indices > n_dims {
		panic("Too many indices")
	}

	newShape := make(Shape, n_dims-n_indices)
	copy(newShape, tensor.shape[n_indices:])

	flatIndex := 0 // index of the first elem in the sub tensor
	for i, ind := range indices {
		// resolve negative indexes
		if ind < 0 {
			ind = int(tensor.shape[i]) + ind
		}
		if ind < 0 {
			panic(fmt.Sprintf("Index %v is out of bounds", ind))
		}
		flatIndex += tensor.strides[i] * ind
	}
	// fmt.Println("flatIndex", flatIndex, tensor.data[flatIndex])
	if n_indices == n_dims {
		return InitTensor([]T{tensor.data[flatIndex]}, Shape{1})
	} else {
		// TODO add case for contiguous data: dim_order 0,1,2,3...

		innerShape := tensor.shape[len(indices):]
		// fmt.Println("innerShape", innerShape)
		var innerShapeProd Dim = 1
		for _, dim := range innerShape {
			innerShapeProd *= dim
		}

		subData := make([]T, innerShapeProd)

		row := int(innerShape[len(innerShape)-1]) // deepest axis
		rowStride := tensor.strides[len(tensor.strides)-1]
		for j := 0; j < int(innerShapeProd); j++ {
			rem := (j / row) * row
			count := j % row
			deepIndex := (flatIndex + rem) + rowStride*count
			subData[j] = tensor.data[deepIndex]
		}
		return InitTensor(subData, innerShape)
	}
}
