package iter

import "gograd/tensor/types"

// tensor iterator

type TensorIterator[T types.TensorType] struct {
	shape          types.Shape
	currentIndexes []int
	dataLen        int
	index          int
}

// iterates over tensor data and gives N-dim index for each element
func CreateIterator[T types.TensorType](data []T, shape types.Shape) *TensorIterator[T] {
	ti := TensorIterator[T]{
		dataLen: len(data),
		shape:   shape,
		// currentIndexes will not be copied for each Next() call.
		currentIndexes: make([]int, len(shape)),
		index:          0,
	}
	return &ti
}

func (ti *TensorIterator[T]) Index() int {
	return ti.index
}

func (ti *TensorIterator[T]) Iterate() bool {
	return ti.index != ti.dataLen
}

func (ti *TensorIterator[T]) Next() []int {
	if ti.index == 0 {
		ti.index++
		return ti.currentIndexes
	}

	indexes := ti.currentIndexes
	shape := ti.shape
	for j := len(indexes) - 1; j >= 0; j-- {
		indexes[j]++
		if indexes[j] < int(shape[j]) {
			break
		}
		indexes[j] = 0
	}
	ti.currentIndexes = indexes
	ti.index++
	return ti.currentIndexes
}
