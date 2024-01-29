package iter

import (
	"gograd/tensor/types"
)

// iterates a tensor and for each step returns an N-dim index.
//
// Example:
//
// it := tensor.CreateIterator() // or
//
// it = iter.CreateIterator(len(tensor.data()), tensor.shape)
//
//	for it.Iterate(){
//		flat_index := it.Index() // to get iterator index. Note: the index is incremented after the Next() is invoked
//		idx := it.Next()  // Required to update the iterator state.
//	 //Returns the index to access internal data layout. Copy before changing its values!
//		tensor.Index(idx...) // can be used here
//	}
type TensorIterator struct {
	shape          types.Shape
	currentIndexes []int
	dataLen        int
	index          int
}

// iterates over tensor data and gives N-dim index for each element
func CreateIterator(dataLen int, shape types.Shape) *TensorIterator {
	ti := TensorIterator{
		dataLen: dataLen,
		shape:   shape,
		// currentIndexes will not be copied for each Next() call.
		currentIndexes: make([]int, len(shape)),
		index:          0,
	}
	return &ti
}

func (ti *TensorIterator) Index() int {
	return ti.index
}

func (ti *TensorIterator) Iterate() bool {
	return ti.index < ti.dataLen
}

func (ti *TensorIterator) Next() []int {
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
