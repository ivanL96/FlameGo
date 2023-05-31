package tensor

import (
	"fmt"
	types "gograd/tensor/types"
)

func (tensor *Tensor[T]) getFlatIndex(indices ...int) int {
	flatIndex := 0
	for i, ind := range indices {
		// resolve negative indexes
		if ind < 0 {
			ind = int(tensor.shape[i]) + ind
			if ind < 0 {
				panic(fmt.Sprintf("Index %v is out of bounds", ind))
			}
		}
		// bound check
		if ind >= int(tensor.shape[i]) {
			panic(fmt.Sprintf("Index %v is out of bounds for dim %v", ind, tensor.shape[i]))
		}
		flatIndex += tensor.strides[i] * ind
	}
	return flatIndex
}

// faster Get() without bounds checking. Does not support negative indexing
func (tensor *Tensor[T]) Get_fast(indices ...int) T {
	return tensor.data[get_flat_idx_fast(tensor.strides, indices...)]
}

func get_flat_idx_fast(strides []int, indices ...int) int {
	flatIndex := 0
	idxlen := len(indices)
	switch idxlen {
	case 1:
		flatIndex = strides[0] * indices[0]
	case 2:
		flatIndex = strides[0]*indices[0] + strides[1]*indices[1]
	default:
		for i, ind := range indices {
			flatIndex += strides[i] * ind
		}
	}
	return flatIndex
}

func (tensor *Tensor[T]) Get(indices ...int) T {
	if len(indices) != len(tensor.shape) {
		panic(fmt.Sprintf(
			"Incorrect number of indices. Must be %v got %v", len(tensor.shape), len(indices)))
	}
	flatIndex := tensor.getFlatIndex(indices...)
	return tensor.data[flatIndex]
}

// returns sub data for given indices.
func (tensor *Tensor[T]) Index(indices ...int) *Tensor[T] {
	// TODO advanced indexing
	n_indices := len(indices)
	n_dims := len(tensor.shape)
	if n_indices == 0 {
		panic("At leat one index is required in Index()")
	}
	if n_indices > n_dims {
		panic("Too many indices")
	}

	// index of the first elem in the sub tensor
	flatIndex := tensor.getFlatIndex(indices...)
	if n_indices == n_dims {
		return CreateTensor([]T{tensor.data[flatIndex]}, types.Shape{1})
	}
	innerShape := tensor.shape[n_indices:]

	// continuous data
	// if data layout is continuous we can just take a slice start:end from data
	if isDimOrderInit(tensor.dim_order) {
		endFlatIndex := flatIndex + tensor.strides[n_indices-1]
		subData := tensor.data[flatIndex:endFlatIndex]
		// return CreateTensor(subData, innerShape)
		// TODO finish this. Tensor creation here should be without shape validation.
		// because data can be larger on purpose (buffer)
		return &Tensor[T]{
			data:      subData,
			shape:     innerShape,
			dim_order: initDimOrder(innerShape),
			strides:   getStrides(innerShape),
		}
	}

	// not continuous data. i.e. transposed tensor
	subShape := innerShape
	innerStrides := tensor.strides[n_indices:]
	// expand innerShape
	// TODO this is extra step, better to do something within the loop
	if len(innerShape) == 1 {
		innerShape = types.Shape{1, innerShape[0]}
		innerStrides = []int{innerStrides[0], innerStrides[0]}
	}

	// prealloc output
	var innerShapeProd types.Dim = 1
	for _, dim := range innerShape {
		innerShapeProd *= dim
	}
	subData := make([]T, innerShapeProd)
	innermostStride := tensor.strides[len(tensor.strides)-1]
	row := int(innerShape[len(innerShape)-1]) // innermost axis
	// number of dims around the 'row'. Cannot be zero
	numDims := len(innerStrides) - 2
	for i := numDims; i >= 0; i-- {
		stride := innerStrides[i]
		subDataIdx := 0
		for s := 0; s < int(innerShape[i]); s++ {
			for j := 0; j < row; j++ {
				// from innermost to outermost
				deepIndex := flatIndex + innermostStride*j + stride*s
				subData[subDataIdx] = tensor.data[deepIndex]
				subDataIdx++
			}
		}
	}
	return CreateTensor(subData, subShape)
}

// TODO GetAxis not finished
func (tensor *Tensor[T]) GetAxis(axis uint, shift uint) *Tensor[T] {
	stride := tensor.strides[axis]
	index := int(shift)
	for i := 0; i < int(tensor.shape[axis]); i++ {
		fmt.Println(tensor.data[index])
		index += stride
	}
	return tensor
}

// reorders data layout to continuous format.
// it is useful for optimizing indexing/iterating for transposed & other non-continuous tensors
func (tensor *Tensor[T]) AsContinuous(out *Tensor[T]) *Tensor[T] {
	if isDimOrderInit(tensor.dim_order) {
		return tensor
	}
	if tensor.hasFlag(SameValuesFlag) {
		return tensor
	}

	outTensor := PrepareOutTensor(out, tensor.shape)
	iter := tensor.CreateIterator()
	for iter.Iterate() {
		dataIndex := iter.Index()
		valueIndexes := iter.Next()
		val := tensor.Get_fast(valueIndexes...)
		outTensor.data[dataIndex] = val
	}
	return outTensor
}
