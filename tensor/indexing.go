package tensor

import (
	types "flamego/tensor/types"
	"fmt"
)

// func (tensor *Tensor[T]) unflatIndex(flatIdx int) []int {
// TODO finish
// 	ndims := len(tensor.strides)
// 	unflatIdx := make([]int, ndims)
// 	if flatIdx == 0 {
// 		return unflatIdx
// 	}
// 	i := 0
// 	for true {
// 		stride := tensor.strides[i]
// 		if stride < flatIdx {
// 			flatIdx -= stride
// 			unflatIdx[i]++
// 		}
// 	}
// }

func (tensor *Tensor[T]) getFlatIndex(indices ...int) int {
	flatIndex := 0
	for i, ind := range indices {
		dim := int(tensor.shape[i])
		// resolve negative indexes
		if ind < 0 {
			norm_ind := dim + ind
			if norm_ind < 0 {
				panic(fmt.Sprintf("Index %v is out of bounds", ind))
			}
			ind = norm_ind
		}
		// bound check
		if ind >= dim {
			panic(fmt.Sprintf("Index %v is out of bounds for dim %v", ind, dim))
		}
		flatIndex += tensor.strides[i] * ind
	}
	return flatIndex
}

func get_flat_idx_fast(strides []int, indices ...int) int {
	idxlen := len(indices)
	switch idxlen {
	case 1:
		return strides[0] * indices[0]
	case 2:
		return strides[0]*indices[0] + strides[1]*indices[1]
	default:
		flatIndex := 0
		for i, ind := range indices {
			flatIndex += strides[i] * ind
		}
		return flatIndex
	}
}

// faster Get() without bounds checking. Does not support negative indexing
func (tensor *Tensor[T]) Get_fast(indices ...int) T {
	return tensor.data()[get_flat_idx_fast(tensor.strides, indices...)]
}

func (tensor *Tensor[T]) Get(indices ...int) T {
	if len(indices) != len(tensor.shape) {
		panic(fmt.Sprintf(
			"Incorrect number of indices. Must be %v got %v", len(tensor.shape), len(indices)))
	}
	flatIndex := tensor.getFlatIndex(indices...)
	return tensor.data()[flatIndex]
}

// returns sub data for given indices.
func (tensor *Tensor[T]) Index(indices ...int) *Tensor[T] {
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
		return Scalar[T](tensor.data()[flatIndex])
	}
	innerShape := tensor.shape[n_indices:]

	// continuous data
	// if data layout is continuous we can just take a slice start:end from data
	if isDimOrderInit(tensor.dim_order) {
		endFlatIndex := flatIndex + tensor.strides[n_indices-1]
		subData := tensor.data()[flatIndex:endFlatIndex]
		return CreateTensor(subData, innerShape)
		// TODO finish this. Tensor creation here should be without shape validation.
		// because data can be larger on purpose (buffer)
		// return &Tensor[T]{
		// 	data:      subData,
		// 	shape:     innerShape,
		// 	dim_order: initDimOrder(innerShape),
		// 	strides:   getStrides(innerShape),
		// }
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
	origData := tensor.data()
	innermostStride := tensor.strides[len(tensor.strides)-1]
	row := int(innerShape[len(innerShape)-1]) // innermost axis
	// number of dims around the 'row'. Cannot be zero
	numDims := len(innerStrides) - 2
	for i := numDims; i >= 0; i-- {
		stride := innerStrides[i]
		subDataIdx := 0
		for j := 0; j < int(innerShape[i]); j++ {
			for k := 0; k < row; k++ {
				// from innermost to outermost
				deepIndex := flatIndex + innermostStride*k + stride*j
				subData[subDataIdx] = origData[deepIndex]
				subDataIdx++
			}
		}
	}
	return CreateTensor(subData, subShape)
}

// IdxRange is used to create a slice along specific axis.
// Setting start & end is needed to apply slicing boundaries.

type IdxRange struct {
	start int
	end   int
}

// Idx() is an utility function is used for taking a specific index
func I(val int) *IdxRange {
	return &IdxRange{val, val}
}

// Axis() is used for taking entire axis-wide slice
func Axis() *IdxRange {
	return &IdxRange{0, -1}
}

func ISlc(start, end uint) *IdxRange {
	return &IdxRange{int(start), int(end)}
}

// Advanced indexing allows to specify index ranges.
//
// Example: with given tensor:
//
//	[[1,2,3],
//	[4,5,6]]
//
// should return
// tensor.IndexAdv(Axis(), I(0)) ==> [1,4]
func (tensor *Tensor[T]) IndexAdv(indices ...*IdxRange) *Tensor[T] {
	// TODO advanced indexing
	if len(indices) == 0 {
		panic("At least one index is required")
	}
	if len(indices) > len(tensor.shape) {
		panic("Too many indices")
	}
	are_constants := true
	are_axis_wide := true
	for i := 0; i < len(indices); i++ {
		index_range := indices[i]
		if index_range.start == 0 && index_range.end == -1 {
			// axis
			are_constants = false
		} else if index_range.end != index_range.start {
			// sub axis
			are_constants = false
		} else if index_range.start == index_range.end {
			// constant
			are_axis_wide = false
		}
	}

	if are_constants {
		idxs := make([]int, len(indices))
		for i, idx_range := range indices {
			idxs[i] = idx_range.start
		}
		return tensor.Index(idxs...)
	}
	if are_axis_wide {
		return tensor.Copy()
	}
	// tensor.Index(index_range.end)
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
		outTensor.data()[dataIndex] = val
	}
	return outTensor
}
