package tensor

import (
	"errors"
	"fmt"
	types "gograd/tensor/types"
)

// tensor.IndexMask() can work in two modes. When 'enumerate' is false
// each value of the mask will represent each element(or subtensor) of 'tensor'.
//
// Example:
//
// tensor_a: [[1,2,3][4,5,6][7,8,9]]
// mask: [1,0,2]
//
// If enumerate is turned off:
//
// tensor_a.IndexMask(mask, false) => [[4,5,6],[1,2,3],[7,8,9]]. Here we get shuffled tensor with 1st, 0th and 2nd subtensors
//
// if enumerate is enabled:
//
// tensor_a.IndexMask(mask, true) => [2,4,9]
//
// Also enumerate can be recreated using this mask:
//
// tensor_a.IndexMask([0,1],[1,0],[2,2]], false) => [2,4,9] - same result
func (tensor *Tensor[T]) IndexMask(mask_tensor *Tensor[T], enumerate bool) *Tensor[T] {
	if tensor.Err != nil {
		return tensor
	}
	// TODO support for nDim masks
	if len(mask_tensor.Shape().Squeeze()) > 2 {
		tensor.Err = errors.New("mask_tensor must have 2 dims max")
		return tensor
	}
	dims := len(tensor.Shape())
	if dims < len(mask_tensor.Shape()) {
		tensor.Err = fmt.Errorf("mask_tensor must have no more dimensions than `tensor`. %v < %v",
			dims, len(mask_tensor.Shape()))
		return tensor
	}

	mask_tensor = mask_tensor.AsContiguous()

	to_reduce := make([]int, len(mask_tensor.shape)-1)
	for i := 0; i < len(mask_tensor.shape)-1; i++ {
		to_reduce[i] = i + 1
	}
	out_shape := append(types.Shape{}, tensor.shape...)
	if len(to_reduce) > 0 {
		out_shape = tensor.shape.ReduceDim(to_reduce...).SqueezeInner()
	}

	// tlist := &TensorList[T]{}
	stacked := CreateEmptyTensor[T](out_shape...)
	start := 0
	mask_tensor_int := AsType[T, int](mask_tensor)

	for i := 0; i < int(tensor.Shape()[0]); i++ {
		mask_i := mask_tensor_int.Index(i).data()
		// mask_i := AsType[T, int](mask_tensor.Index(i)).data()
		var masked *Tensor[T]
		if enumerate {
			masked = tensor.Index(i).Index(mask_i...).Unsqueeze(0)
		} else {
			masked = tensor.Index(mask_i...).Unsqueeze(0)
		}
		// tlist.Append(masked)
		end := int(masked.Size())
		copy(stacked.data()[start:start+end], masked.data())
		start += end
	}
	// return tlist.StackedTensors
	return stacked
}

func (tensor *Tensor[T]) SetByIndexMask(mask_tensor *Tensor[T], enumerate bool, value T) {
	if tensor.Err != nil {
		return
	}
	if mask_tensor.Shape()[0] != tensor.Shape()[0] {
		tensor.Err = errors.New("mask_tensor and tensor must have same 0-dim")
		return
	}
	if len(mask_tensor.Squeeze().Shape()) > 2 {
		tensor.Err = errors.New("mask_tensor must have 2 dims max")
		return
	}
	dims := len(tensor.Shape())
	if dims < len(mask_tensor.Shape()) {
		tensor.Err = fmt.Errorf("mask_tensor must have no more dimensions than `tensor`. %v < %v",
			dims, len(mask_tensor.Shape()))
		return
	}
	mask_tensor = mask_tensor.AsContiguous()

	for i := 0; i < int(tensor.Shape()[0]); i++ {
		mask_i := AsType[T, int](mask_tensor.Index(i)).data()
		if enumerate {
			mask_i = append([]int{i}, mask_i...)
		}
		tensor.Set(mask_i, value)
	}
}

// tries to find a value in tensor and returns its index
//
// Example:
//
// tensor [[1,2,3],[4,5,6]]
//
// tensor.Find(3) => [0,2], value 3 can be found at index (0,2)
func (tensor *Tensor[T]) Find(value T) ([]int, error) {
	if tensor.Err != nil {
		return []int{}, tensor.Err
	}

	it := tensor.CreateIterator()
	for it.Iterate() {
		i := it.Index()
		idx := it.Next()
		if tensor.data()[i] == value {
			return idx, nil
		}
	}
	return []int{}, nil
}
