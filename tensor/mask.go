package tensor

import (
	"errors"
	"fmt"
)

// Example:
//
// tensor_a: [[1,2,3][4,5,6][7,8,9]]
//
// mask: [1,0,2]
//
// tensor_a.IndexMask(mask, true) => [2,4,9]
// if enumerate is enabled, the mask will be applied for each tensor element.
//
// If enumerate is turned off:
//
// tensor_a.IndexMask(mask, false) => [[4,5,6],[1,2,3],[7,8,9]]
func (tensor *Tensor[T]) IndexMask(mask_tensor *Tensor[T], enumerate bool) *Tensor[T] {
	if tensor.Err != nil {
		return tensor
	}
	if len(mask_tensor.Squeeze().Shape()) > 2 {
		tensor.Err = errors.New("mask_tensor must have 2 dims max")
		return tensor
	}
	dims := len(tensor.Shape())
	if dims < len(mask_tensor.Shape()) {
		tensor.Err = fmt.Errorf("mask_tensor must have no more dimensions than `tensor`. %v < %v",
			dims, len(mask_tensor.Shape()))
		return tensor
	}

	mask_tensor = mask_tensor.AsContinuous()

	// TODO create a result buffer in advance instead of TensorList
	tlist := &TensorList[T]{}
	for i := 0; i < int(tensor.Shape()[0]); i++ {
		mask_i := AsType[T, int](mask_tensor.Index(i)).data()
		if enumerate {
			mask_i = append([]int{i}, mask_i...)
		}
		masked := tensor.Index(mask_i...).Unsqueeze(0)
		tlist.Append(masked)
	}
	return tlist.StackedTensors
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
	mask_tensor = mask_tensor.AsContinuous()

	for i := 0; i < int(tensor.Shape()[0]); i++ {
		mask_i := AsType[T, int](mask_tensor.Index(i)).data()
		if enumerate {
			mask_i = append([]int{i}, mask_i...)
		}
		tensor.Set(mask_i, value)
	}
}
