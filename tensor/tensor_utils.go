package tensor

import (
	types "gograd/tensor/types"
	"reflect"
)

func squeeze_shape(shape types.Shape) types.Shape {
	result := types.Shape{1}
	for _, v := range shape {
		if v > 1 {
			result = append(result, v)
		}
	}
	return result
}

func IsScalarLike(shape types.Shape) bool {
	if len(shape) == 1 && shape[0] == 1 {
		return true
	}
	return false
}

func getStrides(shape types.Shape) []int {
	if len(shape) == 1 {
		return []int{1}
	}
	strides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= int(shape[i])
	}
	return strides
}

func initDimOrder(shape types.Shape) []uint16 {
	if len(shape) == 1 {
		return []uint16{0}
	}
	dimOrder := make([]uint16, len(shape))
	for i := range dimOrder {
		dimOrder[i] = uint16(i)
	}
	return dimOrder
}

// if dim order is not shuffled
func isDimOrderInit(dimOrder []uint16) bool {
	var min uint16 = 0
	for _, dim := range dimOrder {
		if dim > min {
			return false
		}
		min += 1
	}
	return true
}

func isIntKind(tensorDType reflect.Type) bool {
	switch tensorDType.Kind() {
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64,
		reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return true
	default:
		return false
	}
}

func PrepareOutTensor[T types.TensorType](out *Tensor[T], shape types.Shape) *Tensor[T] {
	if out == nil {
		return InitEmptyTensor[T](shape...)
	}
	return out
}
