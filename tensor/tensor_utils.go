package tensor

import (
	"fmt"
	types "gograd/tensor/types"
	"reflect"
)

// if dim order is not shuffled
func isDimOrderInit(dimOrder []uint16) bool {
	switch len(dimOrder) {
	case 1:
		return true
	case 2:
		return dimOrder[0] == 0
	default:
		var min uint16 = 0
		for _, dim := range dimOrder {
			if dim > min {
				return false
			}
			min += 1
		}
		return true
	}
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
