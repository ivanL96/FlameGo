package tensor

import (
	types "gograd/tensor/types"
	"reflect"
)

func EqualSlices[T types.TensorType](slice1, slice2 []T) bool {
	if len(slice1) != len(slice2) {
		return false
	}
	for i, element := range slice1 {
		if element != slice2[i] {
			return false
		}
	}
	return true
}

func getTypeArray[T types.TensorType](arr []T) reflect.Type {
	return reflect.TypeOf(arr).Elem()
}

func get_param[T types.TensorType](params ...*Tensor[T]) *Tensor[T] {
	if len(params) == 1 {
		return params[0]
	}
	if len(params) > 1 {
		panic("Only one out tensor is allowed")
	}
	return nil
}
