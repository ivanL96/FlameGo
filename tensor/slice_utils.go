package tensor

import (
	types "gograd/tensor/types"
	"reflect"
	"unsafe"
)

func reverse_slice_inplace[T any](slice []T) {
	for i := len(slice)/2 - 1; i >= 0; i-- {
		opp := len(slice) - 1 - i
		slice[i], slice[opp] = slice[opp], slice[i]
	}
}

func reverse_slice_copy[T any](slice []T) []T {
	rev_slice := make([]T, len(slice))
	for i := range slice {
		rev_slice[i] = slice[len(slice)-1-i]
	}
	return rev_slice
}

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

func convert_slice_type[OLD_T, NEW_T types.TensorType](slice []OLD_T) []NEW_T {
	// doesnt work yet
	origSizeof := unsafe.Sizeof(slice[0])
	newSizeof := unsafe.Sizeof(NEW_T(0))
	n_Elements := uintptr(len(slice)) * origSizeof / newSizeof // length of a new slice type
	converted := unsafe.Slice((*NEW_T)(unsafe.Pointer(&slice[0])), n_Elements)
	return converted
}

func repeatSlice[T types.TensorType](data []T, ntimes uint) []T {
	length := len(data) * int(ntimes)
	replicated := make([]T, int(length))
	for i := 0; i < int(ntimes); i++ {
		copy(replicated[i*len(data):(i+1)*len(data)], data)
	}
	return replicated
}

func enumerate(n uint) []int {
	slice := make([]int, n)
	for i := 0; i < len(slice); i++ {
		slice[i] = i
	}
	return slice
}

// sets specific value to []T buffer using loop unrolling opt.
// Don't use for buffer with less than 4 elements
func fill_data_unroll4[T types.TensorType](
	buffer *[]T, value T) {
	data := *buffer
	for i := 0; i < len(data); i += 4 {
		bb := (*[4]T)(unsafe.Pointer(&data[i]))
		bb[0] = value
		bb[1] = value
		bb[2] = value
		bb[3] = value
	}
}

func get_param[T types.TensorType](params ...*Tensor[T]) *Tensor[T] {
	if len(params) == 1 {
		return params[0]
	}
	return nil
}
