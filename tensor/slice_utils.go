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

func get_param[T types.TensorType](params ...*Tensor[T]) *Tensor[T] {
	if len(params) == 1 {
		return params[0]
	}
	if len(params) > 1 {
		panic("Only one out tensor is allowed")
	}
	return nil
}

// sets specific value to []T buffer using loop unrolling opt.
func fill_data_loop[T types.TensorType](buffer []T, value T) {
	lb := len(buffer)
	n := 8
	for i := 0; i < lb/n; i += n {
		buffer[i] = value
		buffer[i+1] = value
		buffer[i+2] = value
		buffer[i+3] = value
		buffer[i+4] = value
		buffer[i+5] = value
		buffer[i+6] = value
		buffer[i+7] = value
	}
	for i := lb - lb%n; i < lb; i++ {
		buffer[i] = value
	}
}

func convert_type_loop[OLD_T, NEW_T types.TensorType](data []OLD_T, out_data []NEW_T) {
	lb := len(data)
	n := 8
	for i := 0; i < lb/n; i += n {
		out_data[i] = NEW_T(data[i])
		out_data[i+1] = NEW_T(data[i+1])
		out_data[i+2] = NEW_T(data[i+2])
		out_data[i+3] = NEW_T(data[i+3])
		out_data[i+4] = NEW_T(data[i+4])
		out_data[i+5] = NEW_T(data[i+5])
		out_data[i+6] = NEW_T(data[i+6])
		out_data[i+7] = NEW_T(data[i+7])
	}
	for i := lb - lb%n; i < lb; i++ {
		out_data[i] = NEW_T(data[i])
	}
}
