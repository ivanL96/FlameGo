package tensor

import (
	"reflect"
	"unsafe"
)

func create_slice[T any](n int, value T) []T {
	slice := make([]T, n)
	for i := range slice {
		slice[i] = value
	}
	return slice
}

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

func Equal_1D_slices[T Number](slice1, slice2 []T) bool {
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

func get_type_array[T Number](arr []T) reflect.Type {
	return reflect.TypeOf(arr).Elem()
}

func convert_slice_type[OLD_T, NEW_T Number](slice []OLD_T) []NEW_T {
	// doesnt work yet
	orig_sizeof := unsafe.Sizeof(slice[0])
	new_sizeof := unsafe.Sizeof(NEW_T(0))
	n_elements := uintptr(len(slice)) * orig_sizeof / new_sizeof // length of a new slice type
	slice_addr := &slice[0]
	converted := unsafe.Slice((*NEW_T)(unsafe.Pointer(slice_addr)), n_elements)
	return converted
}

func repeat_slice[T Number](data []T, ntimes uint) []T {
	length := len(data) * int(ntimes)
	replicated_data := make([]T, 0, int(length))
	for i := 0; i < int(ntimes); i++ {
		replicated_data = append(replicated_data, data...)
	}
	return replicated_data
}

func add_left_padding[T Number](slice []T, padding_size, padding_val int) []T {
	expanded_slice_size := len(slice) + padding_size
	expanded_slice := make([]T, expanded_slice_size)
	for i := 0; i < expanded_slice_size; i++ {
		if i < padding_size {
			expanded_slice[i] = 1
			continue
		}
		expanded_slice[i] = slice[i-padding_size]
	}
	return expanded_slice
}
