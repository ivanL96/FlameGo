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

// sets specific value to []T buffer using loop unrolling opt.
func fill_data_loop[T types.TensorType](buffer []T, value T) {
	lb := len(buffer)
	for i := 0; i < lb/8; i += 8 {
		buffer[i] = value
		buffer[i+1] = value
		buffer[i+2] = value
		buffer[i+3] = value
		buffer[i+4] = value
		buffer[i+5] = value
		buffer[i+6] = value
		buffer[i+7] = value
	}
	for i := lb - lb%8; i < lb; i++ {
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

func transpose_cont2D_loop[T types.TensorType](data, transposed []T, i, cols, rows int) {
	i_cols := i * cols
	n := 8
	for j := 0; j < cols/n; j += n {
		i_cols_j := i_cols + j
		transposed[j*rows+i] = data[i_cols_j]
		transposed[(j+1)*rows+i] = data[i_cols_j+1]
		transposed[(j+2)*rows+i] = data[i_cols_j+2]
		transposed[(j+3)*rows+i] = data[i_cols_j+3]

		transposed[(j+4)*rows+i] = data[i_cols_j+4]
		transposed[(j+5)*rows+i] = data[i_cols_j+5]
		transposed[(j+6)*rows+i] = data[i_cols_j+6]
		transposed[(j+7)*rows+i] = data[i_cols_j+7]
	}
	for j := cols - cols%n; j < cols; j++ {
		transposed[j*rows+i] = data[i_cols+j]
	}
}
