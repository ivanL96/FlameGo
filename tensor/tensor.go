package tensor

import "fmt"

func make_tensor[T Number](data []T, shape Shape) *Tensor[T] {
	var ndim Dim = Dim(len(shape))

	new_tensor := &Tensor[T]{
		shape: shape,
		data:  data,
		ndim:  ndim,
		len:   uint(len(data)),
		dtype: get_type_array(data),
	}
	return new_tensor
}

func InitTensor[T Number](value []T, shape Shape) *Tensor[T] {
	// inits a tensor with data
	var prod Dim = 1
	for _, dim := range shape {
		prod *= dim
	}
	if int(prod) != len(value) {
		panic(fmt.Sprintf("Value length cannot have shape %v", shape))
	}
	return make_tensor(value, shape)
}

func InitEmptyTensor[T Number](shape ...Dim) *Tensor[T] {
	var length Dim = 1
	for _, dim := range shape {
		length *= dim
	}
	zeroes := make([]T, length)
	return make_tensor(zeroes, shape)
}

func (tensor *Tensor[T]) Set(value []T) *Tensor[T] {
	// sets new value of the same shape
	length := uint(len(value))
	var prod uint = 1
	for _, dim := range tensor.shape {
		prod = uint(dim) * prod
	}
	if prod != length {
		msg := fmt.Sprintf(
			"Shape %v cannot fit the number of elements %v. Change the shape first",
			tensor.shape, length)
		panic(msg)
	}
	tensor.len = length
	tensor.data = value
	return tensor
}

func AsType[T Number, DT Number](tensor *Tensor[T]) *Tensor[DT] {
	// naive impl with copying the data & tensor
	data := make([]DT, tensor.len)
	for i, val := range tensor.data {
		data[i] = DT(val)
	}
	new_tensor := InitTensor[DT](data, tensor.shape)
	return new_tensor
}

func (tensor *Tensor[T]) Copy() *Tensor[T] {
	new_data := make([]T, len(tensor.data))
	copy(new_data, tensor.data)
	new_copy := InitTensor[T](new_data, tensor.shape)
	return new_copy
}

func (tensor *Tensor[T]) Compare(other_tensor *Tensor[T]) bool {
	// iterates over two tensors and compares elementwise
	if tensor.len != other_tensor.len {
		return false
	}
	for i, element := range tensor.data {
		if element != other_tensor.data[i] {
			return false
		}
	}
	return true
}

func (tensor *Tensor[T]) Fill(fill_value T) *Tensor[T] {
	for i := range tensor.data {
		tensor.data[i] = fill_value
	}
	return tensor
}
