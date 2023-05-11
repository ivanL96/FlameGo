package tensor

func InitTensor[T Number](shape ...uint) *Tensor[T] {
	// inits a tensor with no data
	var prod uint = 1
	for _, dim := range shape {
		prod *= dim
	}
	var ndim uint = uint(len(shape))

	data := make([]T, prod)
	new_tensor := &Tensor[T]{shape: shape, data: data, ndim: ndim, len: prod, dtype: get_type_array(data)}
	return new_tensor
}

func (tensor *Tensor[T]) Set(value []T) *Tensor[T] {
	// sets new value of the same shape
	length := uint(len(value))
	var prod uint = 1
	for _, dim := range tensor.shape {
		prod = dim * prod
	}
	if prod != length {
		panic("Initialized shape of tensor not equal the number of elements. Change the shape first")
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
	new_tensor := InitTensor[DT](tensor.shape...)
	new_tensor.data = data
	return new_tensor
}

func (tensor *Tensor[T]) Copy() *Tensor[T] {
	new_copy := InitTensor[T](tensor.shape...)
	copy(new_copy.data, tensor.data)
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

func (tensor *Tensor[T]) Broadcast(shape Shape) *Tensor[T] {
	// tries to broadcast the shape and replicate the data accordingly
	tensor.shape = broadcast(tensor.shape, shape)
	tensor.ndim = uint(len(tensor.shape))
	return tensor
}

func (tensor *Tensor[T]) Index(indices ...int) *Tensor[T] {
	// Example: array [0 1 2 3 4 5], shape=(3,2), index=2 => array[2] = [4,5]
	if len(indices) > int(tensor.ndim) {
		panic("Too many indices")
	}

	// for i, index := range indices { // (1, 1)
	// 	if index >= tensor.shape[i] {
	// 		panic("Index out of range")
	// 	}
	// 	index_dim := tensor.shape[i]
	// 	start += index * index_dim
	// 	size *= tensor.shape[i]
	// 	fmt.Print("start: ")
	// 	fmt.Println(start)
	// }
	for _, index := range indices {
		get_value_by_index(tensor, index)
	}

	return tensor
}

func get_value_by_index[T Number](tensor *Tensor[T], index int) *Tensor[T] {
	// if tensor.ndim <= 1 {
	// 	return &Tensor{data:tensor.data[index])
	// }
	return tensor
}
