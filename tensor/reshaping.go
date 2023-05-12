package tensor

import "fmt"

// set of operations for shaping routines

func (tensor *Tensor[T]) Broadcast(shape ...Dim) *Tensor[T] {
	// tries to broadcast the shape and replicate the data accordingly
	tensor.shape = broadcast(tensor.shape, shape)
	tensor.ndim = Dim(len(tensor.shape))
	return tensor
}

func (tensor *Tensor[T]) Flatten() *Tensor[T] {
	tensor.shape = Shape{Dim(len(tensor.data))}
	tensor.ndim = 1
	return tensor
}

func (tensor *Tensor[T]) Reshape(new_shape ...Dim) *Tensor[T] {
	var new_shape_prod Dim = 1
	for _, dim := range new_shape {
		new_shape_prod *= dim
	}
	if Dim(tensor.len) != new_shape_prod {
		panic(fmt.Sprintf("Cannot reshape to shape %v", new_shape))
	}
	tensor.shape = new_shape
	tensor.ndim = Dim(len(new_shape))
	return tensor
}

func (tensor *Tensor[T]) Index(indices ...int) *Tensor[T] {
	if len(indices) > int(tensor.ndim) {
		panic("Too many indices")
	}

	var shape_prod Dim = 1
	for _, dim := range tensor.shape {
		shape_prod *= dim
	}

	sub_data := tensor.data
	for i, ind := range indices {
		if ind >= int(tensor.shape[i]) {
			panic(fmt.Sprintf("Index %v out of range", ind))
		}
		dim := tensor.shape[i]
		shape_prod /= dim
		start := int(shape_prod) * ind
		end := start + int(shape_prod)
		sub_data = sub_data[start:end]
		fmt.Println(start, end, sub_data)
	}
	//[0, 1, 2,
	// 3, 4, 5, <--
	// 6, 7, 8,
	// 9, 10, 11,
	// 12, 13, 14]
	// Example: array [0 1 2 3 4 5], shape=(3,2), index=2 => array[2] = [4,5]
	return tensor
}

func get_value_by_index[T Number](tensor *Tensor[T], index int) *Tensor[T] {
	// if tensor.ndim <= 1 {
	// 	return &Tensor{data:tensor.data[index])
	// }
	if index < 0 {
		panic("negative index not supported yet")
	}
	// dim := tensor.shape[0]
	// offset := dim * uint(index)
	return tensor
}
