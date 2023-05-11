package tensor

import "fmt"

// set of operations for shaping routines

func (tensor *Tensor[T]) Broadcast(shape Shape) *Tensor[T] {
	// tries to broadcast the shape and replicate the data accordingly
	tensor.shape = broadcast(tensor.shape, shape)
	tensor.ndim = uint(len(tensor.shape))
	return tensor
}

func (tensor *Tensor[T]) Flatten() *Tensor[T] {
	tensor.shape = Shape{uint(len(tensor.data))}
	tensor.ndim = 1
	return tensor
}

func (tensor *Tensor[T]) Reshape(new_shape ...uint) *Tensor[T] {
	var new_shape_prod uint = 1
	for _, dim := range new_shape {
		new_shape_prod *= dim
	}
	if tensor.len != new_shape_prod {
		panic(fmt.Sprintf("Cannot reshape to shape %v", new_shape))
	}
	tensor.shape = new_shape
	tensor.ndim = uint(len(new_shape))
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
