package tensor

import "fmt"

// set of operations for shaping routines

func (tensor *Tensor[T]) Broadcast(shape ...Dim) *Tensor[T] {
	// tries to broadcast the shape and replicate the data accordingly
	if Equal_1D_slices(tensor.shape, shape) {
		return tensor
	}

	tensor.shape = broadcast(tensor.shape, shape)
	var length Dim = 1 // new number of elements
	for _, dim := range tensor.shape {
		length *= dim
	}

	// repeat data
	ntimes := int(length) / len(tensor.data)
	replicated_data := repeat_slice(tensor.data, uint(ntimes))
	tensor.data = replicated_data
	return tensor
}

func (tensor *Tensor[T]) Flatten() *Tensor[T] {
	tensor.shape = Shape{Dim(len(tensor.data))}
	return tensor
}

func (tensor *Tensor[T]) Reshape(new_shape ...Dim) *Tensor[T] {
	var new_shape_prod Dim = 1
	for _, dim := range new_shape {
		new_shape_prod *= dim
	}
	if len(tensor.data) != int(new_shape_prod) {
		panic(fmt.Sprintf("Cannot reshape to shape %v", new_shape))
	}
	tensor.shape = new_shape
	return tensor
}

// returns sub data for given indices. Doesn't create a new tensor
func (tensor *Tensor[T]) View(indices ...int) []T {
	if len(indices) > int(len(tensor.shape)) {
		panic("Too many indices")
	}

	var shape_prod Dim = 1
	for _, dim := range tensor.shape {
		shape_prod *= dim
	}

	sub_data := tensor.data
	for i, ind := range indices {
		dim := tensor.shape[i]
		if ind < 0 {
			ind = int(dim) + ind
		}

		if ind < 0 || ind >= int(dim) {
			panic(fmt.Sprintf("Index %v out of range", indices[i]))
		}

		shape_prod /= dim
		start := int(shape_prod) * ind
		end := start + int(shape_prod)
		sub_data = sub_data[start:end]
	}
	return sub_data
}

func (tensor *Tensor[T]) Index(indices ...int) *Tensor[T] {
	new_shape_size := len(tensor.shape) - len(indices)
	if new_shape_size == 0 {
		return InitTensor(tensor.View(indices...), Shape{1})
	}
	sub_shape := make(Shape, new_shape_size)
	copy(sub_shape, tensor.shape[len(indices):])
	return InitTensor(tensor.View(indices...), sub_shape)
}
