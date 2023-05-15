package tensor

import "fmt"

// set of operations for shaping routines

func (tensor *Tensor[T]) Broadcast(shape ...Dim) *Tensor[T] {
	// tries to broadcast the shape and replicate the data accordingly
	if Equal_1D_slices(tensor.shape, shape) {
		return tensor
	}

	broadcastedShape := broadcast(tensor.shape, shape)
	var shapeProd Dim = 1 // new number of elements
	for _, dim := range broadcastedShape {
		shapeProd *= dim
	}

	// repeat data
	ntimes := int(shapeProd) / len(tensor.data)
	fmt.Println("ntimes", ntimes)
	data := repeat_slice(tensor.data, uint(ntimes))
	return &Tensor[T]{
		shape:     broadcastedShape,
		data:      data,
		dtype:     getTypeArray(data),
		shapeProd: shapeProd,
	}
}

func (tensor *Tensor[T]) Flatten() *Tensor[T] {
	dim := Dim(len(tensor.data))
	tensor.shape = Shape{dim}
	tensor.shapeProd = dim
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
	tensor.shapeProd = new_shape_prod
	tensor.shape = new_shape
	return tensor
}

// returns sub data for given indices. Doesn't copy tensor
func (tensor *Tensor[T]) View(indices ...int) *Tensor[T] {
	if len(indices) > len(tensor.shape) {
		panic("Too many indices")
	}

	var shape_prod Dim = tensor.shapeProd

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
	sub_shape := make(Shape, len(tensor.shape)-len(indices))
	if len(sub_shape) == 0 {
		sub_shape = Shape{1}
	} else {
		copy(sub_shape, tensor.shape[len(indices):])
	}
	return InitTensor[T](sub_data, sub_shape)
}

// same as tensor.View but with copying the data
func (tensor *Tensor[T]) Index(indices ...int) *Tensor[T] {
	sub_tensor := tensor.View(indices...)
	sub_data_copy := make([]T, len(sub_tensor.data))
	copy(sub_data_copy, sub_tensor.data)
	return InitTensor(sub_data_copy, sub_tensor.shape)
}

// func (tensor *Tensor[T]) Transpose(axes ...uint) *Tensor[T] {
// 	if len(axes) == 0 {
// 		axes = make([]uint, len(tensor.shape))
// 		for i := range axes {
// 			axes[i] = uint(len(axes) - i - 1)
// 		}
// 	}

// 	if len(axes) != len(tensor.shape) {
// 		panic("The number of axes does not match the dimension of the tensor")
// 	}

// 	// new shape with swapped axes
// 	newShape := make(Shape, len(tensor.shape))
// 	for i, axis := range axes {
// 		newShape[i] = tensor.shape[axis]
// 	}

// 	oldStrides := getStrides(tensor.shape)
// 	newStrides := getStrides(newShape)

// 	newData := make([]T, len(tensor.data))
// 	for i := range tensor.data {
// 		newIndex := 0
// 		oldIndex := i
// 		for j := 0; j < len(tensor.shape); j++ {
// 			newIndex += (oldIndex / oldStrides[j]) * newStrides[axes[j]]
// 			oldIndex %= oldStrides[j]
// 		}
// 		newData[i] = tensor.data[newIndex]
// 	}

// 	return &Tensor[T]{data: newData, shape: newShape, dtype: tensor.dtype, shapeProd: tensor.shapeProd}
// }

func (tensor *Tensor[T]) Transpose(axes ...int) *Tensor[T] {
	if len(axes) == 0 {
		axes = make([]int, len(tensor.shape))
		for i := range axes {
			axes[i] = len(axes) - i - 1
		}
	}

	newTensor := &Tensor[T]{
		data:      tensor.data,
		shape:     make(Shape, len(tensor.shape)),
		strides:   make([]int, len(tensor.shape)),
		dtype:     tensor.dtype,
		shapeProd: tensor.shapeProd,
		dim_order: make([]int, len(tensor.shape)),
	}

	for i, axis := range axes {
		newTensor.shape[i] = tensor.shape[axis]
		newTensor.strides[i] = tensor.strides[axis]
		newTensor.dim_order[i] = tensor.dim_order[axis]
	}

	return newTensor
}

func (tensor *Tensor[T]) Get(index ...int) T {
	if len(index) != len(tensor.shape) {
		panic("Incorrect number of indices")
	}

	flat_index := 0
	for i, ind := range index {
		flat_index += int(tensor.strides[tensor.dim_order[i]]) * ind
	}

	return tensor.data[flat_index]
}
