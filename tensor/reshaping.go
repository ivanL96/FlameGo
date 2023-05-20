package tensor

import "fmt"

// set of operations for shaping routines

func are_broadcastable(shape_a, shape_b Shape) bool {
	if (isScalarLike(shape_a) && isScalarLike(shape_b)) ||
		Equal_1D_slices(shape_a, shape_b) {
		return true
	}
	// If one shape has more dimensions than the other, prepend 1s to the shape of the smaller array
	if len(shape_a) < len(shape_b) {
		ones_size := len(shape_b) - len(shape_a)
		shape_a = addLeftPadding(shape_a, ones_size, 1)
	} else if len(shape_b) < len(shape_a) {
		ones_size := len(shape_a) - len(shape_b)
		shape_b = addLeftPadding(shape_b, ones_size, 1)
	}
	// Start from the trailing dimensions and work forward
	for i := len(shape_a) - 1; i >= 0; i-- {
		dim1 := shape_a[i]
		dim2 := shape_b[i]
		if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
			return false
		}
	}
	return true
}

func broadcast(shape_a, shape_b Shape) Shape {
	if isScalarLike(shape_a) && isScalarLike(shape_b) || Equal_1D_slices(shape_a, shape_b) {
		return shape_a
	}
	if len(shape_a) < len(shape_b) {
		ones_size := len(shape_b) - len(shape_a)
		shape_a = addLeftPadding(shape_a, ones_size, 1)
	} else if len(shape_b) < len(shape_a) {
		ones_size := len(shape_a) - len(shape_b)
		shape_b = addLeftPadding(shape_b, ones_size, 1)
	}
	// # Start from the trailing dimensions and work forward
	result_shape := make(Shape, len(shape_a))
	for i := len(shape_a) - 1; i >= 0; i-- {
		dim1 := shape_a[i]
		dim2 := shape_b[i]
		if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
			panic(
				fmt.Sprintf(
					"Shapes %v and %v are not broadcastable: Dim1 '%v' not equal Dim2 '%v'", shape_a, shape_b, dim1, dim2,
				),
			)
		}
		if dim1 == dim2 {
			result_shape[i] = dim1
		} else if dim2 != 1 {
			result_shape[i] = dim2
		} else if dim1 != 1 {
			result_shape[i] = dim1
		} else {
			panic("Something went wrong during broadcasting")
		}
	}
	return result_shape
}

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
	data := repeatSlice(tensor.data, uint(ntimes))
	return &Tensor[T]{
		shape: broadcastedShape,
		data:  data,
		dtype: getTypeArray(data),
	}
}

func (tensor *Tensor[T]) Flatten() *Tensor[T] {
	outTensor := tensor.Copy()
	outTensor.shape = Shape{Dim(len(tensor.data))}
	outTensor.strides = []int{1}
	return outTensor
}

func (tensor *Tensor[T]) Squeeze() *Tensor[T] {
	outTensor := tensor.Copy()
	if isScalarLike(tensor.shape) {
		return outTensor
	}
	outTensor.shape = squeeze_shape(tensor.shape)
	outTensor.strides = getStrides(outTensor.shape)
	outTensor.dim_order = initDimOrder(outTensor.shape)
	return outTensor
}

func (tensor *Tensor[T]) Reshape(newShape ...Dim) *Tensor[T] {
	var new_shape_prod Dim = 1
	for _, dim := range newShape {
		new_shape_prod *= dim
	}
	if len(tensor.data) != int(new_shape_prod) {
		panic(fmt.Sprintf("Cannot reshape to %v", newShape))
	}
	tensor.shape = newShape
	tensor.strides = getStrides(newShape)
	tensor.dim_order = initDimOrder(newShape)
	return tensor
}

func (tensor *Tensor[T]) Transpose(axes ...int) *Tensor[T] {
	if len(axes) == 0 {
		axes = make([]int, len(tensor.shape))
		for i := range axes {
			axes[i] = len(axes) - i - 1
		}
	}

	// TODO replace with InitTensor
	outTensor := &Tensor[T]{
		data:      tensor.data,
		shape:     make(Shape, len(tensor.shape)),
		strides:   make([]int, len(tensor.shape)),
		dtype:     tensor.dtype,
		dim_order: make([]int, len(tensor.shape)),
	}

	for i, axis := range axes {
		outTensor.shape[i] = tensor.shape[axis]
		outTensor.strides[i] = tensor.strides[axis]
		outTensor.dim_order[i] = tensor.dim_order[axis]
	}
	return outTensor
}
