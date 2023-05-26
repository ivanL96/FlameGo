package tensor

import (
	"fmt"
	types "gograd/tensor/types"
)

// set of operations for shaping routines

func AreBroadcastable(shape_a, shape_b types.Shape) bool {
	if (IsScalarLike(shape_a) && IsScalarLike(shape_b)) ||
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

func BroadcastShapes(shape_a, shape_b types.Shape) (types.Shape, int) {
	if IsScalarLike(shape_a) && IsScalarLike(shape_b) || Equal_1D_slices(shape_a, shape_b) {
		return shape_a, -1
	}
	if len(shape_a) < len(shape_b) {
		ones_size := len(shape_b) - len(shape_a)
		shape_a = addLeftPadding(shape_a, ones_size, 1)
	} else if len(shape_b) < len(shape_a) {
		ones_size := len(shape_a) - len(shape_b)
		shape_b = addLeftPadding(shape_b, ones_size, 1)
	}
	// # Start from the trailing dimensions
	result_shape := make(types.Shape, len(shape_a))
	nDimBroadcasted := 0
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
			nDimBroadcasted = i
			result_shape[i] = dim2
		} else if dim1 != 1 {
			nDimBroadcasted = i
			result_shape[i] = dim1
		} else {
			panic("Something went wrong during broadcasting")
		}
	}
	return result_shape, nDimBroadcasted
}

func (tensor *Tensor[T]) Broadcast(shape ...types.Dim) *Tensor[T] {
	// tries to broadcast the shape and replicate the data accordingly
	if Equal_1D_slices(tensor.shape, shape) {
		return tensor
	}

	broadcastedShape, nDimBroadcasted := BroadcastShapes(tensor.shape, shape)
	var shapeProd types.Dim = 1 // new number of elements
	for _, dim := range broadcastedShape {
		shapeProd *= dim
	}

	// repeat data
	ntimes := int(shapeProd) / len(tensor.data)
	outTensor := InitEmptyTensor[T](broadcastedShape...)
	iter := tensor.CreateIterator()
	for iter.Iterate() {
		idx := iter.Next()
		value := tensor.Get(idx...)
		outIdx := append([]int(nil), idx...)
		for i := 0; i < ntimes; i++ {
			outTensor.data[outTensor.getFlatIndex(outIdx...)] = value
			outIdx[nDimBroadcasted]++
		}
	}
	return outTensor
}

func (tensor *Tensor[T]) Flatten() *Tensor[T] {
	outTensor := tensor.Copy()
	outTensor.shape = types.Shape{types.Dim(len(tensor.data))}
	outTensor.strides = []int{1}
	return outTensor
}

func (tensor *Tensor[T]) Squeeze() *Tensor[T] {
	outTensor := tensor.Copy()
	if IsScalarLike(tensor.shape) {
		return outTensor
	}
	outTensor.shape = squeeze_shape(tensor.shape)
	outTensor.strides = getStrides(outTensor.shape)
	outTensor.dim_order = initDimOrder(outTensor.shape)
	return outTensor
}

func (tensor *Tensor[T]) Reshape(newShape ...types.Dim) *Tensor[T] {
	var new_shape_prod types.Dim = 1
	for _, dim := range newShape {
		new_shape_prod *= dim
	}
	if new_shape_prod == 0 {
		panic("Shape cannot have 0 dim size.")
	}
	if len(tensor.data) != int(new_shape_prod) {
		panic(fmt.Sprintf("Cannot reshape to %v", newShape))
	}
	tensor.shape = newShape
	tensor.strides = getStrides(newShape)
	tensor.dim_order = initDimOrder(newShape)
	return tensor
}

func (tensor *Tensor[T]) Transpose(axes ...uint) *Tensor[T] {
	n_dims := len(tensor.shape)
	if len(axes) == 0 {
		axes = make([]uint, n_dims)
		for i := range axes {
			axes[i] = uint(len(axes) - i - 1)
		}
	}

	outTensor := InitTensor(tensor.data, tensor.shape)

	for i, axis := range axes {
		outTensor.shape[i] = tensor.shape[axis]
		outTensor.strides[i] = tensor.strides[axis]
		outTensor.dim_order[i] = tensor.dim_order[axis]
	}
	return outTensor
}
