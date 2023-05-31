package tensor

import (
	"fmt"
	types "gograd/tensor/types"
)

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

func BroadcastShapes(shape_a, shape_b types.Shape) (types.Shape, []int) {
	if IsScalarLike(shape_a) && IsScalarLike(shape_b) || Equal_1D_slices(shape_a, shape_b) {
		return shape_a, make([]int, 0)
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
	broadcasted_dims := make([]int, len(shape_a))
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
			// fmt.Printf("is broadcasted %v\n", i)
			broadcasted_dims[i] = 1
			result_shape[i] = dim2
		} else if dim1 != 1 {
			// fmt.Printf("is broadcasted %v\n", i)
			broadcasted_dims[i] = 1
			result_shape[i] = dim1
		} else {
			panic("Something went wrong during broadcasting")
		}
	}
	return result_shape, broadcasted_dims
}

func (tensor *Tensor[T]) Broadcast(shape ...types.Dim) *Tensor[T] {
	// tries to broadcast the shape and replicate the data accordingly
	if Equal_1D_slices(tensor.shape, shape) {
		return tensor
	}

	broadcastedShape, _ := BroadcastShapes(tensor.shape, shape)

	var shapeProd types.Dim = 1 // new number of elements
	for _, dim := range broadcastedShape {
		shapeProd *= dim
	}

	outTensor := CreateEmptyTensor[T](broadcastedShape...)
	if tensor.hasFlag(SameValuesFlag) {
		outTensor.setFlag(SameValuesFlag)
		outTensor.Fill(tensor.data[0])
		return outTensor
	}

	// shape diff
	// compares shapes and sets 1 if the dim is changed
	shape_diff := make([]int, len(broadcastedShape))
	for i := 0; i < len(broadcastedShape); i++ {
		j := len(broadcastedShape) - i - 1
		if len(tensor.shape)-i > 0 {
			if broadcastedShape[j] != tensor.shape[len(tensor.shape)-i-1] {
				shape_diff[j] = 1
			}
		} else {
			shape_diff[j] = 1
		}
	}
	// fmt.Println("shape_diff", shape_diff)

	// repeat data to fill broadcasted dims
	sub_index := make([]int, len(tensor.shape))
	repeat := 1
	is_innermost_broadcast := true
	for i := len(shape_diff) - 1; i >= 0; i-- {
		if shape_diff[i] != 1 {
			// if shape_diff[i] == 0 that means that broadsacting
			// will be applied to sub tensors
			is_innermost_broadcast = false
			continue
		}
		repeat *= int(broadcastedShape[i])
		if i > 0 && shape_diff[i-1] == 1 {
			// if shape_diff[i-1] == 1 previous (inner) dim was changed
			continue
		}
		// fmt.Println("repeat", repeat)

		if is_innermost_broadcast {
			is_innermost_broadcast = false
			for i, val := range tensor.data {
				for j := 0; j < repeat; j++ {
					outTensor.data[i*repeat+j] = val
				}
			}
			continue
		}

		var sub *Tensor[T] = tensor
		_sub_index := sub_index[:i]

		if i > 0 {
			sub = tensor.Index(_sub_index...)
		}
		for j := 0; j < repeat; j++ { // repeat
			start := j * len(sub.data)
			end := len(sub.data) * (j + 1)
			copy(outTensor.data[start:end], sub.data)
		}

		repeat = 1
	}
	outTensor.clearFlag(SameValuesFlag)
	return outTensor
}
