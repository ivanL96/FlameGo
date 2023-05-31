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

func BroadcastShapes(shape_a, shape_b types.Shape) types.Shape {
	if IsScalarLike(shape_a) && IsScalarLike(shape_b) || Equal_1D_slices(shape_a, shape_b) {
		return shape_a
	}
	if len(shape_a) < len(shape_b) {
		ones_size := len(shape_b) - len(shape_a)
		shape_a = addLeftPadding(shape_a, ones_size, 1)
	} else if len(shape_b) < len(shape_a) {
		ones_size := len(shape_a) - len(shape_b)
		shape_b = addLeftPadding(shape_b, ones_size, 1)
	}
	// start from the trailing dimensions
	result_shape := make(types.Shape, len(shape_a))
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

// tries to broadcast the shape and replicate the data accordingly
func (tensor *Tensor[T]) Broadcast(shape ...types.Dim) *Tensor[T] {
	// TODO test with transpose
	if Equal_1D_slices(tensor.shape, shape) {
		return tensor
	}

	broadcastedShape := BroadcastShapes(tensor.shape, shape)

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
	last_repeat := 1
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

		// TODO test for other shapes
		if is_innermost_broadcast {
			is_innermost_broadcast = false
			for i, val := range tensor.data {
				for j := 0; j < repeat; j++ {
					outTensor.data[i*repeat+j] = val
				}
			}
			last_repeat = repeat * len(tensor.data)
			repeat = 1
			continue
		}

		if last_repeat == 1 {
			// if last_repeat == 1 this is the first time we need to broadcast,
			// so we take a sub tensor using Index() from original tensor and repeat it.
			// sub: [1,2,3], repeat 2 => [1,2,3,1,2,3]
			var sub *Tensor[T] = tensor

			if i > 0 {
				sub = tensor.Index(sub_index[:i]...)
			}
			// fmt.Println("sub what is it ", sub.ToString())
			for j := 0; j < repeat; j++ { // repeat
				start := j * len(sub.data)
				end := len(sub.data) * (j + 1)
				copy(outTensor.data[start:end], sub.data)
			}
		} else {
			// if broadcasting already happened in inner dimensions,
			// that means outTensor contains repeated data, we just repeat is again with the current factor
			sub_data := outTensor.data[:last_repeat]
			for j := 1; j < repeat; j++ { // repeat
				start := j * len(sub_data)
				end := len(sub_data) * (j + 1)
				copy(outTensor.data[start:end], sub_data)
			}
		}

		last_repeat = repeat
		repeat = 1
	}
	outTensor.clearFlag(SameValuesFlag)
	return outTensor
}
