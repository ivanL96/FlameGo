package tensor

import (
	types "gograd/tensor/types"
)

// tries to broadcast the shape and replicate the data accordingly
func (tensor *Tensor[T]) Broadcast(shape ...types.Dim) *Tensor[T] {
	// TODO test with transpose
	if tensor.shape.Equals(shape) {
		return tensor
	}

	broadcastedShape := tensor.shape.BroadcastShapes(shape)

	outTensor := CreateEmptyTensor[T](broadcastedShape...)

	// shape diff
	// compares shapes and sets 1 if the dim is changed
	tensor_dims := len(tensor.shape)
	shape_diff := make([]int, len(broadcastedShape))
	for i := 0; i < len(broadcastedShape); i++ {
		j := len(broadcastedShape) - i - 1
		if tensor_dims-i > 0 {
			if broadcastedShape[j] != tensor.shape[tensor_dims-i-1] {
				shape_diff[j] = 1
			}
		} else {
			shape_diff[j] = 1
		}
	}
	// fmt.Println("shape_diff", shape_diff)

	// repeat data to fill broadcasted dims
	sub_index := make([]int, tensor_dims)
	repeat, last_repeat := 1, 1
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

		// TODO test for other shapes
		if is_innermost_broadcast {
			is_innermost_broadcast = false
			for i, val := range tensor.data() {
				for j := 0; j < repeat; j++ {
					outTensor.data()[i*repeat+j] = val
				}
			}
			last_repeat = repeat * len(tensor.data())
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
				start := j * len(sub.data())
				end := len(sub.data()) * (j + 1)
				copy(outTensor.data()[start:end], sub.data())
			}
		} else {
			// if broadcasting already happened in inner dimensions,
			// that means outTensor contains repeated data, we just repeat is again with the current factor
			sub_data := outTensor.data()[:last_repeat]
			for j := 1; j < repeat; j++ { // repeat
				start := j * len(sub_data)
				end := len(sub_data) * (j + 1)
				copy(outTensor.data()[start:end], sub_data)
			}
		}

		last_repeat = repeat
		repeat = 1
	}
	return outTensor
}
