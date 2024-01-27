package types

import "fmt"

// adds pad value at the beginning of the slice.
// example: addLeftPadding([1,2,3], 4, 0) ==> [0,0,0,0,1,2,3]
func addLeftPadding[T TensorType](slice []T, padding_size int, padding_val T) []T {
	if padding_size == 0 {
		return slice
	}
	expandedSlice := make([]T, len(slice)+padding_size)
	for i := range expandedSlice {
		if i < padding_size {
			if padding_val != 0 {
				expandedSlice[i] = padding_val
			}
			continue
		}
		expandedSlice[i] = slice[i-padding_size]
	}
	return expandedSlice
}

func (shape Shape) Squeeze() Shape {
	result := make(Shape, 0, len(shape))
	for _, v := range shape {
		if v > 1 {
			result = append(result, v)
		}
	}
	if len(result) == 0 {
		return Shape{1}
	}
	return result
}

// Adds a new dimension for given axis:
//
// example: AddDim(0) for shape (4,2,3) returns (1,4,2,3)
//
// or: AddDim(1) for shape (4,2,3) returns (4,1,2,3)
func (shape Shape) AddDim(axis uint) Shape {
	newShape := append(shape[:axis], append(Shape{1}, shape[axis:]...)...)
	return newShape
}

func (shape Shape) Equals(other_shape Shape) bool {
	if len(shape) != len(other_shape) {
		return false
	}
	for i, dim := range shape {
		other_dim := other_shape[i]
		if dim != other_dim {
			return false
		}
	}
	return true
}

func (shape Shape) AreBroadcastable(other Shape) bool {
	if (shape.IsScalarLike() && other.IsScalarLike()) ||
		shape.Equals(other) {
		return true
	}
	// If one shape has more dimensions than the other, prepend 1s to the shape of the smaller array
	if len(shape) < len(other) {
		ones_size := len(other) - len(shape)
		shape = addLeftPadding(shape, ones_size, 1)
	} else if len(other) < len(shape) {
		ones_size := len(shape) - len(other)
		other = addLeftPadding(other, ones_size, 1)
	}
	// Start from the trailing dimensions and work forward
	for i := len(shape) - 1; i >= 0; i-- {
		dim1 := shape[i]
		dim2 := other[i]
		if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
			return false
		}
	}
	return true
}

func (shape_a Shape) BroadcastShapes(shape_b Shape) Shape {
	if shape_a.IsScalarLike() && shape_b.IsScalarLike() ||
		shape_a.Equals(shape_b) {
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

// compares two broadcastable shapes
// func (shape Shape) Less(other Shape) bool {
// 	broadcastable := shape.AreBroadcastable(other)
// 	if !broadcastable {
// 		return false
// 	}
// 	for
// }

func (shape Shape) IsScalarLike() bool {
	if len(shape) == 1 && shape[0] == 1 {
		return true
	}
	for _, dim := range shape {
		if dim > 1 {
			return false
		}
	}
	return true
}

func (shape Shape) GetStrides() []int {
	if len(shape) == 1 {
		return []int{1}
	}
	strides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= int(shape[i])
	}
	return strides
}

func (shape Shape) InitDimOrder() []uint16 {
	if len(shape) == 1 {
		return []uint16{0}
	}
	dimOrder := make([]uint16, len(shape))
	for i := range dimOrder {
		dimOrder[i] = uint16(i)
	}
	return dimOrder
}
