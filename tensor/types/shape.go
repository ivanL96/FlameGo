package types

import (
	"errors"
	"fmt"
)

// adds pad value at the beginning of the slice.
// example: addLeftPadding([1,2,3], 4, 0) ==> [0,0,0,0,1,2,3]
func addLeftPadding[T TensorType](slice []T, padding_size int, padding_val T) []T {
	if padding_size == 0 {
		return slice
	}
	expandedSlice := make([]T, len(slice)+padding_size)
	for i := 0; i < padding_size; i++ {
		if i == padding_size {
			break
		}
		if padding_val != 0 {
			expandedSlice[i] = padding_val
		}
	}
	copy(expandedSlice[padding_size:], slice)
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

// works like squeeze but ignores edge dims
func (shape Shape) SqueezeInner() Shape {
	result := make(Shape, 0, len(shape))
	for i, v := range shape {
		if v > 1 || i == 0 || i == len(shape)-1 {
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

// Compares two Shapes. If each dimension is equal to other shape dimension at the same position
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

func (shape Shape) ReduceDim(axes ...int) Shape {
	if len(axes) == 0 {
		return shape
	}
	new_shape := append(Shape{}, shape...)
	for _, axis := range axes {
		new_shape[axis] = 1
	}
	return new_shape
}

// stacks shapes together by given axis. rest of dims must be the same:
// Stacked by axis 0 (1,2,3), (2,2,3), (4,2,3) => (7,2,3)
func StackShapes(axis int, other_shapes ...Shape) (Shape, error) {
	if len(other_shapes) == 0 {
		return nil, errors.New("at least 1 shape must be set")
	}
	result := make(Shape, len(other_shapes[0]))
	copy(result, other_shapes[0])
	prev_shape := other_shapes[0]

	for i := 1; i < len(other_shapes); i++ {
		sh := other_shapes[i]
		result[axis] += sh[axis]
		for j := 0; j < len(sh); j++ {
			if j == axis {
				continue
			}
			if sh[j] != prev_shape[j] {
				return nil, fmt.Errorf("shapes %v and %v cannot be stacked together", sh, prev_shape)
			}
		}
		prev_shape = sh
	}
	return result, nil
}
