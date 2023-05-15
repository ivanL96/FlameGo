package tensor

import (
	"fmt"
)

func Log(value ...interface{}) {
	str := fmt.Sprintf("%x", value)
	fmt.Println("Debug: " + str)
}

func squeeze_shape(shape Shape) Shape {
	result := Shape{1}
	for _, v := range shape {
		if v > 1 {
			result = append(result, v)
		}
	}
	return result
}

func isScalarLike(shape Shape) bool {
	if len(shape) <= 1 && shape[0] <= 1 {
		return true
	}
	return false
}

func are_broadcastable(shape_a, shape_b Shape) bool {
	if (isScalarLike(shape_a) && isScalarLike(shape_b)) || Equal_1D_slices(shape_a, shape_b) {
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

func getStrides(shape Shape) []int {
	strides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= int(shape[i])
	}
	return strides
}
