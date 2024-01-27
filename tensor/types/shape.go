package types

func (shape Shape) Squeeze() Shape {
	result := Shape{1}
	for _, v := range shape {
		if v > 1 {
			result = append(result, v)
		}
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
