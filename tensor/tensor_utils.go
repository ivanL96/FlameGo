package tensor

import (
	types "gograd/tensor/types"
	"reflect"
)

func squeeze_shape(shape types.Shape) types.Shape {
	result := types.Shape{1}
	for _, v := range shape {
		if v > 1 {
			result = append(result, v)
		}
	}
	return result
}

func IsScalarLike(shape types.Shape) bool {
	if len(shape) == 1 && shape[0] == 1 {
		return true
	}
	return false
}

func getStrides(shape types.Shape) []int {
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

func initDimOrder(shape types.Shape) []uint16 {
	if len(shape) == 1 {
		return []uint16{0}
	}
	dimOrder := make([]uint16, len(shape))
	for i := range dimOrder {
		dimOrder[i] = uint16(i)
	}
	return dimOrder
}

// if dim order is not shuffled
func isDimOrderInit(dimOrder []uint16) bool {
	var min uint16 = 0
	for _, dim := range dimOrder {
		if dim > min {
			return false
		}
		min += 1
	}
	return true
}

func isIntKind(tensorDType reflect.Type) bool {
	switch tensorDType.Kind() {
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64,
		reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return true
	default:
		return false
	}
}

func PrepareOutTensor[T types.TensorType](out *Tensor[T], shape types.Shape) *Tensor[T] {
	if out == nil {
		return InitEmptyTensor[T](shape...)
	}
	return out
}

func SplitTensor[T types.TensorType](
	tensor, outA, outB, outC, outD *Tensor[T],
) (a, b, c, d *Tensor[T]) {
	if len(tensor.shape) != 2 {
		panic("Tensor must have (N,N) shape")
	}
	// only 2-dim, squared matrices
	cols, rows := tensor.shape[0], tensor.shape[1]
	row2, col2 := rows/2, cols/2
	subTensorShape := types.Shape{row2, col2}
	a = PrepareOutTensor(outA, subTensorShape)
	b = PrepareOutTensor(outB, subTensorShape)
	c = PrepareOutTensor(outC, subTensorShape)
	d = PrepareOutTensor(outD, subTensorShape)
	iter := tensor.CreateIterator()
	icol2, irow2 := int(col2), int(row2)
	for iter.Iterate() {
		idx := iter.Next()
		idx0 := idx[0]
		idx1 := idx[1]
		value := tensor.get_fast(idx0, idx1)
		if idx0 < irow2 && idx1 < icol2 {
			a.data[a.get_flat_idx_fast(idx0, idx1)] = value
		} else if idx0 < irow2 && idx1 >= icol2 {
			if idx1 >= icol2 {
				idx1 -= icol2
			}
			b.data[b.get_flat_idx_fast(idx0, idx1)] = value
		} else if idx0 >= irow2 && idx1 < icol2 {
			if idx0 >= irow2 {
				idx0 -= irow2
			}
			c.data[c.get_flat_idx_fast(idx0, idx1)] = value
		} else if idx0 >= irow2 && idx1 >= icol2 {
			if idx0 >= irow2 {
				idx0 -= irow2
			}
			if idx1 >= icol2 {
				idx1 -= icol2
			}
			d.data[d.get_flat_idx_fast(idx0, idx1)] = value
		}
	}
	return
}

