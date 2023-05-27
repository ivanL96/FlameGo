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
	rows, cols := tensor.shape[0], tensor.shape[1]
	row2, col2 := int(rows/2), int(cols/2)
	sub_tensor_shape := types.Shape{types.Dim(row2), types.Dim(col2)}
	a = PrepareOutTensor(outA, sub_tensor_shape)
	b = PrepareOutTensor(outB, sub_tensor_shape)
	c = PrepareOutTensor(outC, sub_tensor_shape)
	d = PrepareOutTensor(outD, sub_tensor_shape)
	iter := tensor.CreateIterator()
	for iter.Iterate() {
		idx := iter.Next()
		idx0 := idx[0]
		idx1 := idx[1]
		value := tensor.get_fast(idx0, idx1)
		if idx0 < row2 && idx1 < col2 {
			a.data[a.get_flat_idx_fast(idx0, idx1)] = value
		} else if idx0 < row2 && idx1 >= col2 {
			if idx1 >= col2 {
				idx1 -= col2
			}
			b.data[b.get_flat_idx_fast(idx0, idx1)] = value
		} else if idx0 >= row2 && idx1 < col2 {
			if idx0 >= row2 {
				idx0 -= row2
			}
			c.data[c.get_flat_idx_fast(idx0, idx1)] = value
		} else if idx0 >= row2 && idx1 >= col2 {
			if idx0 >= row2 {
				idx0 -= row2
			}
			if idx1 >= col2 {
				idx1 -= col2
			}
			d.data[d.get_flat_idx_fast(idx0, idx1)] = value
		}
	}
	return
}

func SplitTensor2[T types.TensorType](
	tensor, outA, outB, outC, outD *Tensor[T],
) (a, b, c, d *Tensor[T]) {
	if len(tensor.shape) != 2 {
		panic("Tensor must have (N,N) shape")
	}
	// only 2-dim, squared matrices
	rows := int(tensor.shape[0])
	row2 := rows / 2
	sub_tensor_shape := types.Shape{types.Dim(row2), types.Dim(row2)}
	a = PrepareOutTensor(outA, sub_tensor_shape)
	b = PrepareOutTensor(outB, sub_tensor_shape)
	c = PrepareOutTensor(outC, sub_tensor_shape)
	d = PrepareOutTensor(outD, sub_tensor_shape)
	// assume continuous data
	// TODO parallel exec
	astride, bstride, cstride, dstride := 0, 0, 0, 0
	for i := 0; i < int(rows)*2; i++ {
		row := tensor.data[row2*i : row2*(i+1)]
		switch i % 2 {
		case 0:
			if i < rows {
				step := astride + row2
				copy(a.data[astride:step], row)
				astride = step
			} else if i >= rows {
				step := cstride + row2
				copy(c.data[cstride:step], row)
				cstride = step
			}
		case 1:
			if i < rows {
				step := bstride + row2
				copy(b.data[bstride:step], row)
				bstride = step
			} else if i >= rows {
				step := dstride + row2
				copy(d.data[dstride:step], row)
				dstride = step
			}
		}
	}
	return
}

// TODO
// unites subtensors splitted by SplitTensor
// func UniteTensors[T types.TensorType](a, b, c, d *Tensor[T]) *Tensor[T] {
// 	outShape := a.shape
// 	outShape[0] *= 2
// 	outShape[1] *= 2
// 	outTensor := InitEmptyTensor[T](outShape...)

// 	for i := 0; i < int(outShape[0]); i++ {
// 		if i < a.shape
// 	}
// }
