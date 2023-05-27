package ops

import (
	"gograd/tensor/types"
)

// copy of the same method from indexing.go
func get_flat_idx_fast(strides []int, indices ...int) int {
	flatIndex := 0
	for i, ind := range indices {
		flatIndex += strides[i] * ind
	}
	return flatIndex
}

func MatMulImplSimple[T types.TensorType](tensor_a, tensor_b, outTensor types.ITensor[T]) {
	for i := 0; i < int(tensor_a.Shape()[0]); i++ {
		for j := 0; j < int(tensor_b.Shape()[1]); j++ {
			for k := 0; k < int(tensor_a.Shape()[1]); k++ {
				idx := get_flat_idx_fast(outTensor.Strides(), i, j)
				outTensor.Data()[idx] += tensor_a.Get_fast(i, k) * tensor_b.Get_fast(k, j)
			}
		}
	}
}

// func SplitTensorImplSlow[T types.TensorType](
// 	tensor types.ITensor[T],
// 	nrows int,
// 	a,
// 	b,
// 	c,
// 	d types.ITensor[T],
// ) (types.ITensor[T], types.ITensor[T], types.ITensor[T], types.ITensor[T]) {
// 	// only 2-dim, squared matrices
// 	row2 := nrows / 2
// 	it := iter.CreateIterator(tensor.Data(), tensor.Shape())
// 	for it.Iterate() {
// 		idx := it.Next()
// 		idx0 := idx[0]
// 		idx1 := idx[1]
// 		value := tensor.Get_fast(idx0, idx1)
// 		if idx0 < row2 && idx1 < row2 {
// 			a.Data()[get_flat_idx_fast(a.Strides(), idx0, idx1)] = value
// 		} else if idx0 < row2 && idx1 >= row2 {
// 			if idx1 >= row2 {
// 				idx1 -= row2
// 			}
// 			b.Data()[get_flat_idx_fast(b.Strides(), idx0, idx1)] = value
// 		} else if idx0 >= row2 && idx1 < row2 {
// 			if idx0 >= row2 {
// 				idx0 -= row2
// 			}
// 			c.Data()[get_flat_idx_fast(c.Strides(), idx0, idx1)] = value
// 		} else if idx0 >= row2 && idx1 >= row2 {
// 			if idx0 >= row2 {
// 				idx0 -= row2
// 			}
// 			if idx1 >= row2 {
// 				idx1 -= row2
// 			}
// 			d.Data()[get_flat_idx_fast(d.Strides(), idx0, idx1)] = value
// 		}
// 	}
// 	return a, b, c, d
// }

// ~30x faster compared to v1
func SplitTensorImpl[T types.TensorType](
	tensor_data []T,
	nrows int,
	a_data,
	b_data,
	c_data,
	d_data []T,
) (a, b, c, d []T) {
	// only 2-dim, squared matrices
	// assume continuous data
	// TODO parallel exec
	row2 := nrows / 2
	astride, bstride, cstride, dstride := 0, 0, 0, 0
	for i := 0; i < int(nrows)*2; i++ {
		row := tensor_data[row2*i : row2*(i+1)]
		switch i % 2 {
		case 0:
			if i < nrows {
				step := astride + row2
				copy(a_data[astride:step], row)
				astride = step
			} else if i >= nrows {
				step := cstride + row2
				copy(c_data[cstride:step], row)
				cstride = step
			}
		case 1:
			if i < nrows {
				step := bstride + row2
				copy(b_data[bstride:step], row)
				bstride = step
			} else if i >= nrows {
				step := dstride + row2
				copy(d_data[dstride:step], row)
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
