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

// input A and B, both n by n matrices
// initialize C to be an n by n matrix of all zeros
// for i from 1 to n:
//
//	for j from 1 to n:
//	    for k from 1 to n:
//	        C[i][j] = C[i][j] + A[i][k]*B[k][j]
//
// output C (as A*B)
func MatMulImplSimple[T types.TensorType](
	a_data,
	b_data []T,
	a_shape,
	b_shape types.Shape,
	a_strides []int,
	b_strides []int,
	out_data []T,
	out_strides []int,
) {
	a_dim0 := int(a_shape[0])
	a_dim1 := int(a_shape[1])
	b_dim1 := int(b_shape[1])
	for i := 0; i < a_dim0; i++ {
		for j := 0; j < b_dim1; j++ {
			for k := 0; k < a_dim1; k++ {
				out_idx := get_flat_idx_fast(out_strides, i, j)
				// out_strides[0]
				a_idx := get_flat_idx_fast(a_strides, i, k)
				b_idx := get_flat_idx_fast(b_strides, k, j)
				av := a_data[a_idx]
				bv := b_data[b_idx]
				out_data[out_idx] += av * bv
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
