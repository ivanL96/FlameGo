package ops

import (
	"gograd/tensor/types"
)

// Naive implementation of matrix multiplication with time complexity n^3
// input A and B, both n by n matrices
// initialize C to be an n by n matrix of all zeros
// for i from 1 to n:
//
//	for j from 1 to n:
//	    for k from 1 to n:
//	        C[i][j] = C[i][j] + A[i][k]*B[k][j]
//
// output C (as A*B)
func MatMulNaiveImpl[T types.TensorType](
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
	out_stride0 := out_strides[0]
	a_stride0 := a_strides[0]
	b_stride0 := b_strides[0]
	for i := 0; i < a_dim0; i++ {
		a_stride0_i := a_stride0 * i
		out_stride0_i := out_stride0 * i
		a_idx := a_stride0_i + 1
		for j := 0; j < b_dim1; j++ {
			// k=0
			out_idx := out_stride0_i + j
			out_data[out_idx] += a_data[a_stride0_i] * b_data[j]
			// k=1
			b_idx := b_stride0 + j
			out_data[out_idx] += a_data[a_idx] * b_data[b_idx]
			for k := 2; k < a_dim1; k++ {
				out_data[out_idx] += a_data[a_stride0_i+k] * b_data[b_stride0*k+j]
			}
		}
	}
}

// matMul for square matrices with the same shape: (N,N)
func MatMulSquareNaiveImpl[T types.TensorType](
	a_data,
	b_data []T,
	a_shape types.Shape,
	a_strides []int,
	out_data []T,
) {
	a_dim0 := int(a_shape[0])
	a_stride0 := a_strides[0]
	for i := 0; i < a_dim0; i++ {
		a_stride0_i := a_stride0 * i
		out_stride0_i := a_stride0 * i
		a_idx := a_stride0_i + 1
		for j := 0; j < a_dim0; j++ {
			// k=0
			out_idx := out_stride0_i + j
			out_data[out_idx] += a_data[a_stride0_i] * b_data[j]
			//k=1
			b_idx := a_stride0 + j
			out_data[out_idx] += a_data[a_idx] * b_data[b_idx]
			for k := 2; k < a_dim0; k++ {
				out_data[out_idx] += a_data[a_stride0_i+k] * b_data[a_stride0*k+j]
			}
		}
	}
}

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

// unites subtensors splitted by SplitTensor
// assumed all tensors are square and have same shapes
func UniteTensors[T types.TensorType](
	sub_dim,
	sub_stride int,
	a_data,
	b_data,
	c_data,
	d_data,
	out_data []T,
) {
	outDim := sub_dim * 2
	out_stride := sub_stride * 2

	for i := 0; i < int(outDim); i++ {
		start, end := i*out_stride, out_stride*(i+1)
		if i < sub_dim {
			copy(out_data[start:start+sub_stride], a_data[i*sub_stride:(i+1)*sub_stride])
			copy(out_data[start+sub_stride:end], b_data[i*sub_stride:(i+1)*sub_stride])
		} else {
			j := i - sub_dim
			copy(out_data[start:start+sub_stride], c_data[j*sub_stride:(j+1)*sub_stride])
			copy(out_data[start+sub_stride:end], d_data[j*sub_stride:(j+1)*sub_stride])
		}
	}
}
