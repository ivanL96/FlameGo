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

func MatMulStrassen[T types.TensorType](a_data, b_data []T, a_shape types.Shape) []T {
	if a_shape[0] == 1 {
		return MulMatx(a_data, b_data, nil)
	}

	original_shape := a_shape
	if original_shape[0]%2 == 1 {
		a_data_pad, a_shape_pad := PaddingMat(a_data, a_shape, 0, 1)
		b_data, _ = PaddingMat(b_data, a_shape, 0, 1)
		a_data = a_data_pad
		a_shape = a_shape_pad
	}
	nrows := int(a_shape[0])

	subsize := (nrows / 2) * (nrows / 2)
	a, b, c, d := make([]T, subsize), make([]T, subsize), make([]T, subsize), make([]T, subsize)
	e, f, g, h := make([]T, subsize), make([]T, subsize), make([]T, subsize), make([]T, subsize)
	SplitTensorImpl(a_data, nrows, a, b, c, d)
	SplitTensorImpl(b_data, nrows, e, f, g, h)
	sub_shape := types.Shape{a_shape[0] / 2, a_shape[1] / 2}

	p1 := MatMulStrassen(a, SubMatx(f, h, nil), sub_shape)
	p2 := MatMulStrassen(AddMatx(a, b, nil), h, sub_shape)
	p3 := MatMulStrassen(AddMatx(c, d, nil), e, sub_shape)
	p4 := MatMulStrassen(d, SubMatx(g, e, nil), sub_shape)
	p5 := MatMulStrassen(AddMatx(a, d, nil), AddMatx(e, h, nil), sub_shape)
	p6 := MatMulStrassen(SubMatx(b, d, nil), AddMatx(g, h, nil), sub_shape)
	p7 := MatMulStrassen(SubMatx(a, c, nil), AddMatx(e, f, nil), sub_shape)
	c11 := AddMatx(SubMatx(AddMatx(p5, p4, nil), p2, nil), p6, nil)
	c12 := AddMatx(p1, p2, nil)
	c21 := AddMatx(p3, p4, nil)
	c22 := SubMatx(SubMatx(AddMatx(p1, p5, nil), p3, nil), p7, nil)

	out := make([]T, len(a_data))
	UniteTensors(int(sub_shape[0]), int(sub_shape[1]), c11, c12, c21, c22, out)
	if original_shape[0]%2 == 1 {
		out = RemovePaddingMat(out, a_shape, 0, 1)
	}
	return out
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
	// only 2-dim, squared matrices with even dims
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

func PaddingMat[T types.TensorType](data []T, shape types.Shape, pad_before, pad_after uint) ([]T, types.Shape) {
	if len(shape) != 2 {
		panic("Shape must be 2-dim")
	}
	pad_shape := make(types.Shape, len(shape))
	pad_shape_prod := 1
	shape_prod := 1
	for i, dim := range shape {
		out_dim := dim + types.Dim(pad_before+pad_after)
		pad_shape[i] = out_dim
		pad_shape_prod *= int(out_dim)
		shape_prod *= int(dim)
	}

	pad_data := make([]T, pad_shape_prod)
	before := 0
	pad_outermost_dim := pad_shape_prod / int(pad_shape[0])
	if pad_before > 0 {
		// skip rows that are padded
		before = pad_outermost_dim * int(pad_before)
	}
	stride := shape_prod / int(shape[0])
	offset := 0
	for i := 0; i < int(shape[0]); i++ {
		row := data[i*stride : stride*(i+1)]
		if i > 0 {
			offset += int(pad_before + pad_after)
		}
		pad_data_start := before + int(pad_before) + stride*i + offset
		pad_data_end := before + int(pad_before) + stride*(i+1) + offset

		copy(pad_data[pad_data_start:pad_data_end], row)
	}
	return pad_data, pad_shape
}

func RemovePaddingMat[T types.TensorType](data []T, shape types.Shape, pad_before, pad_after uint) []T {
	shape_prod := 1
	unpadded_shape := make(types.Shape, len(shape))
	for i, dim := range shape {
		unpdim := int(dim) - int(pad_before+pad_after)
		shape_prod *= unpdim
		unpadded_shape[i] = types.Dim(unpdim)
	}
	out := make([]T, shape_prod)
	row_len := int(unpadded_shape[1])
	skip_rows_before := int(pad_before) * int(shape[1])
	offset := 0
	for i := 0; i < int(unpadded_shape[0]); i++ {
		start := row_len*i + int(pad_before) + skip_rows_before + offset
		end := row_len*(i+1) + int(pad_before) + skip_rows_before + offset
		offset += int(pad_before + pad_after)
		copy(out[row_len*i:row_len*(i+1)], data[start:end])
	}
	return out
}
