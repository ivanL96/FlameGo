package ops

import (
	"gograd/tensor/intrinsics/cpu"
	"gograd/tensor/intrinsics/noasm"
	"gograd/tensor/types"
)

// Naive implementation of matrix multiplication with time complexity n^3
// input A and B, both n by n matrices
// initialize C to be an n by n matrix of all zeros
//
//	 for i from 1 to n:
//		 for j from 1 to n:
//		    for k from 1 to n:
//		        C[i][j] = C[i][j] + A[i][k]*B[k][j]
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
	// TODO make indexing transpose invariant
	a_dim0 := int(a_shape[0])
	a_dim1 := int(a_shape[1])
	b_dim1 := int(b_shape[1])
	out_stride0 := out_strides[0]
	a_stride0 := a_strides[0]
	b_stride0 := b_strides[0]
	for i := 0; i < a_dim0; i++ {
		a_stride0_i := a_stride0 * i
		out_stride0_i := out_stride0 * i
		for j := 0; j < b_dim1; j++ {
			var out_val T
			out_idx := out_stride0_i + j
			for k := 0; k < a_dim1; k++ {
				out_val += a_data[a_stride0_i+k] * b_data[b_stride0*k+j]
			}
			out_data[out_idx] = out_val
		}
	}
}

func MatMulNaiveImpl_AVX(
	a_data,
	b_data []float32,
	a_shape,
	b_shape types.Shape,
	a_strides []int,
	b_strides []int,
	out_data []float32,
	out_strides []int,
) {
	a_dim0 := int(a_shape[0])
	out_stride0 := out_strides[0]
	a_stride0 := a_strides[0]
	b_stride0 := b_strides[0]
	for i := 0; i < a_dim0; i++ {
		a_stride0_i := a_stride0 * i
		a_stride0_i_end := a_stride0 * (i + 1)
		out_stride0_i := out_stride0 * i
		for j := 0; j < a_dim0; j++ {
			out_val := cpu.AVX.Dot(
				a_data[a_stride0_i:a_stride0_i_end],
				b_data[b_stride0*j:b_stride0*(j+1)],
			)
			out_data[out_stride0_i+j] = out_val
		}
	}
}

func MatMul_AVX_VectorsToScalar(
	a_data,
	b_data,
	out_data []float32,
) {
	out_data[0] = cpu.AVX.Dot(a_data, b_data)
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

// Strassen's matmul implementation with sub-cubic time complexity
// insanely slow compared to Naive impl
func MatMulStrassen[T types.TensorType](a_data, b_data []T, a_shape types.Shape) []T {
	if a_shape[0] == 1 {
		return []T{a_data[0] * b_data[0]}
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

	var outMat []T
	// outMat := make([]T, len(a))

	p1 := MatMulStrassen(a, noasm.SubMatx(f, h, outMat), sub_shape)
	p2 := MatMulStrassen(noasm.AddMatx(a, b, outMat), h, sub_shape)
	p3 := MatMulStrassen(noasm.AddMatx(c, d, outMat), e, sub_shape)
	p4 := MatMulStrassen(d, noasm.SubMatx(g, e, outMat), sub_shape)
	p5 := MatMulStrassen(noasm.AddMatx(a, d, outMat), noasm.AddMatx(e, h, outMat), sub_shape)
	p6 := MatMulStrassen(noasm.SubMatx(b, d, outMat), noasm.AddMatx(g, h, outMat), sub_shape)
	p7 := MatMulStrassen(noasm.SubMatx(a, c, outMat), noasm.AddMatx(e, f, outMat), sub_shape)
	c11 := noasm.AddMatx(noasm.SubMatx(noasm.AddMatx(p5, p4, nil), p2, nil), p6, nil)
	c12 := noasm.AddMatx(p1, p2, nil)
	c21 := noasm.AddMatx(p3, p4, nil)
	c22 := noasm.SubMatx(noasm.SubMatx(noasm.AddMatx(p1, p5, nil), p3, nil), p7, nil)

	out := make([]T, len(a_data))
	UniteTensors(int(sub_shape[0]), int(sub_shape[1]), c11, c12, c21, c22, out)
	if original_shape[0]%2 == 1 {
		out = RemovePaddingMat(out, a_shape, 0, 1)
	}
	return out
}
