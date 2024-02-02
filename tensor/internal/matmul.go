package internal

import (
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

// func MatMulNaiveImpl_GEN(
// 	impl device.Implementation,
// 	a_data,
// 	b_data []float32,
// 	a_shape,
// 	b_shape types.Shape,
// 	a_strides []int,
// 	b_strides []int,
// 	out_data []float32,
// 	out_strides []int,
// ) {
// 	a_dim0 := int(a_shape[0])
// 	b_dim0 := int(b_shape[0])
// 	out_stride0 := out_strides[0]
// 	a_stride0 := a_strides[0]
// 	b_stride0 := b_strides[0]

// 	runtime.GOMAXPROCS(numCPU)

// 	var wg sync.WaitGroup
// 	for i := 0; i < a_dim0; i++ {
// 		wg.Add(1)
// 		go func(i int) {
// 			defer wg.Done()
// 			// Set affinity to a specific CPU core
// 			runtime.LockOSThread()
// 			defer runtime.UnlockOSThread()

// 			out_stride0_i := out_stride0 * i
// 			_a := a_data[a_stride0*i : a_stride0*(i+1)]
// 			for j := 0; j < b_dim0; j++ {
// 				_b := b_data[b_stride0*j : b_stride0*(j+1)]
// 				out_data[out_stride0_i+j] = device.Dot(impl, _a, _b)
// 			}
// 		}(i)
// 	}
// 	wg.Wait()
// }

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
