package internal

import (
	"gograd/tensor/types"
	"runtime"
	"sync"
)

var numCPU int = runtime.NumCPU()

func traverse[T types.TensorType](
	a, out []T,
	astrides []int, ashape types.Shape,
	axis, step int, flat *int,
) {
	stride := astrides[axis]
	dim := int(ashape[axis])

	if axis+1 == len(astrides) {
		for i := 0; i < dim; i++ {
			out[*flat] = a[step+i*stride]
			*flat += 1
		}
		return
	}
	for i := 0; i < dim; i++ {
		traverse(a, out, astrides, ashape, axis+1, step+i*stride, flat)
	}
}

func TraverseAsContiguous[T types.TensorType](a, out []T,
	a_strides []int, a_shape types.Shape,
) {
	i := 0
	traverse(a, out, a_strides, a_shape, 0, 0, &i)
}

func TraverseAsContiguous2D[T types.TensorType](a, out []T, ashape types.Shape) {
	var wg sync.WaitGroup
	rows := int(ashape[0])
	cols := int(ashape[1])

	chunk_size := (cols + numCPU - 1) / numCPU
	wg.Add(numCPU)

	for i := 0; i < numCPU; i++ {
		start := i * chunk_size
		end := (i + 1) * chunk_size
		if end > cols {
			end = cols
		}
		go func(start, end int) {
			defer wg.Done()
			for j := start; j < end; j++ {
				Transpose_cont2D_loop(a, out, j, rows, cols)
			}
		}(start, end)
	}
	wg.Wait()
}

func parallel[T types.TensorType](
	f func(int, int, []T, []T, []T, *sync.Mutex),
	a, b, out []T,
) {
	la := len(a)
	if len(a) == 1 {
		la = len(b)
	}
	chunk_size := (la + numCPU - 1) / numCPU

	var wg sync.WaitGroup
	var mu sync.Mutex
	wg.Add(numCPU)

	for i := 0; i < numCPU; i++ {
		start := i * chunk_size
		end := (i + 1) * chunk_size
		if end > la {
			end = la
		}

		go func(start, end int) {
			defer wg.Done()
			// runtime.LockOSThread()
			// defer runtime.UnlockOSThread()
			f(start, end, a, b, out, &mu)
		}(start, end)
	}
	wg.Wait()
}

// sets specific value to []T buffer using loop unrolling opt.
func Fill_data_loop[T types.TensorType](buffer []T, value T) {
	lb := len(buffer)
	for i := 0; i < lb/8; i += 8 {
		buffer[i] = value
		buffer[i+1] = value
		buffer[i+2] = value
		buffer[i+3] = value
		buffer[i+4] = value
		buffer[i+5] = value
		buffer[i+6] = value
		buffer[i+7] = value
	}
	for i := lb - lb%8; i < lb; i++ {
		buffer[i] = value
	}
}

func Convert_type_loop[OLD_T, NEW_T types.TensorType](data []OLD_T, out_data []NEW_T) {
	lb := len(data)
	n := 8
	for i := 0; i < lb/n; i += n {
		out_data[i] = NEW_T(data[i])
		out_data[i+1] = NEW_T(data[i+1])
		out_data[i+2] = NEW_T(data[i+2])
		out_data[i+3] = NEW_T(data[i+3])
		out_data[i+4] = NEW_T(data[i+4])
		out_data[i+5] = NEW_T(data[i+5])
		out_data[i+6] = NEW_T(data[i+6])
		out_data[i+7] = NEW_T(data[i+7])
	}
	for i := lb - lb%n; i < lb; i++ {
		out_data[i] = NEW_T(data[i])
	}
}

func Transpose_cont2D_loop[T types.TensorType](data, transposed []T, i, cols, rows int) {
	i_cols := i * cols
	for j := 0; j < cols/8; j += 8 {
		i_cols_j := i_cols + j
		transposed[j*rows+i] = data[i_cols_j]
		transposed[(j+1)*rows+i] = data[i_cols_j+1]
		transposed[(j+2)*rows+i] = data[i_cols_j+2]
		transposed[(j+3)*rows+i] = data[i_cols_j+3]

		transposed[(j+4)*rows+i] = data[i_cols_j+4]
		transposed[(j+5)*rows+i] = data[i_cols_j+5]
		transposed[(j+6)*rows+i] = data[i_cols_j+6]
		transposed[(j+7)*rows+i] = data[i_cols_j+7]
	}
	for j := cols - cols%8; j < cols; j++ {
		transposed[j*rows+i] = data[i_cols+j]
	}
}
