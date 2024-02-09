package internal

import (
	"gograd/tensor/types"
	"sync"
)

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
	astrides []int, ashape types.Shape,
) {
	i := 0
	traverse(a, out, astrides, ashape, 0, 0, &i)
}

func TraverseAsContiguous2D[T types.TensorType](a, out []T, ashape types.Shape) {
	var wg sync.WaitGroup
	rows := int(ashape[0])
	cols := int(ashape[1])
	for j := 0; j < cols; j++ {
		wg.Add(1)
		go func(j int) {
			defer wg.Done()
			j_rows := j * rows
			for i := 0; i < rows; i++ {
				out[i*cols+j] = a[j_rows+i]
			}
		}(j)
	}
	wg.Wait()
}
