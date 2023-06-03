package ops

import "flamego/tensor/types"

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
// assumed all tensors are squares and have the same shapes
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
