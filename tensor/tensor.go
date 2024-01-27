package tensor

import (
	"fmt"
	"gograd/tensor/iter"
	types "gograd/tensor/types"
	"math"
)

// set of primitive common tensor methods
//tensor initialization-----------------------------------------------------

func makeTensor[T types.TensorType](dataPtr *[]T, shape types.Shape) *Tensor[T] {
	var shapeProd types.Dim = 1
	for _, dim := range shape {
		shapeProd *= dim
	}
	var data []T
	if dataPtr == nil {
		// if nil ptr create an empty slice with size of 'shapeProd'
		data = make([]T, shapeProd)
	} else {
		// copies data
		data = append([]T(nil), (*dataPtr)...)
	}
	if len(shape) == 0 || int(shapeProd) != len(data) {
		panic(fmt.Sprintf("makeTensor: Value length %v cannot have shape %v", len(data), shape))
	}
	tensor := &Tensor[T]{
		shape:     append(types.Shape(nil), shape...),
		strides:   shape.GetStrides(),
		data_buff: data,
		dim_order: shape.InitDimOrder(),
	}

	return tensor
}

// inits a tensor with data
func CreateTensor[T types.TensorType](value []T, shape types.Shape) *Tensor[T] {
	return makeTensor(&value, shape)
}

// inits an empty tensor with specific shape
func CreateEmptyTensor[T types.TensorType](shape ...types.Dim) *Tensor[T] {
	return makeTensor[T](nil, shape)
}

func Ones[T types.TensorType](shape ...types.Dim) *Tensor[T] {
	return CreateEmptyTensor[T](shape...).Fill(1)
}

func Zeros[T types.TensorType](shape ...types.Dim) *Tensor[T] {
	return CreateEmptyTensor[T](shape...)
}

// A tensor with Shape 1
func Scalar[T types.TensorType](value T) *Tensor[T] {
	return &Tensor[T]{
		shape:     types.Shape{1},
		strides:   []int{1},
		data_buff: []T{value},
		dim_order: []uint16{0},
	}
}

// Creates a tensor without copying the data
func AsTensor[T types.TensorType](data []T, shape types.Shape) *Tensor[T] {
	t := &Tensor[T]{
		strides:   shape.GetStrides(),
		data_buff: data,
		dim_order: shape.InitDimOrder(),
		shape:     shape,
	}
	return t
}

//-----------------------------------------------------

func AsType[OLDT types.TensorType, NEWT types.TensorType](tensor *Tensor[OLDT]) *Tensor[NEWT] {
	// naive impl with copying the data & tensor
	// example:
	// AsType(int32, float64)(tensor) ==> float64 tensor
	data := make([]NEWT, len(tensor.data()))
	for i, val := range tensor.data() {
		data[i] = NEWT(val)
	}
	return CreateTensor(data, tensor.shape)
}

func (tensor *Tensor[T]) Copy() *Tensor[T] {
	newData := make([]T, len(tensor.data()))
	copy(newData, tensor.data())
	newTensor := CreateTensor(newData, tensor.shape)
	newTensor.strides = tensor.strides
	newTensor.dim_order = tensor.dim_order
	return newTensor
}

// Creates a tensor with data ranged from 'start' to 'end'
// limits: min 1 and max 3 arguments. Start, End, Step
func Range[T types.TensorType](limits ...int) *Tensor[T] {
	if len(limits) == 0 {
		panic("Range requires at least one argument")
	}
	start, end, step := 0, 0, 1
	if len(limits) == 1 {
		end = limits[0]
	}
	if len(limits) >= 2 {
		start = limits[0]
		end = limits[1]
	}
	if len(limits) == 3 {
		step = limits[2]
	}
	length := ((end - start) + step - 1) / step
	tensor := CreateEmptyTensor[T](types.Dim(length))
	for i := 0; i < length; i++ {
		tensor.data()[i] = T(start)
		start += step
	}
	return tensor
}

// Fills tensor with same value
func (tensor *Tensor[T]) Fill(value T) *Tensor[T] {
	data := tensor.data()
	if len(data) >= 12 {
		fill_data_unroll4(&data, value)
	} else {
		for i := range data {
			data[i] = value
		}
	}
	return tensor
}

// creates an array (2d tensor) with ones on the diagonal and zeros elsewhere
func Eye[T types.TensorType](x, y types.Dim) *Tensor[T] {
	eye := CreateEmptyTensor[T](types.Shape{x, y}...)
	for i := 0; i < int(x); i++ {
		for j := 0; j < int(y); j++ {
			if i == j {
				fidx := get_flat_idx_fast(eye.strides, i, j)
				eye.data()[fidx] = 1
			}
		}
	}
	return eye
}

// sets new value of the same shape
func (tensor *Tensor[T]) SetData(value []T) *Tensor[T] {
	length := uint(len(value))
	var prod uint = 1
	for _, dim := range tensor.shape {
		prod = uint(dim) * prod
	}
	if prod != length {
		panic(fmt.Sprintf(
			"Shape %v cannot fit the number of elements %v. Change the shape first",
			tensor.shape, length),
		)
	}
	// TODO avoid data_buff
	tensor.data_buff = value
	return tensor
}

// set scalar to specific index
func (tensor *Tensor[T]) Set(indexes []int, value T) {
	flatIndex := tensor.getFlatIndex(indexes...)
	tensor.data()[flatIndex] = value
}

func (tensor *Tensor[T]) CreateIterator() *iter.TensorIterator {
	return iter.CreateIterator(len(tensor.data()), tensor.shape)
}

// Compares shapes and data:
// iterates over two tensors and compares elementwise
func (tensor *Tensor[T]) IsEqual(other *Tensor[T]) bool {
	if !tensor.shape.Equals(other.shape) {
		return false
	}

	it := tensor.CreateIterator()
	for it.Iterate() {
		idx := it.Next()
		if tensor.Get_fast(idx...) != other.Get_fast(idx...) {
			return false
		}
	}
	return true
}

func (tensor *Tensor[T]) IsAllClose(tensor_or_scalar *Tensor[T], tol float64) bool {
	if tensor_or_scalar.Shape().IsScalarLike() {
		other_val := tensor_or_scalar.data()[0]
		for _, val := range tensor.data() {
			if math.Abs(float64(val-other_val)) > tol {
				return false
			}
		}
	} else if tensor_or_scalar.Shape().Equals(tensor.Shape()) {
		it := tensor.CreateIterator()
		for it.Iterate() {
			idx := it.Next()
			a := tensor.Get_fast(idx...)
			b := tensor_or_scalar.Get_fast(idx...)
			if math.Abs(float64(a-b)) > tol {
				return false
			}
		}
	} else {
		panic("Other argument must be either tensor with the same shape or scalar.")
	}
	return true
}
