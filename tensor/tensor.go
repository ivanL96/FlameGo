package tensor

import (
	"fmt"
	"gograd/tensor/intrinsics/cpu"
	"gograd/tensor/iter"
	types "gograd/tensor/types"
)

var auto_impl cpu.Implementation = cpu.DetectImpl()

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
		strides:   getStrides(shape),
		data:      data,
		dim_order: initDimOrder(shape),
	}

	if auto_impl == cpu.AVX {
		tensor.UseAVX()
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

func Scalar[T types.TensorType](value T) *Tensor[T] {
	return makeTensor(&[]T{value}, types.Shape{1})
}

// Creates a tensor without copying the data
func AsTensor[T types.TensorType](data []T, shape types.Shape) *Tensor[T] {
	t := &Tensor[T]{
		strides:   getStrides(shape),
		data:      data,
		dim_order: initDimOrder(shape),
		shape:     shape,
	}
	return t
}

//-----------------------------------------------------

func (tensor *Tensor[T]) UseAVX() {
	if cpu.IsImplAvailable(cpu.AVX) {
		panic("AVX is not available.")
	}
	tensor.setFlag(UseAVXFlag)
}

func (tensor *Tensor[T]) UseDefault() {
	tensor.clearFlag(UseAVXFlag)
}

func AsType[OLDT types.TensorType, NEWT types.TensorType](tensor *Tensor[OLDT]) *Tensor[NEWT] {
	// naive impl with copying the data & tensor
	// example:
	// AsType(int32, float64)(tensor) ==> float64 tensor
	data := make([]NEWT, len(tensor.data))
	for i, val := range tensor.data {
		data[i] = NEWT(val)
	}
	return CreateTensor(data, tensor.shape)
}

func (tensor *Tensor[T]) Copy() *Tensor[T] {
	newData := make([]T, len(tensor.data))
	copy(newData, tensor.data)
	newTensor := CreateTensor(newData, tensor.shape)
	newTensor.strides = tensor.strides
	newTensor.dim_order = tensor.dim_order
	newTensor.flags = tensor.flags
	return newTensor
}

func (tensor *Tensor[T]) IsEqual(otherTensor *Tensor[T]) bool {
	// iterates over two tensors and compares elementwise
	if !Equal_1D_slices(tensor.shape, otherTensor.shape) {
		return false
	}

	it := tensor.CreateIterator()
	for it.Iterate() {
		idx := it.Next()
		if tensor.Get_fast(idx...) != otherTensor.Get_fast(idx...) {
			return false
		}
	}
	return true
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
		tensor.data[i] = T(start)
		start += step
	}
	return tensor
}

// Fills tensor with same value
func (tensor *Tensor[T]) Fill(value T) *Tensor[T] {
	tensor.setFlag(SameValuesFlag)
	if len(tensor.data) >= 12 {
		fill_data_unroll4(&tensor.data, value)
	} else {
		for i := range tensor.data {
			tensor.data[i] = value
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
				eye.data[fidx] = 1
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
	tensor.data = value
	tensor.clearFlag(SameValuesFlag)
	return tensor
}

// set scalar to specific index
func (tensor *Tensor[T]) Set(indexes []int, value T) {
	tensor.clearFlag(SameValuesFlag)
	flatIndex := tensor.getFlatIndex(indexes...)
	tensor.data[flatIndex] = value
}

func (tensor *Tensor[T]) CreateIterator() *iter.TensorIterator {
	return iter.CreateIterator(len(tensor.data), tensor.shape)
}
