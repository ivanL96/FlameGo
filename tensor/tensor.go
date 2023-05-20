package tensor

import (
	"fmt"
)

func makeTensor[T TensorType](dataPtr *[]T, shape Shape) *Tensor[T] {
	var shapeProd Dim = 1
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
	dim_order := initDimOrder(shape)
	return &Tensor[T]{
		shape:     append(Shape(nil), shape...),
		strides:   getStrides(shape),
		data:      data,
		dtype:     getTypeArray(data),
		dim_order: dim_order,
	}
}

// inits a tensor with data
func InitTensor[T TensorType](value []T, shape Shape) *Tensor[T] {
	return makeTensor(&value, shape)
}

func InitEmptyTensor[T TensorType](shape ...Dim) *Tensor[T] {
	return makeTensor[T](nil, shape)
}

func (tensor *Tensor[T]) SetData(value []T) *Tensor[T] {
	// sets new value of the same shape
	length := uint(len(value))
	var prod uint = 1
	for _, dim := range tensor.shape {
		prod = uint(dim) * prod
	}
	if prod != length {
		msg := fmt.Sprintf(
			"Shape %v cannot fit the number of elements %v. Change the shape first",
			tensor.shape, length)
		panic(msg)
	}
	tensor.data = value
	return tensor
}

func AsType[OLDT TensorType, NEWT TensorType](tensor *Tensor[OLDT]) *Tensor[NEWT] {
	// naive impl with copying the data & tensor
	// example:
	// AsType(int32, float64)(tensor) ==> float64 tensor
	data := make([]NEWT, len(tensor.data))
	for i, val := range tensor.data {
		data[i] = NEWT(val)
	}
	return InitTensor(data, tensor.shape)
}

func (tensor *Tensor[T]) Copy() *Tensor[T] {
	newData := make([]T, len(tensor.data))
	copy(newData, tensor.data)
	newTensor := InitTensor(newData, tensor.shape)
	newTensor.strides = tensor.strides
	newTensor.dim_order = tensor.dim_order
	return newTensor
}

func (tensor *Tensor[T]) IsEqual(otherTensor *Tensor[T]) bool {
	// FIXME this should be aware of tensor.strides
	// iterates over two tensors and compares elementwise
	if !Equal_1D_slices(tensor.shape, otherTensor.shape) {
		return false
	}
	if !Equal_1D_slices(tensor.strides, otherTensor.strides) {
		return false
	}
	if !Equal_1D_slices(tensor.dim_order, otherTensor.dim_order) {
		return false
	}
	if !Equal_1D_slices(tensor.data, otherTensor.data) {
		return false
	}
	return true
}

func (tensor *Tensor[T]) Fill(value T) *Tensor[T] {
	for i := range tensor.data {
		tensor.data[i] = value
	}
	return tensor
}

func Range[T TensorType](limits ...int) *Tensor[T] {
	// Created a tensor with data ranged from 'start' to 'end'
	// limits: min 1 and max 3 arguments. Start, End, Step
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
	tensor := InitEmptyTensor[T](Dim(length))
	for i := 0; i < length; i++ {
		tensor.data[i] = T(start)
		start += step
	}
	return tensor
}
