package tensor

import (
	"fmt"
	"math"
)

type BinaryScalarOp[T TensorType] func(T, T) T
type UnaryScalarOp[T TensorType] func(T) T

func binElementwiseRoutine[T TensorType](
	tensor_a,
	tensor_b *Tensor[T],
	binOp BinaryScalarOp[T],
	out *Tensor[T],
) *Tensor[T] {
	// tensors should have equal shapes or at least one of them should be scalar-like
	is_broadcastable_ := are_broadcastable(tensor_a.shape, tensor_b.shape)
	if !is_broadcastable_ {
		panic(
			fmt.Sprintf("Shapes: %x, %x are not broadcastable", tensor_a.shape, tensor_b.shape),
		)
	}
	var new_tensor *Tensor[T] = nil
	if out == nil {
		new_tensor = InitEmptyTensor[T](tensor_a.shape...)
	} else {
		// TODO use shape from "out" tensor
		new_tensor = out
	}
	if isScalarLike(tensor_a.shape) && isScalarLike(tensor_b.shape) {
		// most trivial case
		new_tensor.data[0] = binOp(tensor_a.data[0], tensor_b.data[0])
	} else if len(tensor_a.data) == len(tensor_b.data) {
		// same shapes
		for i, val := range tensor_a.data {
			new_tensor.data[i] = binOp(val, tensor_b.data[i])
		}
	} else if len(tensor_b.data) == 1 {
		// one of them scalar
		for i, val := range tensor_a.data {
			new_tensor.data[i] = binOp(val, tensor_b.data[0])
		}
	} else if len(tensor_a.data) == 1 {
		for i, val := range tensor_b.data {
			new_tensor.data[i] = binOp(val, tensor_a.data[0])
		}
	} else {
		// apply operation for non scalar broadcastable tensors
		var broadcasted_shape Shape = broadcast(tensor_a.shape, tensor_b.shape)

		// define which tensor (at least 1) should be broadcasted
		var broadcasted_tensor_a *Tensor[T] = nil
		var broadcasted_tensor_b *Tensor[T] = nil
		for i, brc_dim := range broadcasted_shape {
			if i < len(tensor_a.shape) {
				dim_a := tensor_a.shape[i]
				if dim_a < brc_dim && broadcasted_tensor_a == nil {
					// need to broadcast tensor_a
					// TODO how to avoid additional Broadcast()
					broadcasted_tensor_a = tensor_a.Broadcast(broadcasted_shape...)
				}
			}
			if i < len(tensor_b.shape) {
				dim_b := tensor_b.shape[i]
				if dim_b < brc_dim && broadcasted_tensor_b == nil {
					// need to broadcast tensor_b
					broadcasted_tensor_b = tensor_b.Broadcast(broadcasted_shape...)
				}
			}
		}
		new_tensor.shape = broadcasted_shape
		var length Dim = 1
		for _, dim := range broadcasted_shape {
			length *= dim
		}
		new_tensor.data = make([]T, int(length))
		if broadcasted_tensor_a == nil {
			broadcasted_tensor_a = tensor_a
		}
		if broadcasted_tensor_b == nil {
			broadcasted_tensor_b = tensor_b
		}
		for i, val := range broadcasted_tensor_a.data {
			new_tensor.data[i] = binOp(val, broadcasted_tensor_b.data[i])
		}
	}
	return new_tensor
}

func unaryElementwiseRoutine[T TensorType](
	tensor *Tensor[T],
	unaryOp UnaryScalarOp[T],
	out *Tensor[T],
) *Tensor[T] {
	var outTensor *Tensor[T] = nil
	if out == nil {
		outTensor = InitEmptyTensor[T](tensor.shape...)
	} else {
		outTensor = out
	}
	if isScalarLike(tensor.shape) {
		outTensor.data[0] = unaryOp(tensor.data[0])
		return outTensor
	}
	for i, val := range tensor.data {
		outTensor.data[i] = unaryOp(val)
	}
	return outTensor
}

func _add[T TensorType](a, b T) T {
	return a + b
}

func (tensor *Tensor[T]) Add(other_tensor *Tensor[T]) *Tensor[T] {
	return binElementwiseRoutine(tensor, other_tensor, _add[T], nil)
}

func _sub[T TensorType](a, b T) T {
	return a - b
}
func (tensor *Tensor[T]) Sub(other_tensor *Tensor[T]) *Tensor[T] {
	return binElementwiseRoutine(tensor, other_tensor, _sub[T], nil)
}

func _mul[T TensorType](a, b T) T {
	return a * b
}
func (tensor *Tensor[T]) Mul(other_tensor *Tensor[T]) *Tensor[T] {
	return binElementwiseRoutine(tensor, other_tensor, _mul[T], nil)
}

func _div[T TensorType](a, b T) T {
	return a / b
}
func (tensor *Tensor[T]) Div(other_tensor *Tensor[T]) *Tensor[T] {
	return binElementwiseRoutine(tensor, other_tensor, _div[T], nil)
}

func _sigmoid[T TensorType](a T) T {
	return T(1. / (1. + math.Pow(math.E, float64(-a))))
}
func (tensor *Tensor[T]) Sigmoid() *Tensor[T] {
	return unaryElementwiseRoutine(tensor, _sigmoid[T], nil)
}
