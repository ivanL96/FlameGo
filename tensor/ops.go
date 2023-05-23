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
	outTensor := prepareOutTensor(out, tensor_a.shape)

	if isScalarLike(tensor_a.shape) && isScalarLike(tensor_b.shape) {
		// most trivial case (1,) & (1,)
		outTensor.data[0] = binOp(tensor_a.data[0], tensor_b.data[0])
	} else if len(tensor_a.data) == len(tensor_b.data) {
		// same shapes (N,M) & (N,M)
		iter := tensor_a.CreateIterator()
		for iter.Iterate() {
			dataIndex := iter.Index()
			idx := iter.Next()
			outTensor.data[dataIndex] = binOp(tensor_a.Get(idx...), tensor_b.Get(idx...))
		}
	} else if len(tensor_b.data) == 1 {
		// one of them scalar
		// (N, M, ...) & (1,)
		for i, val := range tensor_a.data {
			outTensor.data[i] = binOp(val, tensor_b.data[0])
		}
	} else if len(tensor_a.data) == 1 {
		// (1,) & (N, M, ...)
		for i, val := range tensor_b.data {
			outTensor.data[i] = binOp(val, tensor_a.data[0])
		}
	} else {
		// (A, B ...) & (N, M, ...)
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
		outTensor.shape = broadcasted_shape
		var length Dim = 1
		for _, dim := range broadcasted_shape {
			length *= dim
		}
		outTensor.data = make([]T, int(length))
		if broadcasted_tensor_a == nil {
			broadcasted_tensor_a = tensor_a
		}
		if broadcasted_tensor_b == nil {
			broadcasted_tensor_b = tensor_b
		}
		iter := broadcasted_tensor_a.CreateIterator()
		for iter.Iterate() {
			dataIndex := iter.Index()
			idx := iter.Next()
			outTensor.data[dataIndex] = binOp(
				broadcasted_tensor_a.Get(idx...), broadcasted_tensor_b.Get(idx...))
		}
	}
	return outTensor
}

func unaryElementwiseRoutine[T TensorType](
	tensor *Tensor[T],
	unaryOp UnaryScalarOp[T],
	out *Tensor[T],
) *Tensor[T] {
	outTensor := prepareOutTensor(out, tensor.shape)
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
