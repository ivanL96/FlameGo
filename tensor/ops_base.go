package tensor

import (
	"fmt"
	types "gograd/tensor/types"
)

type BinaryScalarOp[T types.TensorType] func(T, T) T
type UnaryScalarOp[T types.TensorType] func(T) T

func BaseBinElementwiseOp[T types.TensorType](
	tensor_a,
	tensor_b *Tensor[T],
	binOp BinaryScalarOp[T],
	out *Tensor[T],
) *Tensor[T] {
	// tensors should have equal shapes or at least one of them should be scalar-like
	is_broadcastable_ := AreBroadcastable(tensor_a.Shape(), tensor_b.Shape())
	if !is_broadcastable_ {
		panic(
			fmt.Sprintf("Shapes: %x, %x are not broadcastable", tensor_a.Shape(), tensor_b.Shape()),
		)
	}
	outTensor := PrepareOutTensor(out, tensor_a.Shape())

	if IsScalarLike(tensor_a.Shape()) && IsScalarLike(tensor_b.Shape()) {
		// most trivial case (1,) & (1,)
		outTensor.Data()[0] = binOp(tensor_a.Data()[0], tensor_b.Data()[0])
	} else if len(tensor_a.Data()) == len(tensor_b.Data()) {
		// same shapes (N,M) & (N,M)
		iter := tensor_a.CreateIterator()
		for iter.Iterate() {
			dataIndex := iter.Index()
			idx := iter.Next()
			outTensor.Data()[dataIndex] = binOp(tensor_a.Get(idx...), tensor_b.Get(idx...))
		}
	} else if len(tensor_b.Data()) == 1 {
		// one of them scalar
		// (N, M, ...) & (1,)
		for i, val := range tensor_a.Data() {
			outTensor.Data()[i] = binOp(val, tensor_b.Data()[0])
		}
	} else if len(tensor_a.Data()) == 1 {
		// (1,) & (N, M, ...)
		for i, val := range tensor_b.Data() {
			outTensor.Data()[i] = binOp(val, tensor_a.Data()[0])
		}
	} else {
		// (A, B ...) & (N, M, ...)
		// apply operation for non scalar broadcastable tensors
		var broadcasted_shape types.Shape = Broadcast(tensor_a.Shape(), tensor_b.Shape())

		// define which tensor (at least 1) should be broadcasted
		var broadcasted_tensor_a *Tensor[T] = nil
		var broadcasted_tensor_b *Tensor[T] = nil
		for i, brc_dim := range broadcasted_shape {
			if i < len(tensor_a.Shape()) {
				dim_a := tensor_a.Shape()[i]
				if dim_a < brc_dim && broadcasted_tensor_a == nil {
					// need to broadcast tensor_a
					// TODO how to avoid additional Broadcast()
					broadcasted_tensor_a = tensor_a.Broadcast(broadcasted_shape...)
				}
			}
			if i < len(tensor_b.Shape()) {
				dim_b := tensor_b.Shape()[i]
				if dim_b < brc_dim && broadcasted_tensor_b == nil {
					// need to broadcast tensor_b
					broadcasted_tensor_b = tensor_b.Broadcast(broadcasted_shape...)
				}
			}
		}
		outTensor.shape = broadcasted_shape
		var length types.Dim = 1
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
			outTensor.Data()[dataIndex] = binOp(
				broadcasted_tensor_a.Get(idx...), broadcasted_tensor_b.Get(idx...))
		}
	}
	return outTensor
}

func unaryElementwiseRoutine[T types.TensorType](
	tensor *Tensor[T],
	unaryOp UnaryScalarOp[T],
	out *Tensor[T],
) *Tensor[T] {
	outTensor := PrepareOutTensor(out, tensor.Shape())
	if IsScalarLike(tensor.Shape()) {
		outTensor.Data()[0] = unaryOp(tensor.Data()[0])
		return outTensor
	}
	for i, val := range tensor.Data() {
		outTensor.Data()[i] = unaryOp(val)
	}
	return outTensor
}
