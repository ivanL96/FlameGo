package tensor

import (
	"fmt"
	types "gograd/tensor/types"
	"math"
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
	if !AreBroadcastable(tensor_a.Shape(), tensor_b.Shape()) {
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
		var broadcasted_shape types.Shape = BroadcastShapes(tensor_a.Shape(), tensor_b.Shape())

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

// binary ops
func _add[T types.TensorType](a, b T) T {
	return a + b
}

func (tensor *Tensor[T]) Add(other_tensor *Tensor[T]) *Tensor[T] {
	return BaseBinElementwiseOp(tensor, other_tensor, _add[T], nil)
}

func _sub[T types.TensorType](a, b T) T {
	return a - b
}
func (tensor *Tensor[T]) Sub(other_tensor *Tensor[T]) *Tensor[T] {
	return BaseBinElementwiseOp(tensor, other_tensor, _sub[T], nil)
}

func _mul[T types.TensorType](a, b T) T {
	return a * b
}
func (tensor *Tensor[T]) Mul(other_tensor *Tensor[T]) *Tensor[T] {
	return BaseBinElementwiseOp(tensor, other_tensor, _mul[T], nil)
}

func _div[T types.TensorType](a, b T) T {
	return a / b
}
func (tensor *Tensor[T]) Div(other_tensor *Tensor[T]) *Tensor[T] {
	return BaseBinElementwiseOp(tensor, other_tensor, _div[T], nil)
}

func _sigmoid[T types.TensorType](a T) T {
	return T(1. / (1. + math.Pow(math.E, float64(-a))))
}
func (tensor *Tensor[T]) Sigmoid() *Tensor[T] {
	return unaryElementwiseRoutine(tensor, _sigmoid[T], nil)
}

// input A and B, both n by n matrices
// initialize C to be an n by n matrix of all zeros
// for i from 1 to n:
//     for j from 1 to n:
//         for k from 1 to n:
//             C[i][j] = C[i][j] + A[i][k]*B[k][j]
// output C (as A*B)

func (tensor *Tensor[T]) MatMul(other_tensor *Tensor[T]) *Tensor[T] {
	if len(tensor.shape) != 2 || len(other_tensor.shape) != 2 {
		panic("Tensors must be two-dim.")
	}
	if tensor.shape[1] != other_tensor.shape[0] {
		panic(fmt.Sprintf(
			"Tensors inner shapes are different. %v != %v", tensor.shape[1], other_tensor.shape[0],
		))
	}
	tensor_a := tensor.AsContinuous()
	tensor_b := other_tensor.AsContinuous()
	outShape := tensor.shape
	outTensor := InitEmptyTensor[T](outShape...)
	for i := 0; i < int(tensor.shape[0]); i++ {
		for j := 0; j < int(other_tensor.shape[1]); j++ {
			for k := 0; k < int(tensor.shape[1]); k++ {
				dataIdx := outTensor.getFlatIndex(i, j)
				outTensor.data[dataIdx] += tensor_a.Get(i, k) * tensor_b.Get(k, j)
			}
		}
	}
	return outTensor
}
