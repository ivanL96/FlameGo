package tensor

import "fmt"

type BinaryScalarOp[T Number] func(T, T) T

func elementwise_routine[T Number](
	tensor_a,
	tensor_b *Tensor[T],
	binOp BinaryScalarOp[T],
	out *Tensor[T],
) *Tensor[T] {
	// tensors should have equal shapes or at least one of them should be scalar-like
	is_broadcastable_ := are_broadcastable(tensor_a.shape, tensor_b.shape)
	if !is_broadcastable_ {
		err_msg := fmt.Sprintf("Shapes: %x, %x are not broadcastable", tensor_a.shape, tensor_b.shape)
		panic(err_msg)
	}
	var new_tensor *Tensor[T] = nil
	if out == nil {
		new_tensor = InitEmptyTensor[T](tensor_a.shape...)
	} else {
		new_tensor = out
	}
	if tensor_a.len == tensor_b.len {
		// same shapes
		for i, val := range tensor_a.data {
			new_tensor.data[i] = binOp(val, tensor_b.data[i])
		}
	} else if tensor_b.len == 1 {
		// one of them scalar
		for i, val := range tensor_a.data {
			new_tensor.data[i] = binOp(val, tensor_b.data[0])
		}
	} else if tensor_a.len == 1 {
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
					broadcasted_tensor_a = tensor_a.Copy().Broadcast(broadcasted_shape...)
				}
			}
			if i < len(tensor_b.shape) {
				dim_b := tensor_b.shape[i]
				if dim_b < brc_dim && broadcasted_tensor_b == nil {
					// need to broadcast tensor_b
					broadcasted_tensor_b = tensor_b.Copy().Broadcast(broadcasted_shape...)
				}
			}
		}
		new_tensor.shape = broadcasted_shape
		var length Dim = 1
		for _, dim := range broadcasted_shape {
			length *= dim
		}
		new_tensor.ndim = Dim(len(broadcasted_shape))
		new_tensor.data = make([]T, int(length))
		new_tensor.len = uint(length)
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

func _add[T Number](a, b T) T {
	return a + b
}

func (tensor *Tensor[T]) Add(other_tensor *Tensor[T]) *Tensor[T] {
	return elementwise_routine(tensor, other_tensor, _add[T], nil)
}

func _sub[T Number](a, b T) T {
	return a - b
}
func (tensor *Tensor[T]) Sub(other_tensor *Tensor[T]) *Tensor[T] {
	return elementwise_routine(tensor, other_tensor, _sub[T], nil)
}

func _mul[T Number](a, b T) T {
	return a * b
}
func (tensor *Tensor[T]) Mul(other_tensor *Tensor[T]) *Tensor[T] {
	return elementwise_routine(tensor, other_tensor, _mul[T], nil)
}

func _div[T Number](a, b T) T {
	return a / b
}
func (tensor *Tensor[T]) Div(other_tensor *Tensor[T]) *Tensor[T] {
	return elementwise_routine(tensor, other_tensor, _div[T], nil)
}
