package tensor

import "fmt"

type BinaryScalarOp[T Number] func(T, T) T

func elementwise_routine[T Number](tensor_a, tensor_b *Tensor[T], binOp BinaryScalarOp[T], out *Tensor[T]) *Tensor[T] {
	// tensors should have equal shapes or at least one of them should be scalar-like
	is_broadcastable_ := are_broadcastable(tensor_a.shape, tensor_b.shape)
	// Log(is_broadcastable_)
	if !is_broadcastable_ {
		err_msg := fmt.Sprintf("Shapes: %x, %x are not broadcastable", tensor_a.shape, tensor_b.shape)
		panic(err_msg)
	}
	var new_tensor *Tensor[T] = nil
	if out == nil {
		new_tensor = InitTensor[T](tensor_a.shape...)
	} else {
		new_tensor = out
	}
	fmt.Println(tensor_a.ToString(), tensor_b.ToString())
	if tensor_a.len == tensor_b.len {
		for i, val := range tensor_a.data {
			new_tensor.data[i] = binOp(val, tensor_b.data[i])
		}
	} else if tensor_b.len == 1 {
		for i, val := range tensor_a.data {
			new_tensor.data[i] = binOp(val, tensor_b.data[0])
		}
	} else if tensor_a.len == 1 {
		for i, val := range tensor_b.data {
			new_tensor.data[i] = binOp(val, tensor_a.data[0])
		}
	} else {
		new_shape := broadcast(tensor_a.shape, tensor_b.shape)
		fmt.Println(new_shape)
		// TODO apply operation for non scalar broadcastable tensors
		// panic(fmt.Sprintf("Cannot apply elementwise op for %s, %s", tensor_a.ToString(), tensor_b.ToString()))
	}

	new_tensor = new_tensor.Broadcast(tensor_b.shape)
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
