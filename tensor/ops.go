package tensor

import (
	types "gograd/tensor/types"
	"math"
)

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
