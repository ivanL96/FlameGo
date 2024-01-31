package tensor

import "gograd/tensor/internal/device"

func (tensor *Tensor[T]) ApplyFunc(expression_fn func(T) T, out ...*Tensor[T]) *Tensor[T] {
	outTensor := PrepareOutTensor(get_param(out...), tensor.Shape())
	device.ApplyFunc(auto_impl, tensor.data(), expression_fn, outTensor.data())
	return outTensor
}
