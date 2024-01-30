package tensor

import "gograd/tensor/internal/device"

func (tensor *Tensor[T]) Mask(expression_fn func(T) T, out ...*Tensor[T]) *Tensor[T] {
	outTensor := PrepareOutTensor(get_param(out...), tensor.Shape())
	device.Mask(auto_impl, tensor.data(), expression_fn, outTensor.data())
	return outTensor
}
