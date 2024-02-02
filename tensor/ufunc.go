package tensor

import "gograd/tensor/internal/device"

func (tensor *Tensor[T]) ApplyFunc(expression_fn func(T) T, out ...*Tensor[T]) *Tensor[T] {
	if tensor.Err != nil {
		return tensor
	}
	outTensor, err := PrepareOutTensor(get_param(out...), tensor.Shape())
	if err != nil {
		tensor.Err = err
		return tensor
	}
	device.ApplyFunc(AUTO_IMPL, tensor.data(), expression_fn, outTensor.data())
	return outTensor
}
