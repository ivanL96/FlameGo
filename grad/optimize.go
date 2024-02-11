package grad

import (
	"gograd/tensor"
	"gograd/tensor/types"
)

type Optimizer[T types.TensorType] struct {
	lr     *tensor.Tensor[T]
	params []*Var[T]
}

func SGD[T types.TensorType](lr float64) *Optimizer[T] {
	return &Optimizer[T]{
		lr: tensor.Scalar[T](T(lr)),
	}
}

func (opt *Optimizer[T]) Step(parameters ...*Var[T]) {
	// new = old - lr*gradient
	opt.params = parameters
	for _, param := range parameters {
		param.Value.GradientStep(param.Grad, opt.lr)
		// param.Value.Sub(param.Grad.Mul(opt.lr), param.Value)
	}
}

func (opt *Optimizer[T]) ZeroGrads() {
	for _, param := range opt.params {
		param.ZeroGrad()
	}
}
