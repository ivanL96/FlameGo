package grad

import (
	"gograd/tensor"
)

// MSE impl. mean((y_true-y_pred)**2)
//
// 'y_true' is a cosnt by definition
func (y_pred *Var[T]) MSE(y_true *Var[T]) *Var[T] {
	squared := tensor.Scalar[T](2)
	mean := y_true.Value.Sub(y_pred.Value).Pow(squared).Mean(false)
	out := Variable(mean, y_pred)
	if y_pred.Requires_grad {
		y_pred.backward_fn = func() *tensor.Tensor[T] {
			n := tensor.Scalar[T](T(len(y_true.Value.Data())))
			_const := tensor.Scalar[T](2).Div(n) // Neg()
			return out.Grad.Mul(_const.Mul(y_true.Value.Sub(y_pred.Value)))
		}
	}
	return out
}
