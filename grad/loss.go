package grad

import (
	"gograd/tensor"
	"math"
)

// MSE impl. mean((y_true-y_pred)**2)
//
// 'y_true' is a cosnt by definition
func (y_pred *Var[T]) MSE(y_true *Var[T]) *Var[T] {
	squared := tensor.Scalar[T](2)
	mean := y_true.Value.Sub(y_pred.Value).Pow(squared).Mean(false)
	out := Variable(mean, y_pred)
	out.Alias = "MSE"
	if y_pred.Requires_grad {
		y_pred.backward_fn = func() *tensor.Tensor[T] {
			n := tensor.Scalar[T](T(len(y_true.Value.Data())))
			_const := tensor.Scalar[T](2).Div(n).Neg()
			return out.Grad.Mul(_const.Mul(y_true.Value.Sub(y_pred.Value)))
		}
	}
	return out
}

// -log(Ypredicted)
func (logits *Var[T]) SoftmaxCrossEntropy(y_true *Var[T]) *Var[T] {
	// if len(y_true.Value.Shape()) > 1 {
	// 	panic("y_true must be 1 dim")
	// }

	y_pred := logits.Value.Softmax(nil)

	var epsilon float64 = 1e-15
	// cross_entropy := y_pred.IndexMask(y_true.Value, true).Clip(epsilon, 1-epsilon).LnNeg()

	clipped_lnn := func(v T) T {
		// clips value and applies -log()
		fv := float64(v)
		if epsilon > fv {
			fv = epsilon
		}
		if 1-epsilon < fv {
			fv = 1 - epsilon
		}
		return T(-math.Log(fv))
	}
	cross_entropy := y_pred.IndexMask(y_true.Value, true)
	cross_entropy.ApplyFunc(clipped_lnn, cross_entropy)

	out := Variable(cross_entropy, logits).SetAlias("SoftmaxCrossEntropy")

	if logits.Requires_grad {
		n_classes := uint(logits.Value.Shape()[1])
		y_onehot := ToOneHot(y_true.Value, n_classes)
		// y_onehot := tensor.AsType[int, T](ToOneHot(y_true.Value, n_classes))
		logits.backward_fn = func() *tensor.Tensor[T] {
			return out.Grad.Mul(y_pred.Sub(y_onehot))
		}
	}
	return out
}
