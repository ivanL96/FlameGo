package grad

import (
	"fmt"
	"gograd/tensor"
	"gograd/tensor/types"
)

func toposort[T types.TensorType](
	topo_sorted *[]*Var[T],
	visited *VarSet[T],
	v *Var[T],
) {
	if visited.Contains(v) {
		return
	}
	visited.Add(v)
	for _, child := range v.Children {
		toposort(topo_sorted, visited, child)
	}
	*topo_sorted = append(*topo_sorted, v)
}

func reverse_vars_inplace[T types.TensorType](slice []*Var[T]) {
	l := len(slice)
	if l <= 1 {
		return
	}
	for i := l/2 - 1; i >= 0; i-- {
		opp := l - 1 - i
		slice[i], slice[opp] = slice[opp], slice[i]
	}
}

// This method performs gradient computation for each Variable.
// Parameter `gradient` is optional and must be set in cases where the result (`this`) Variable is not scalar.
func (this *Var[T]) Backward(gradient *tensor.Tensor[T]) {
	this.MustAssert()
	// toposort
	topo_sorted := make([]*Var[T], 0)
	visited := CreateVarSet[T]()
	toposort(&topo_sorted, visited, this)
	reverse_vars_inplace(topo_sorted)

	// gradient w.r.t the current Var is ones tensor
	if gradient == nil {
		if this.Value.Shape().IsScalarLike() {
			this.Grad = tensor.Ones[T](this.Value.Shape()...)
		} else {
			panic(
				fmt.Sprintf(
					"The result value is not scalar and has shape (%v). Initial gradient should be set explicitly",
					this.Value.Shape(),
				),
			)
		}
	} else {
		this.Grad = gradient
	}

	for _, v := range topo_sorted {
		if v.backward_fn != nil {
			new_grad := v.backward_fn()
			if !new_grad.Shape().AreBroadcastable(v.Grad.Shape()) {
				panic(fmt.Sprintf("Error at '%v' backprop. Old grad Shape is %v but new grad Shape is %v",
					v.Alias, v.Grad.Shape(), new_grad.Shape()))
			}
			v.Grad.Add(new_grad, v.Grad).MustAssert()
		}
	}
}

const EPSILON = 0.00000000001

// numerical derivative calc can be used for verifying auto-diff expressions
//
// Example:
// f=x*5, where x=4;
//
//	_mul := func(x float64) float64 {
//		return x * 5
//	}
//
// dx := grad.NumericDeriv(grad.EPSILON, 4, _mul)
//
// dx ~5
func NumericDeriv(epsilon, value float64, op_func func(x float64) float64) float64 {
	return (op_func(value+epsilon) - op_func(value)) / epsilon
}
