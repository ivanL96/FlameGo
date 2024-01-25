package grad

import (
	"gograd/tensor"
	"gograd/tensor/types"
)

func _toposort[T types.TensorType](
	topo_sorted *[]*Var[T],
	visited *VarSet[T],
	v *Var[T],
) {
	if visited.Contains(v) {
		return
	}
	visited.Add(v)
	for _, child := range v.Children {
		_toposort(topo_sorted, visited, child)
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
	// toposort
	topo_sorted := make([]*Var[T], 0, 8)
	visited := CreateVarSet[T]()
	_toposort(&topo_sorted, visited, this)
	reverse_vars_inplace(topo_sorted)

	// gradient w.r.t the current variable is ones tensor
	if gradient == nil {
		this.Grad = tensor.Ones[T](this.Value.Shape()...)
	} else {
		this.Grad = gradient
	}
	for _, v := range topo_sorted {
		if v.backward_fn != nil {
			v.Grad = v.Grad.Add(v.backward_fn())
		}
	}
}

const EPSILON = 0.00000000001

func NumericDeriv(epsilon, value float64, op_func func(x float64) float64) float64 {
	return (op_func(value+epsilon) - op_func(value)) / epsilon
}
