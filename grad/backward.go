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

func (this *Var[T]) Backward() {
	// toposort
	topo_sorted := make([]*Var[T], 0, 8)
	visited := CreateVarSet[T]()
	_toposort(&topo_sorted, visited, this)
	reverse_vars_inplace(topo_sorted)

	this.Grad = tensor.Ones[T](this.Value.Shape()...) // gradient w.r.t the current variable is ones tensor
	for _, v := range topo_sorted {
		if v.backward_fn != nil {
			v.Grad = v.Grad.Add(v.backward_fn())
		}
	}
}

func NumericDeriv(epsilon, value float64, op_func func(x float64) float64) float64 {
	return (op_func(value+epsilon) - op_func(value)) / epsilon
}
