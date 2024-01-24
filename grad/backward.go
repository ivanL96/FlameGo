package grad

import (
	"gograd/tensor"
	"gograd/tensor/types"
)

func _toposort[T types.TensorType](topo_sorted *[]*Var[T], visited *VarSet[T], v *Var[T]) {
	if visited.Contains(v) {
		return
	}
	visited.Add(v)
	for child := range v.Children.values {
		_toposort(topo_sorted, visited, child)
	}
	*topo_sorted = append(*topo_sorted, v)
}

func (v *Var[T]) toposort() []*Var[T] {
	topo_sorted := make([]*Var[T], 0, 8)
	visited := CreateVarSet[T]()
	_toposort(&topo_sorted, visited, v)
	return topo_sorted
}

func reverse_var_list[T types.TensorType](slice []*Var[T]) {
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
	var_list := this.toposort()
	reverse_var_list(var_list)
	this.Grad = tensor.Ones[T](this.Value.Shape()...)
	for _, v := range var_list {
		if v.backward_fn == nil {
			continue
		}
		v.backward_fn()
	}
}

func NumericDeriv(epsilon, value float64, op_func func(x float64) float64) float64 {
	return (op_func(value+epsilon) - op_func(value)) / epsilon
}
