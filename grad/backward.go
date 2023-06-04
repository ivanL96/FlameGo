package grad

import (
	"flamego/tensor"
	"flamego/tensor/types"
)

func _toposort[T types.TensorType](topo_sorted *[]*Var[T], visited *VarSet[T], v *Var[T]) {
	if !visited.Contains(v) {
		visited.Add(v)
		for child := range v.Children.values {
			_toposort(topo_sorted, visited, child)
		}
		*topo_sorted = append(*topo_sorted, v)
	}
}

func (v *Var[T]) toposort() []*Var[T] {
	topo_sorted := make([]*Var[T], 0, 16)
	visited := CreateVarSet[T]()
	_toposort(&topo_sorted, visited, v)
	return topo_sorted
}

func reverse_slice_inplace[T any](slice []T) {
	for i := len(slice)/2 - 1; i >= 0; i-- {
		opp := len(slice) - 1 - i
		slice[i], slice[opp] = slice[opp], slice[i]
	}
}

func (v *Var[T]) Backward() {
	topo := v.toposort()
	reverse_slice_inplace(topo)
	v.Grad = tensor.Scalar[T](1)
	for _, v := range topo {
		if v.backward_fn != nil {
			v.backward_fn()
		}
	}
}
