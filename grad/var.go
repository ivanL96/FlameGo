package grad

import (
	"flamego/tensor"
	"flamego/tensor/types"
	"fmt"
)

type Var[T types.TensorType] struct {
	Children    *VarSet[T]
	Value       *tensor.Tensor[T]
	Grad        *tensor.Tensor[T]
	backward_fn func()
}

// type BinBackward[T types.TensorType] [2]*Var[T]

func Variable[T types.TensorType](tensor_val *tensor.Tensor[T], children ...*Var[T]) *Var[T] {
	return &Var[T]{
		Value: tensor_val, Grad: tensor.Scalar[T](0),
		Children: CreateVarSet[T](children...),
	}
}

func (v *Var[T]) ToString() string {
	return fmt.Sprintf("Var: %v", v.Value.ToString())
}

func (v *Var[T]) Mul(other *Var[T]) *Var[T] {
	out := Variable(v.Value.Mul(other.Value), v, other)
	v.backward_fn = func() {
		v.Grad.Add(other.Value.Mul(out.Grad), v.Grad)     // v += other*out
		other.Grad.Add(v.Value.Mul(out.Grad), other.Grad) // other += v*out
	}
	return out
}
