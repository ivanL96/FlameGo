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
		Value:    tensor_val,
		Grad:     tensor.Scalar[T](0),
		Children: CreateVarSet[T](children...),
	}
}

func (v *Var[T]) ToString() string {
	return fmt.Sprintf("Var: %v", v.Value.ToString())
}

// VARIABLE OPS

func (this *Var[T]) Add(other *Var[T]) *Var[T] {
	out := Variable(this.Value.Add(other.Value), this, other)
	this.backward_fn = func() {
		this.Grad = this.Grad.Add(out.Grad)   // this.g += out.g
		other.Grad = other.Grad.Add(out.Grad) // other.g += out.g
	}
	return out
}

func (this *Var[T]) Sub(other *Var[T]) *Var[T] {
	out := Variable(this.Value.Sub(other.Value), this, other)
	this.backward_fn = func() {
		this.Grad = this.Grad.Sub(out.Grad)   // this.g -= out.g
		other.Grad = other.Grad.Sub(out.Grad) // other.g -= out.g
	}
	return out
}

func (this *Var[T]) Mul(other *Var[T]) *Var[T] {
	out := Variable(this.Value.Mul(other.Value), this, other)
	this.backward_fn = func() {
		this.Grad = this.Grad.Add(other.Value.Mul(out.Grad))  // this.g += other.val * out.g
		other.Grad = other.Grad.Add(this.Value.Mul(out.Grad)) // other.g += this.val * out.g
	}
	return out
}

// this/other
// => d(this): 1/other
// => d(other): (-this) / (other**2)
func (this *Var[T]) Div(other *Var[T]) *Var[T] {
	out := Variable(this.Value.Div(other.Value), this, other)
	// one := tensor.Scalar[T](1)
	this.backward_fn = func() {
		this.Grad = this.Grad.Add(out.Grad.Div(other.Value))                            // this.g += out.g / other.val
		other.Grad = other.Grad.Add(this.Value.Neg().Div(other.Value.Mul(other.Value))) // other.g += -this.val * other.
	}
	return out
}
