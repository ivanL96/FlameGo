package grad

import (
	"fmt"
	"gograd/tensor"
	"gograd/tensor/types"
)

type Var[T types.TensorType] struct {
	Children      []*Var[T]
	Value         *tensor.Tensor[T]
	Grad          *tensor.Tensor[T]
	backward_fn   func() *tensor.Tensor[T]
	Requires_grad bool
}

// type BinBackward[T types.TensorType] [2]*Var[T]

func Variable[T types.TensorType](tensor_val *tensor.Tensor[T], children ...*Var[T]) *Var[T] {
	return &Var[T]{
		Value:         tensor_val,
		Grad:          tensor.Scalar[T](0),
		Children:      children,
		Requires_grad: true,
	}
}

func Constant[T types.TensorType](tensor_val *tensor.Tensor[T], children ...*Var[T]) *Var[T] {
	return &Var[T]{
		Value:         tensor_val,
		Grad:          tensor.Scalar[T](0),
		Children:      children,
		Requires_grad: false,
	}
}

func (v *Var[T]) ToString() string {
	return fmt.Sprintf("Var: %v", v.Value.ToString())
}

// VARIABLE OPS

func (this *Var[T]) Add(other *Var[T]) *Var[T] {
	out := Variable(this.Value.Add(other.Value), this, other)
	if this.Requires_grad {
		this.backward_fn = func() *tensor.Tensor[T] {
			return out.Grad
		}
	}
	if other.Requires_grad {
		other.backward_fn = func() *tensor.Tensor[T] {
			return out.Grad
		}
	}
	return out
}

func (this *Var[T]) Sub(other *Var[T]) *Var[T] {
	out := Variable(this.Value.Sub(other.Value), this, other)
	if this.Requires_grad {
		this.backward_fn = func() *tensor.Tensor[T] {
			return out.Grad.Neg() // -out.g
		}
	}
	if other.Requires_grad {
		other.backward_fn = func() *tensor.Tensor[T] {
			return out.Grad.Neg() // -out.g
		}
	}
	return out
}

func (this *Var[T]) Mul(other *Var[T]) *Var[T] {
	out := Variable(this.Value.Mul(other.Value), this, other)
	if this.Requires_grad {
		this.backward_fn = func() *tensor.Tensor[T] {
			return other.Value.Mul(out.Grad) // other * out.g
		}
	}
	if other.Requires_grad {
		other.backward_fn = func() *tensor.Tensor[T] {
			return this.Value.Mul(out.Grad) // this * out.g
		}
	}
	return out
}

func (this *Var[T]) Pow(other *Var[T]) *Var[T] {
	out := Variable(this.Value.Pow(other.Value), this, other)
	if this.Requires_grad {
		this.backward_fn = func() *tensor.Tensor[T] {
			// out.g * other * this**(other-1)
			// or out.g * other * out / this ),
			return out.Grad.Mul(other.Value.Mul(out.Value.Div(this.Value)))
		}
	}
	if other.Requires_grad {
		other.backward_fn = func() *tensor.Tensor[T] {
			// out.g * out * this.ln()
			return out.Grad.Mul(out.Value.Mul(this.Value.Ln()))
		}
	}
	return out
}

// this/other
// => d(this): 1/other
// => d(other): (-this) / (other**2)
func (this *Var[T]) Div(other *Var[T]) *Var[T] {
	out := Variable(this.Value.Div(other.Value), this, other)
	if this.Requires_grad {
		// one := tensor.Scalar[T](1)
		this.backward_fn = func() *tensor.Tensor[T] {
			return out.Grad.Div(other.Value) // this.g += out.g / other.val
		}
	}
	if other.Requires_grad {
		other.backward_fn = func() *tensor.Tensor[T] {
			return this.Value.Neg().Div(other.Value.Mul(other.Value)) // other.g += -this.val * other.
		}
	}
	return out
}

func (this *Var[T]) MatMul(other *Var[T]) *Var[T] {
	out := Variable(this.Value.MatMul(other.Value), this, other)
	if this.Requires_grad {
		this.backward_fn = func() *tensor.Tensor[T] {
			// out.g @ other.T
			// fmt.Println(out.Grad, other.Value.Transpose())
			return out.Grad.MatMul(other.Value.Transpose())
		}
	}
	if other.Requires_grad {
		other.backward_fn = func() *tensor.Tensor[T] {
			// this.T @ out.g
			return this.Value.Transpose().AsContinuous(nil).MatMul(out.Grad)
		}
	}
	return out
}

func (this *Var[T]) Sigmoid() *Var[T] {
	out := Variable(this.Value.Sigmoid(), this)
	if this.Requires_grad {
		this.backward_fn = func() *tensor.Tensor[T] {
			one := tensor.Ones[T](out.Value.Shape()...)
			return out.Value.Mul(one.Sub(out.Value))
		}
	}
	return out
}
