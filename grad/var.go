package grad

import (
	"fmt"
	"gograd/tensor"
	"gograd/tensor/types"
	"reflect"
)

type Var[T types.TensorType] struct {
	Alias         string
	Children      []*Var[T]
	Value         *tensor.Tensor[T]
	Grad          *tensor.Tensor[T]
	backward_fn   func() *tensor.Tensor[T]
	Requires_grad bool
}

var intKinds map[reflect.Kind]bool = map[reflect.Kind]bool{
	reflect.Uint:   true,
	reflect.Uint8:  true,
	reflect.Uint16: true,
	reflect.Uint32: true,
	reflect.Uint64: true,
	reflect.Int:    true,
	reflect.Int8:   true,
	reflect.Int16:  true,
	reflect.Int32:  true,
	reflect.Int64:  true,
}

func Variable[T types.TensorType](
	tensor_val *tensor.Tensor[T],
	children ...*Var[T],
) *Var[T] {
	v := &Var[T]{
		Value: tensor_val,
		// Grad:  tensor.Scalar[T](0),
		Grad:          tensor.Zeros[T](tensor_val.Shape()...),
		Children:      children,
		Requires_grad: true,
	}
	if v.Requires_grad && intKinds[tensor_val.DType().Kind()] {
		panic("Cannot create variable of Int type that requires gradient.")
	}
	return v
}

func Constant[T types.TensorType](
	tensor_val *tensor.Tensor[T],
	children ...*Var[T],
) *Var[T] {
	v := Variable[T](tensor_val, children...)
	v.Requires_grad = false
	return v
}

func (v *Var[T]) ZeroGrad() {
	v.Grad = tensor.Zeros[T](v.Value.Shape()...)
}

func (v *Var[T]) ToString() string {
	name := "Const"
	if v.Requires_grad {
		name = "Var"
	}
	return fmt.Sprintf("%v(%v)", name, v.Value.ToString())
}

// VARIABLE OPS

// reduce gradient dimensions
func unbroadcast[T types.TensorType](
	grad,
	other *tensor.Tensor[T],
) *tensor.Tensor[T] {
	if !grad.Shape().Equals(other.Shape()) {
		return grad.SumAlongAxis(0, true).Reshape(other.Shape()...)
	}
	return grad
}

func (this *Var[T]) Add(other *Var[T]) *Var[T] {
	out := Variable(this.Value.Add(other.Value), this, other)
	out.Alias = "Add"
	if this.Requires_grad {
		this.backward_fn = func() *tensor.Tensor[T] {
			return unbroadcast(out.Grad, this.Value)
		}
	}
	if other.Requires_grad {
		other.backward_fn = func() *tensor.Tensor[T] {
			return unbroadcast(out.Grad, other.Value)
		}
	}
	return out
}

func (this *Var[T]) Sub(other *Var[T]) *Var[T] {
	out := Variable(this.Value.Sub(other.Value), this, other)
	if this.Requires_grad {
		this.backward_fn = func() *tensor.Tensor[T] {
			// out.g
			return unbroadcast(out.Grad, this.Value)
		}
	}
	if other.Requires_grad {
		other.backward_fn = func() *tensor.Tensor[T] {
			// -out.g
			return unbroadcast(out.Grad.Neg(), other.Value)
		}
	}
	return out
}

func (this *Var[T]) Mul(other *Var[T]) *Var[T] {
	out := Variable(this.Value.Mul(other.Value), this, other)
	if this.Requires_grad {
		this.backward_fn = func() *tensor.Tensor[T] {
			grad := other.Value.Mul(out.Grad) // other * out.g
			return unbroadcast(grad, this.Value)
		}
	}
	if other.Requires_grad {
		other.backward_fn = func() *tensor.Tensor[T] {
			grad := this.Value.Mul(out.Grad) // this * out.g
			return unbroadcast(grad, other.Value)
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
			grad := out.Grad.Mul(other.Value.Mul(out.Value.Div(this.Value)))
			return unbroadcast(grad, this.Value)
		}
	}
	if other.Requires_grad {
		other.backward_fn = func() *tensor.Tensor[T] {
			// out.g * out * this.ln()
			grad := out.Grad.Mul(out.Value.Mul(this.Value.Ln()))
			return unbroadcast(grad, other.Value)
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
			grad := out.Grad.Div(other.Value) // this.g += out.g / other.val
			return unbroadcast(grad, this.Value)

		}
	}
	if other.Requires_grad {
		other.backward_fn = func() *tensor.Tensor[T] {
			grad := this.Value.Neg().Div(other.Value.Mul(other.Value)) // other.g += -this.val * other.
			return unbroadcast(grad, other.Value)
		}
	}
	return out
}

func (this *Var[T]) MatMul(other *Var[T]) *Var[T] {
	out := Variable(this.Value.MatMul(other.Value), this, other)
	out.Alias = "MatMul"
	if this.Requires_grad {
		this.backward_fn = func() *tensor.Tensor[T] {
			// out.g @ other.T
			return out.Grad.MatMul(other.Value.T())
		}
	}
	if other.Requires_grad {
		other.backward_fn = func() *tensor.Tensor[T] {
			// this.T @ out.g
			return this.Value.TrC().MatMul(out.Grad)
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

func (this *Var[T]) Mean() *Var[T] {
	out := Variable(this.Value.Mean(false), this)
	if this.Requires_grad {
		this.backward_fn = func() *tensor.Tensor[T] {
			filler := tensor.Zeros[T](this.Value.Shape()...).Fill(
				T(1. / float32(this.Value.Size())),
			)
			return out.Grad.Mul(filler)
		}
	}
	return out
}
