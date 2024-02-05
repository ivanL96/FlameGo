package grad

import (
	"fmt"
	"gograd/tensor"
	"gograd/tensor/types"
	"reflect"
)

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

type variable[T types.TensorType] struct {
	Value         *tensor.Tensor[T]
	Grad          *tensor.Tensor[T]
	backward_fn   func() *tensor.Tensor[T]
	Alias         string
	Children      []*variable[T]
	Requires_grad bool
}

// VAR init
func Variable[T types.TensorType](
	tensor_val *tensor.Tensor[T],
	children ...*variable[T],
) *variable[T] {
	v := &variable[T]{
		Value:         tensor_val.MustAssert(),
		Grad:          tensor.Zeros[T](tensor_val.Shape()...),
		Children:      children,
		Requires_grad: true,
	}
	if v.Requires_grad && intKinds[tensor_val.DType().Kind()] {
		panic("Cannot create variable of Int type that requires gradient.")
	}
	return v
}

func VarFrom[T types.TensorType](data []T, shape types.Shape) *variable[T] {
	return Variable(tensor.CreateTensor[T](data, shape))
}

func Constant[T types.TensorType](
	tensor_val *tensor.Tensor[T],
	children ...*variable[T],
) *variable[T] {
	v := Variable[T](tensor_val, children...)
	v.Requires_grad = false
	return v
}

// =================

func (v *variable[T]) SetAlias(name string) *variable[T] {
	v.Alias = name
	return v
}

func (v *variable[T]) IsLeaf() bool {
	return len(v.Children) > 0
}

func (v *variable[T]) MustAssert() *variable[T] {
	v.Value.MustAssert()
	return v
}

func (v *variable[T]) ZeroGrad() {
	v.Grad = tensor.Zeros[T](v.Value.Shape()...)
}

func (v *variable[T]) ToString() string {
	name := "Const"
	if v.Requires_grad {
		name = "variable"
	}
	return fmt.Sprintf("%v(%v)", name, v.Value.ToString())
}

// VARIABLE OPS

// reduce gradient shape
func unbroadcast[T types.TensorType](
	grad,
	other *tensor.Tensor[T],
) *tensor.Tensor[T] {
	if !grad.Shape().Equals(other.Shape()) {
		return grad.SumAlongAxis(0, true).Reshape(other.Shape()...)
	}
	return grad
}

func (this *variable[T]) Add(other *variable[T]) *variable[T] {
	out := Variable(this.Value.Add(other.Value), this, other).SetAlias("Add")
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

func (this *variable[T]) Sub(other *variable[T]) *variable[T] {
	out := Variable(this.Value.Sub(other.Value), this, other).SetAlias("Sub")
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

func (this *variable[T]) Mul(other *variable[T]) *variable[T] {
	out := Variable(this.Value.Mul(other.Value), this, other).SetAlias("Mul")
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

func (this *variable[T]) Pow(other *variable[T]) *variable[T] {
	out := Variable(this.Value.Pow(other.Value), this, other).SetAlias("Pow")
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
func (this *variable[T]) Div(other *variable[T]) *variable[T] {
	out := Variable(this.Value.Div(other.Value), this, other).SetAlias("Div")
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

func (this *variable[T]) MatMul(other *variable[T]) *variable[T] {
	out := Variable(this.Value.MatMul(other.Value), this, other).SetAlias("MatMul")
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

// activations
func (this *variable[T]) Sigmoid() *variable[T] {
	out := Variable(this.Value.Sigmoid(), this).SetAlias("Sigmoid")
	if this.Requires_grad {
		this.backward_fn = func() *tensor.Tensor[T] {
			one := tensor.Ones[T](out.Value.Shape()...)
			return unbroadcast(out.Value.Mul(one.Sub(out.Value)), this.Value)
		}
	}
	return out
}

func (this *variable[T]) Relu() *variable[T] {
	out := Variable(this.Value.Relu(), this).SetAlias("Relu")
	if this.Requires_grad {
		this.backward_fn = func() *tensor.Tensor[T] {
			expr := func(a T) T {
				if a > 0 {
					return 1
				}
				return 0
			}
			return out.Grad.Mul(out.Value.ApplyFunc(expr))
		}
	}
	return out
}

// FIXME derivative of the softmax is Jacobian of shape (class, class).
//
// The batched softmax deriv should be (batch, class).
//
// TODO treat each i-th row in batch independently and combine each i-th Jacobian using reduce (sum)
func (this *variable[T]) Softmax() *variable[T] {
	panic("To be implemented")
	nclasses := types.Dim(this.Value.Shape()[1])
	e := this.Value.Exp()
	_softmax := Variable(e.Div(e.Sum(false)), this)

	fmt.Println("this", this.Value.Shape(), "_softmax", _softmax.Value.Shape())
	_softmax.Alias = "Softmax"
	if this.Requires_grad {
		this.backward_fn = func() *tensor.Tensor[T] {
			_dot_self := _softmax.Value.MatMul(_softmax.Value.TrC())
			fmt.Println(_softmax.ToString())
			diag := tensor.Eye[T](nclasses, nclasses) //_softmax.Value.DiagFlat()
			fmt.Println(diag.ToString())
			fmt.Println("diag", diag.Shape(), "_dot_self", _dot_self.Shape())
			J := diag.Sub(_dot_self).MustAssert()
			// x := tensor.Eye[T](n, n).Sub(_softmax.Value.Unsqueeze(-1)).MustAssert()
			// fmt.Println("x", x.Shape(), "sf", _softmax.Value.Shape())
			// ds := _softmax.Value.Mul(x).MustAssert()
			// fmt.Println("ds", ds.Shape())
			// ds = _softmax.Grad.Unsqueeze(-1).Mul(ds).MustAssert()
			// fmt.Println("ds", ds.Shape())
			return J
		}
	}
	return _softmax
}

// reduce
func (this *variable[T]) Mean() *variable[T] {
	out := Variable(this.Value.Mean(false), this)
	out.Alias = "Mean"
	if this.Requires_grad {
		this.backward_fn = func() *tensor.Tensor[T] {
			filler := tensor.Scalar(T(1. / float32(this.Value.Size())))
			return out.Grad.Mul(filler)
		}
	}
	return out
}
