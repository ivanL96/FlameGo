package tensor

import (
	"errors"
	"fmt"
	ops "gograd/tensor/internal"
	"gograd/tensor/internal/device"
	types "gograd/tensor/types"
	"reflect"
)

var AUTO_IMPL device.Implementation = *device.DetectImpl().ShowDebugInfo()

// general use Binary operator
func BaseBinElementwiseOp[T types.TensorType](
	tensor_a,
	tensor_b *Tensor[T],
	// contains function with scalar bin operation
	scalar_impl func(T, T) T,
	// vector is used to accelerate applying operation to vectors
	vector_impl func(device.Implementation, []T, []T, []T),
	out *Tensor[T],
) *Tensor[T] {
	if tensor_a.Err != nil {
		return tensor_a
	}
	if tensor_b.Err != nil {
		return tensor_b
	}
	if out != nil && out.Err != nil {
		return out
	}
	var outTensor *Tensor[T]
	var err error

	// TODO if outTensor equals to a or b,  apply the *_to_const vectorized impl
	// TODO try to vectorize operations for non contiguous tensors. Right now it falls back to scalar impl which is slow
	if scalar_impl == nil && vector_impl == nil {
		panic("no implementation found")
	}

	if tensor_a.shape.IsScalarLike() && tensor_b.shape.IsScalarLike() {
		// sometimes it's important to keep dims for scalar like tensors
		out_shape := tensor_b.shape
		if len(tensor_a.shape) > len(tensor_b.shape) {
			out_shape = tensor_a.shape
		}
		outTensor, err = PrepareOutTensor(out, out_shape)
		if err != nil {
			tensor_a.Err = err
			return tensor_a
		}
		// most trivial case (1,) & (1,)
		vector_impl(AUTO_IMPL, tensor_a.data(), tensor_b.data(), outTensor.data())
		return outTensor
	}

	are_contiguous := tensor_a.IsContiguous() && tensor_b.IsContiguous()

	if len(tensor_a.data()) == len(tensor_b.data()) {
		// same broadcastable shapes (N,M) & (N,M)
		outTensor, err = PrepareOutTensor(out, tensor_a.shape)
		if err != nil {
			tensor_a.Err = err
			return tensor_a
		}
		out_data := outTensor.data()

		if are_contiguous && vector_impl != nil { // vec or avx
			vector_impl(AUTO_IMPL, tensor_a.data(), tensor_b.data(), out_data)
		} else if !are_contiguous || vector_impl == nil {
			iter := tensor_a.CreateIterator()
			for iter.Iterate() {
				idx := iter.Next()
				outIdx := get_flat_idx_fast(outTensor.strides, idx...)
				out_data[outIdx] = scalar_impl(tensor_a.Get_fast(idx...), tensor_b.Get_fast(idx...))
			}
		}
	} else if tensor_b.shape.IsScalarLike() {
		// tensor_b is scalar
		// (N, M, ...) & (1,)
		outTensor, err = PrepareOutTensor(out, tensor_a.shape)
		if err != nil {
			tensor_a.Err = err
			return tensor_a
		}
		out_data := outTensor.data()
		// if vector_to_scalar_impl == nil {
		// 	value := tensor_b.Item()
		// 	for i, val := range tensor_a.data() {
		// 		out_data[i] = scalar_impl(val, value)
		// 	}
		// } else {
		vector_impl(AUTO_IMPL, tensor_a.data(), tensor_b.data(), out_data)
		// }
	} else if tensor_a.shape.IsScalarLike() {
		// tensor_a is scalar
		// (1,) & (N, M, ...)
		outTensor, err = PrepareOutTensor(out, tensor_b.shape)
		if err != nil {
			tensor_a.Err = err
			return tensor_a
		}
		out_data := outTensor.data()
		// if vector_to_scalar_impl == nil {
		// 	value := tensor_a.Item()
		// 	for i, val := range tensor_b.data() {
		// 		out_data[i] = scalar_impl(value, val)
		// 	}
		// } else {
		vector_impl(AUTO_IMPL, tensor_b.data(), tensor_a.data(), out_data)
		// }
	} else {
		// tensors should have equal shapes or at least one of them should be scalar-like
		if !tensor_a.shape.AreBroadcastable(tensor_b.shape) {
			tensor_a.Err = fmt.Errorf("shapes: %v, %v are not broadcastable", tensor_a.shape, tensor_b.shape)
			return tensor_a
		}

		// both tensors are not scalar but have completely different shapes
		// (A, B ...) & (N, M, ...)
		// apply operation for non scalar broadcastable tensors
		broadcasted_shape := tensor_a.Shape().BroadcastShapes(tensor_b.Shape())

		// determine which tensor (at least 1) should be broadcasted
		var broadcasted_tensor_a *Tensor[T] = nil
		var broadcasted_tensor_b *Tensor[T] = nil
		for i, brc_dim := range broadcasted_shape {
			if broadcasted_tensor_a != nil && broadcasted_tensor_b != nil {
				continue
			}
			if i < len(tensor_a.shape) && tensor_a.shape[i] < brc_dim && broadcasted_tensor_a == nil {
				// need to broadcast tensor_a
				broadcasted_tensor_a = tensor_a.Broadcast(broadcasted_shape...)
			}
			if i < len(tensor_b.shape) && tensor_b.shape[i] < brc_dim && broadcasted_tensor_b == nil {
				// need to broadcast tensor_b
				broadcasted_tensor_b = tensor_b.Broadcast(broadcasted_shape...)
			}
		}
		outTensor, err = PrepareOutTensor(out, broadcasted_shape)
		if err != nil {
			tensor_a.Err = err
			return tensor_a
		}
		out_data := outTensor.data()
		if broadcasted_tensor_a == nil {
			broadcasted_tensor_a = tensor_a
		}
		if broadcasted_tensor_b == nil {
			broadcasted_tensor_b = tensor_b
		}
		if vector_impl != nil && are_contiguous {
			vector_impl(AUTO_IMPL, broadcasted_tensor_a.data(), broadcasted_tensor_b.data(), out_data)
		} else if vector_impl == nil || !are_contiguous {
			iter := broadcasted_tensor_a.CreateIterator()
			for iter.Iterate() {
				dataIndex := iter.Index()
				idx := iter.Next()
				out_data[dataIndex] = scalar_impl(
					broadcasted_tensor_a.Get_fast(idx...), broadcasted_tensor_b.Get_fast(idx...))
			}
		}
	}
	return outTensor
}

func unaryElementwiseRoutine[T types.TensorType](
	tensor *Tensor[T],
	scalar_impl func(T) T,
	vector_impl func(device.Implementation, []T, []T),
	out *Tensor[T],
) *Tensor[T] {
	if tensor.Err != nil {
		return tensor
	}
	if scalar_impl == nil && vector_impl == nil {
		panic("no implementation found")
	}
	outTensor, err := PrepareOutTensor(out, tensor.Shape())
	if err != nil {
		tensor.Err = err
		return tensor
	}
	if tensor.shape.IsScalarLike() && scalar_impl != nil {
		outTensor.data()[0] = scalar_impl(tensor.Item())
		return outTensor
	}
	if vector_impl == nil {
		for i, val := range tensor.data() {
			outTensor.data()[i] = scalar_impl(val)
		}
	} else {
		vector_impl(AUTO_IMPL, tensor.data(), outTensor.data())
	}
	return outTensor
}

//
// binary ops
//

func (tensor *Tensor[T]) Add(other_tensor *Tensor[T], out ...*Tensor[T]) *Tensor[T] {
	return BaseBinElementwiseOp(tensor, other_tensor, ops.AddAtomic[T], device.Add[T], get_param(out...))
}

func (tensor *Tensor[T]) Sub(other_tensor *Tensor[T], out ...*Tensor[T]) *Tensor[T] {
	return BaseBinElementwiseOp(tensor, other_tensor, ops.SubAtomic[T], device.Sub[T], get_param(out...))
}

func (tensor *Tensor[T]) Mul(other_tensor *Tensor[T], out ...*Tensor[T]) *Tensor[T] {
	return BaseBinElementwiseOp(tensor, other_tensor, ops.MulAtomic[T], device.Mul[T], get_param(out...))
}

func (tensor *Tensor[T]) Div(other_tensor *Tensor[T], out ...*Tensor[T]) *Tensor[T] {
	return BaseBinElementwiseOp(tensor, other_tensor, ops.DivAtomic[T], device.Div[T], get_param(out...))
}

func (tensor *Tensor[T]) Pow(other_tensor *Tensor[T], out ...*Tensor[T]) *Tensor[T] {
	return BaseBinElementwiseOp(tensor, other_tensor, ops.PowAtomic[T], device.Pow[T], get_param(out...))
}

// unary
func (tensor *Tensor[T]) Neg(out ...*Tensor[T]) *Tensor[T] {
	return unaryElementwiseRoutine(tensor, ops.NegAtomic[T], device.Neg[T], get_param(out...))
}

func (tensor *Tensor[T]) Sigmoid(out ...*Tensor[T]) *Tensor[T] {
	return unaryElementwiseRoutine(tensor, ops.SigmoidAtomic[T], device.Sigmoid[T], get_param(out...))
}

func (tensor *Tensor[T]) Ln(out ...*Tensor[T]) *Tensor[T] {
	return unaryElementwiseRoutine(tensor, ops.LnAtomic[T], nil, get_param(out...))
}

// combination of Ln().Neg()
func (tensor *Tensor[T]) LnNeg(out ...*Tensor[T]) *Tensor[T] {
	return unaryElementwiseRoutine(tensor, ops.LnNegAtomic[T], device.LnNeg[T], get_param(out...))
}

func (tensor *Tensor[T]) Relu(out ...*Tensor[T]) *Tensor[T] {
	return unaryElementwiseRoutine(tensor, ops.ReluAtomic[T], device.Relu[T], get_param(out...))
}

func (tensor *Tensor[T]) Exp(out ...*Tensor[T]) *Tensor[T] {
	return unaryElementwiseRoutine(tensor, ops.ExpAtomic[T], device.Exp[T], get_param(out...))
}

func (tensor *Tensor[T]) Clip(min, max float32, out ...*Tensor[T]) *Tensor[T] {
	clip_fn := func(v T) T {
		if v < T(min) {
			return T(min)
		}
		if v > T(max) {
			return T(max)
		}
		return v
	}
	return tensor.ApplyFunc(clip_fn, out...)
}

func (tensor *Tensor[T]) Softmax(out *Tensor[T]) *Tensor[T] {
	out, err := PrepareOutTensor(out, tensor.shape)
	if err != nil {
		return tensor
	}
	tensor = tensor.AsContiguous()
	device.Softmax[T](AUTO_IMPL, tensor.data(), out.data(), tensor.Strides())
	return out
}

//
// MATRIX OPERATIONS
//

func (tensor *Tensor[T]) Dot(other *Tensor[T]) *Tensor[T] {
	if tensor.Err != nil {
		return tensor
	}
	if other.Err != nil {
		return other
	}
	if len(tensor.shape) != len(other.shape) {
		tensor.Err = errors.New("tensors must have equal number of dims")
		return tensor
	}

	if len(tensor.shape) == 2 {
		return tensor.MatMul(other)
	}

	outer_dims_a := tensor.shape[:len(tensor.shape)-2]
	outer_dims_b := other.shape[:len(tensor.shape)-2]
	if !outer_dims_a.Equals(outer_dims_b) {
		tensor.Err = errors.New("tensors must have equal outer dims")
		return tensor
	}
	var outer_shape_prod types.Dim = 1
	for _, dim := range outer_dims_a {
		outer_shape_prod *= dim
	}
	tensors_stack := make([]*Tensor[T], int(outer_shape_prod))
	shape_iter := CreateIterator(int(outer_shape_prod), outer_dims_a)

	for shape_iter.Iterate() {
		i := shape_iter.Index()
		idx := shape_iter.Next()
		mat_a := tensor.Index(idx...)
		mat_b := other.Index(idx...)
		out := mat_a.MatMul(mat_b)
		tensors_stack[i] = out
	}

	out, err := Stack(tensors_stack...)
	if err != nil {
		tensor.Err = err
		return tensor
	}

	out_shape := make(types.Shape, len(tensor.shape))
	copy(out_shape[:len(outer_dims_a)], outer_dims_a)
	out_shape[len(out_shape)-2] = tensor.shape[len(tensor.shape)-2]
	out_shape[len(out_shape)-1] = other.shape[len(other.shape)-1]

	return out.Reshape(out_shape...)
}

func (tensor *Tensor[T]) MatMul(other *Tensor[T]) *Tensor[T] {
	if tensor.Err != nil {
		return tensor
	}
	if other.Err != nil {
		return other
	}
	if tensor.DType().Kind() != reflect.Float32 {
		tensor.Err = errors.New("matMul() only supports tensors of type []float32")
		return tensor
	}
	if len(tensor.shape) != 2 || len(other.shape) != 2 {
		tensor.Err = errors.New("tensors must be two-dim")
		return tensor
	}
	if tensor.shape[1] != other.shape[0] {
		tensor.Err = fmt.Errorf(
			"tensors inner shapes are different. %v != %v", tensor.shape[1], other.shape[0],
		)
		return tensor
	}
	// if one of tensors is scalar, matmul converges to Mul()
	if tensor.shape.IsScalarLike() || other.shape.IsScalarLike() {
		return tensor.Mul(other)
	}
	adim0, bdim1 := tensor.shape[0], other.shape[1]
	out_data := make([]T, int(adim0*bdim1))
	out_shape := types.Shape{adim0, bdim1}

	// isVec2Scalar := adim0 == 1 && bdim1 == 1

	tensor = tensor.AsContiguous()
	// needs to be in column-major format for the AVX support
	if other.IsContiguous() {
		other = other.TrC2D()
	} else {
		other = other.TrC()
	}

	device.MatMul(
		AUTO_IMPL,
		tensor.data(),
		other.data(),
		out_data,
		tensor.shape,
		other.shape,
		tensor.strides,
		other.strides,
		out_shape.GetStrides(),
	)
	return CreateTensor(out_data, out_shape)
}

func SplitTensor[T types.TensorType](
	tensor, outA, outB, outC, outD *Tensor[T],
) (a, b, c, d *Tensor[T]) {
	if len(tensor.shape) != 2 {
		panic("Tensor must have (N,N) shape")
	}
	nrows := int(tensor.shape[0])
	rowleft := types.Dim(nrows / 2)
	rowright := rowleft
	if nrows%2 != 0 {
		rowright = types.Dim(nrows) - rowleft
	}
	var err error
	a, err = PrepareOutTensor(outA, types.Shape{rowleft, rowleft})
	a.Err = err
	b, err = PrepareOutTensor(outB, types.Shape{rowleft, rowright})
	b.Err = err
	c, err = PrepareOutTensor(outC, types.Shape{rowright, rowleft})
	c.Err = err
	d, err = PrepareOutTensor(outD, types.Shape{rowright, rowright})
	d.Err = err
	if t := AnyErrors(a, b, c, d); t != nil {
		return
	}
	ops.SplitTensorImpl(tensor.data(), nrows, a.data(), b.data(), c.data(), d.data())
	return
}

func UniteTensors[T types.TensorType](a, b, c, d, out *Tensor[T]) *Tensor[T] {
	out_tensor, err := PrepareOutTensor(out, types.Shape{a.shape[0] * 2, a.shape[0] * 2})
	if err != nil {
		panic(err)
	}
	ops.UniteTensors(int(a.shape[0]), a.strides[0], a.data(), b.data(), c.data(), d.data(), out_tensor.data())
	return out_tensor
}
