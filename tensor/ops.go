package tensor

import (
	"flamego/tensor/internal/cpu"
	"flamego/tensor/iter"
	ops "flamego/tensor/ops"
	types "flamego/tensor/types"
	"fmt"
)

var auto_impl cpu.Implementation = cpu.DetectImpl()

type UnaryScalarOp[T types.TensorType] func(T) T

type BinaryOp[T types.TensorType] struct {
	// required prop, contains function with scalar bin operation
	scalar func(T, T) T
	// vector is optional prop used to accelerate applying operation to vectors
	vector func(cpu.Implementation, []T, []T, []T)
	// despite one of the args will be scalar, it's generally unknown which one is exactly.
	// Therefore let the vector_to_scalar impl define the order of args.
	vector_to_scalar func(cpu.Implementation, []T, []T, []T)
}

// general use Binary operator
func BaseBinElementwiseOp[T types.TensorType](
	tensor_a,
	tensor_b *Tensor[T],
	op *BinaryOp[T],
	out *Tensor[T],
) *Tensor[T] {
	var outTensor *Tensor[T]
	// TODO if outTensor equals to a or b,  apply the *_to_const vectorized impl

	binOp, binVec, binVec2Scalar := op.scalar, op.vector, op.vector_to_scalar
	if binOp == nil {
		panic("At least op.scalar function must be set")
	}

	if IsScalarLike(tensor_a.shape) && IsScalarLike(tensor_b.shape) {
		outTensor = PrepareOutTensor(out, tensor_a.shape)
		// most trivial case (1,) & (1,)
		outTensor.data()[0] = binOp(tensor_a.data()[0], tensor_b.data()[0])
		return outTensor
	}

	are_continuous := isDimOrderInit(tensor_a.dim_order) && isDimOrderInit(tensor_b.dim_order)

	// tensors should have equal shapes or at least one of them should be scalar-like
	if !AreBroadcastable(tensor_a.shape, tensor_b.shape) {
		panic(
			fmt.Sprintf("Shapes: %x, %x are not broadcastable", tensor_a.shape, tensor_b.shape),
		)
	}
	if len(tensor_a.data()) == len(tensor_b.data()) {
		// same broadcastable shapes (N,M) & (N,M)
		outTensor = PrepareOutTensor(out, tensor_a.shape)
		out_data := outTensor.data()

		if are_continuous && binVec != nil { // vec or avx
			binVec(auto_impl, tensor_a.data(), tensor_b.data(), out_data)
		} else if !are_continuous || binVec == nil {
			iter := tensor_a.CreateIterator()
			for iter.Iterate() {
				idx := iter.Next()
				outIdx := get_flat_idx_fast(outTensor.strides, idx...)
				out_data[outIdx] = binOp(tensor_a.Get_fast(idx...), tensor_b.Get_fast(idx...))
			}
		}
	} else if len(tensor_b.data()) == 1 {
		// tensor_b is scalar
		// (N, M, ...) & (1,)
		outTensor = PrepareOutTensor(out, tensor_a.shape)
		out_data := outTensor.data()
		if binVec2Scalar == nil {
			value := tensor_b.data()[0]
			for i, val := range tensor_a.data() {
				out_data[i] = binOp(val, value)
			}
		} else {
			binVec2Scalar(auto_impl, tensor_a.data(), tensor_b.data(), out_data)
		}
	} else if len(tensor_a.data()) == 1 {
		// tensor_a is scalar
		// (1,) & (N, M, ...)
		outTensor = PrepareOutTensor(out, tensor_b.shape)
		out_data := outTensor.data()
		if binVec2Scalar == nil {
			value := tensor_a.data()[0]
			for i, val := range tensor_b.data() {
				out_data[i] = binOp(value, val)
			}
		} else {
			binVec2Scalar(auto_impl, tensor_b.data(), tensor_a.data(), out_data)
		}
	} else {
		// (A, B ...) & (N, M, ...)
		// apply operation for non scalar broadcastable tensors
		broadcasted_shape := BroadcastShapes(tensor_a.Shape(), tensor_b.Shape())

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
		outTensor = PrepareOutTensor(out, broadcasted_shape)
		out_data := outTensor.data()
		if broadcasted_tensor_a == nil {
			broadcasted_tensor_a = tensor_a
		}
		if broadcasted_tensor_b == nil {
			broadcasted_tensor_b = tensor_b
		}
		if binVec != nil && are_continuous {
			binVec(auto_impl, broadcasted_tensor_a.data(), broadcasted_tensor_b.data(), out_data)
		} else if binVec == nil || !are_continuous {
			iter := broadcasted_tensor_a.CreateIterator()
			for iter.Iterate() {
				dataIndex := iter.Index()
				idx := iter.Next()
				out_data[dataIndex] = binOp(
					broadcasted_tensor_a.Get_fast(idx...), broadcasted_tensor_b.Get_fast(idx...))
			}
		}
	}
	return outTensor
}

func unaryElementwiseRoutine[T types.TensorType](
	tensor *Tensor[T],
	unaryOp UnaryScalarOp[T],
	out *Tensor[T],
) *Tensor[T] {
	outTensor := PrepareOutTensor(out, tensor.Shape())
	if IsScalarLike(tensor.Shape()) {
		outTensor.data()[0] = unaryOp(tensor.data()[0])
		return outTensor
	}
	for i, val := range tensor.data() {
		outTensor.data()[i] = unaryOp(val)
	}
	return outTensor
}

//
// binary ops
//

func (tensor *Tensor[T]) Add(other_tensor *Tensor[T], out ...*Tensor[T]) *Tensor[T] {
	add := BinaryOp[T]{
		scalar: ops.AddAtomic[T],
		vector: cpu.Add[T],
	}
	return BaseBinElementwiseOp(tensor, other_tensor, &add, get_param(out...))
}

func (tensor *Tensor[T]) Sub(other_tensor *Tensor[T], out ...*Tensor[T]) *Tensor[T] {
	sub := BinaryOp[T]{
		scalar: ops.SubAtomic[T],
		vector: cpu.Sub[T],
	}
	return BaseBinElementwiseOp(tensor, other_tensor, &sub, get_param(out...))
}

func (tensor *Tensor[T]) Mul(other_tensor *Tensor[T], out ...*Tensor[T]) *Tensor[T] {
	mul := BinaryOp[T]{
		scalar:           ops.MulAtomic[T],
		vector:           cpu.Mul[T],
		vector_to_scalar: cpu.MulToConst[T],
	}
	return BaseBinElementwiseOp(tensor, other_tensor, &mul, get_param(out...))
}

func (tensor *Tensor[T]) Div(other_tensor *Tensor[T], out ...*Tensor[T]) *Tensor[T] {
	div := BinaryOp[T]{
		scalar: ops.DivAtomic[T],
		vector: cpu.Div[T],
	}
	return BaseBinElementwiseOp(tensor, other_tensor, &div, get_param(out...))
}

func (tensor *Tensor[T]) Pow(other_tensor *Tensor[T], out ...*Tensor[T]) *Tensor[T] {
	pow := BinaryOp[T]{
		scalar: ops.PowAtomic[T],
		vector: cpu.Pow[T],
	}
	return BaseBinElementwiseOp(tensor, other_tensor, &pow, get_param(out...))
}

// unary
func (tensor *Tensor[T]) Neg(out ...*Tensor[T]) *Tensor[T] {
	return unaryElementwiseRoutine(tensor, ops.NegAtomic[T], get_param(out...))
}

func (tensor *Tensor[T]) Sigmoid(out ...*Tensor[T]) *Tensor[T] {
	return unaryElementwiseRoutine(tensor, ops.SigmoidAtomic[T], get_param(out...))
}

//
// MATRIX OPERATIONS
//

func (tensor *Tensor[T]) Dot(other *Tensor[T]) *Tensor[T] {
	if len(tensor.shape) != len(other.shape) {
		panic("Tensors must have equal number of dims.")
	}

	if len(tensor.shape) == 2 {
		return tensor.MatMul(other)
	}

	outer_dims_a := tensor.shape[:len(tensor.shape)-2]
	outer_dims_b := other.shape[:len(tensor.shape)-2]
	if !Equal_1D_slices(outer_dims_a, outer_dims_b) {
		panic("Tensors must have equal outer dims. ")
	}
	var outer_shape_prod types.Dim = 1
	for _, dim := range outer_dims_a {
		outer_shape_prod *= dim
	}

	tensors_stack := make([]*Tensor[T], int(outer_shape_prod))
	shape_iter := iter.CreateIterator(int(outer_shape_prod), outer_dims_a)
	for shape_iter.Iterate() {
		i := shape_iter.Index()
		idx := shape_iter.Next()
		mat_a := tensor.Index(idx...)
		mat_b := other.Index(idx...)
		out := mat_a.MatMul(mat_b)
		tensors_stack[i] = out
	}

	out := Unite(tensors_stack...)

	out_shape := make(types.Shape, len(tensor.shape))
	copy(out_shape[:len(outer_dims_a)], outer_dims_a)
	out_shape[len(out_shape)-2] = tensor.shape[len(tensor.shape)-2]
	out_shape[len(out_shape)-1] = other.shape[len(other.shape)-1]

	out = out.Reshape(out_shape...)
	return out
}

func (tensor *Tensor[T]) MatMul(other *Tensor[T]) *Tensor[T] {
	if len(tensor.shape) != 2 || len(other.shape) != 2 {
		panic("Tensors must be two-dim.")
	}
	if tensor.shape[1] != other.shape[0] {
		panic(fmt.Sprintf(
			"Tensors inner shapes are different. %v != %v", tensor.shape[1], other.shape[0],
		))
	}
	adim0, bdim1 := tensor.shape[0], other.shape[1]
	outTensor := CreateEmptyTensor[T](adim0, bdim1)

	// isVec2Scalar := adim0 == 1 && bdim1 == 1

	tensor = tensor.AsContinuous(nil)
	// needs to be in column-major format for the AVX support
	other = other.Transpose().AsContinuous(nil)

	a_data := tensor.data()
	b_data := other.data()
	out_data := outTensor.data()
	// 	ops.MatMul_AVX_VectorsToScalar(a_data, b_data, out_data)
	// gen impl
	a_data_ := types.Any(a_data).([]float32)
	b_data_ := types.Any(b_data).([]float32)
	out_data_ := types.Any(out_data).([]float32)
	ops.MatMulNaiveImpl_GEN(
		auto_impl, a_data_, b_data_, tensor.shape, other.shape,
		tensor.strides, other.strides, out_data_, outTensor.strides)

	// 	ops.MatMulNaiveImpl(a_data, b_data, tensor.shape, other.shape,
	// 		tensor.strides, other.strides,
	// 		out_data, outTensor.strides)
	// }
	outTensor.data_buff = types.Any(outTensor.data()).([]T)
	return outTensor
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
	a = PrepareOutTensor(outA, types.Shape{rowleft, rowleft})
	b = PrepareOutTensor(outB, types.Shape{rowleft, rowright})
	c = PrepareOutTensor(outC, types.Shape{rowright, rowleft})
	d = PrepareOutTensor(outD, types.Shape{rowright, rowright})
	ops.SplitTensorImpl(tensor.data(), nrows, a.data(), b.data(), c.data(), d.data())
	return
}

func UniteTensors[T types.TensorType](a, b, c, d, out *Tensor[T]) *Tensor[T] {
	out_tensor := PrepareOutTensor(out, types.Shape{a.shape[0] * 2, a.shape[0] * 2})
	ops.UniteTensors(int(a.shape[0]), a.strides[0], a.data(), b.data(), c.data(), d.data(), out_tensor.data())
	return out_tensor
}
