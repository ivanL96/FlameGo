package tensor

import (
	"fmt"
	"gograd/tensor/iter"
	ops "gograd/tensor/ops"
	types "gograd/tensor/types"
)

type BinaryScalarOp[T types.TensorType] func(T, T) T
type UnaryScalarOp[T types.TensorType] func(T) T

// general use Binary operator
func BaseBinElementwiseOp[T types.TensorType](
	tensor_a,
	tensor_b *Tensor[T],
	binOp BinaryScalarOp[T],
	out *Tensor[T],
) *Tensor[T] {
	var outTensor *Tensor[T]
	if IsScalarLike(tensor_a.shape) && IsScalarLike(tensor_b.shape) {
		outTensor = PrepareOutTensor(out, tensor_a.shape)
		// most trivial case (1,) & (1,)
		outTensor.data[0] = binOp(tensor_a.data[0], tensor_b.data[0])
		return outTensor
	}

	// tensors should have equal shapes or at least one of them should be scalar-like
	if !AreBroadcastable(tensor_a.shape, tensor_b.shape) {
		panic(
			fmt.Sprintf("Shapes: %x, %x are not broadcastable", tensor_a.shape, tensor_b.shape),
		)
	}
	if len(tensor_a.data) == len(tensor_b.data) {
		// same broadcastable shapes (N,M) & (N,M)
		outTensor = PrepareOutTensor(out, tensor_a.shape)

		// if two tensors are filled with same values. For example [2,2,2] and [3,3,3]
		// FIXME wrong behaviour if out tensor has bigger size than input tensors
		if Equal_1D_slices(outTensor.shape, tensor_a.shape) &&
			tensor_a.hasFlag(SameValuesFlag) && tensor_b.hasFlag(SameValuesFlag) {
			outTensor.Fill(binOp(tensor_a.data[0], tensor_b.data[0]))
			return outTensor
		}

		iter := tensor_a.CreateIterator()
		for iter.Iterate() {
			idx := iter.Next()
			outIdx := get_flat_idx_fast(outTensor.strides, idx...)
			outTensor.data[outIdx] = binOp(tensor_a.Get_fast(idx...), tensor_b.Get_fast(idx...))
		}
	} else if len(tensor_b.data) == 1 {
		// tensor_b is scalar
		// (N, M, ...) & (1,)
		outTensor = PrepareOutTensor(out, tensor_a.shape)
		value := tensor_b.data[0]
		for i, val := range tensor_a.data {
			outTensor.data[i] = binOp(val, value)
		}
	} else if len(tensor_a.data) == 1 {
		// tensor_a is scalar
		// (1,) & (N, M, ...)
		outTensor = PrepareOutTensor(out, tensor_b.shape)
		value := tensor_a.data[0]
		for i, val := range tensor_b.data {
			outTensor.data[i] = binOp(val, value)
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
		if broadcasted_tensor_a == nil {
			broadcasted_tensor_a = tensor_a
		}
		if broadcasted_tensor_b == nil {
			broadcasted_tensor_b = tensor_b
		}
		iter := broadcasted_tensor_a.CreateIterator()
		for iter.Iterate() {
			dataIndex := iter.Index()
			idx := iter.Next()
			outTensor.data[dataIndex] = binOp(
				broadcasted_tensor_a.Get_fast(idx...), broadcasted_tensor_b.Get_fast(idx...))
		}
	}
	outTensor.clearFlag(SameValuesFlag)
	return outTensor
}

func unaryElementwiseRoutine[T types.TensorType](
	tensor *Tensor[T],
	unaryOp UnaryScalarOp[T],
	out *Tensor[T],
) *Tensor[T] {
	outTensor := PrepareOutTensor(out, tensor.Shape())
	if IsScalarLike(tensor.Shape()) {
		outTensor.data[0] = unaryOp(tensor.data[0])
		return outTensor
	}
	for i, val := range tensor.data {
		outTensor.data[i] = unaryOp(val)
	}
	return outTensor
}

//
// binary ops
//

func (tensor *Tensor[T]) Add(other_tensor, out *Tensor[T]) *Tensor[T] {
	return BaseBinElementwiseOp(tensor, other_tensor, ops.Add[T], out)
}

func (tensor *Tensor[T]) Sub(other_tensor, out *Tensor[T]) *Tensor[T] {
	return BaseBinElementwiseOp(tensor, other_tensor, ops.SubAtomic[T], out)
}

func (tensor *Tensor[T]) Mul(other_tensor, out *Tensor[T]) *Tensor[T] {
	return BaseBinElementwiseOp(tensor, other_tensor, ops.MulAtomic[T], out)
}

func (tensor *Tensor[T]) Div(other_tensor, out *Tensor[T]) *Tensor[T] {
	return BaseBinElementwiseOp(tensor, other_tensor, ops.DivAtomic[T], out)
}

// unary
func (tensor *Tensor[T]) Neg(out *Tensor[T]) *Tensor[T] {
	return unaryElementwiseRoutine(tensor, ops.NegAtomic[T], out)
}

func (tensor *Tensor[T]) Sigmoid(out *Tensor[T]) *Tensor[T] {
	return unaryElementwiseRoutine(tensor, ops.SigmoidAtomic[T], out)
}

//
// matrix operations
//

func (tensor *Tensor[T]) Dot(other *Tensor[T]) *Tensor[T] {
	if len(tensor.shape) != len(other.shape) {
		panic("Tensors must have equal number of dims.")
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

	isVec2Scalar := adim0 == 1 && bdim1 == 1

	tensor = tensor.AsContinuous(nil)

	a_data := tensor.data
	b_data := other.data
	out_data := outTensor.data

	if tensor.hasFlag(UseAVXFlag) {
		other = other.Transpose().AsContinuous(nil) // needs to be in column-major format for better AVX support

		a_data := types.Any(tensor.data).([]float32)
		b_data := types.Any(other.data).([]float32)
		out_data := types.Any(outTensor.data).([]float32)

		if isVec2Scalar {
			ops.MatMul_AVX_VectorsToScalar(a_data, b_data, out_data)
		} else {
			// gen impl
			ops.MatMulNaiveImpl_AVX(a_data, b_data, tensor.shape, other.shape,
				tensor.strides, other.strides,
				out_data, outTensor.strides)
		}
	} else {
		other = other.AsContinuous(nil)
		ops.MatMulNaiveImpl(a_data, b_data, tensor.shape, other.shape,
			tensor.strides, other.strides,
			out_data, outTensor.strides)
	}
	outTensor.data = types.Any(out_data).([]T)
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
	ops.SplitTensorImpl(tensor.data, nrows, a.data, b.data, c.data, d.data)
	return
}

func UniteTensors[T types.TensorType](a, b, c, d, out *Tensor[T]) *Tensor[T] {
	out_tensor := PrepareOutTensor(out, types.Shape{a.shape[0] * 2, a.shape[0] * 2})
	ops.UniteTensors(int(a.shape[0]), a.strides[0], a.data, b.data, c.data, d.data, out_tensor.data)
	return out_tensor
}
