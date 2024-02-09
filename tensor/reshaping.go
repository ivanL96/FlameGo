package tensor

import (
	"errors"
	"fmt"
	types "gograd/tensor/types"
	"sync"
)

// set of operations for shaping routines

func (tensor *Tensor[T]) Flatten(out *Tensor[T]) *Tensor[T] {
	if tensor.Err != nil {
		return tensor
	}
	if len(tensor.shape) == 1 {
		return tensor
	}
	outTensor, err := PrepareOutTensor(out, tensor.shape)
	if err != nil {
		tensor.Err = err
		return tensor
	}
	if tensor != outTensor {
		outTensor.SetData(tensor.data())
	}
	outTensor.shape = types.Shape{types.Dim(len(tensor.data()))}
	outTensor.strides = outTensor.shape.GetStrides()
	outTensor.dim_order = outTensor.shape.InitDimOrder()
	return outTensor
}

// Eliminates all one-sized dims in tensor.
//
// inplace operation
func (tensor *Tensor[T]) Squeeze() *Tensor[T] {
	if tensor.Err != nil {
		return tensor
	}
	if tensor.shape.IsScalarLike() {
		return tensor
	}
	tensor.shape = tensor.shape.Squeeze()
	tensor.strides = tensor.shape.GetStrides()
	tensor.dim_order = tensor.shape.InitDimOrder()
	return tensor
}

// Adds a one-sized dim to tensor.
//
// inplace operation
func (tensor *Tensor[T]) Unsqueeze(axis int) *Tensor[T] {
	if tensor.Err != nil {
		return tensor
	}
	if axis < 0 {
		axis = len(tensor.shape) - axis
		if axis < 0 {
			tensor.Err = fmt.Errorf("axis %v is not valid", axis)
			return tensor
		}
	}
	tensor.shape = tensor.shape.AddDim(uint(axis))
	tensor.strides = tensor.shape.GetStrides()
	tensor.dim_order = tensor.shape.InitDimOrder()
	return tensor
}

func (tensor *Tensor[T]) Reshape(newShape ...types.Dim) *Tensor[T] {
	if tensor.Err != nil {
		return tensor
	}
	var new_shape_prod types.Dim = 1
	for _, dim := range newShape {
		new_shape_prod *= dim
	}
	if new_shape_prod == 0 {
		tensor.Err = errors.New("shape cannot have 0 dim size")
		return tensor
	}
	if len(tensor.data()) != int(new_shape_prod) {
		tensor.Err = fmt.Errorf("cannot reshape tensor with size %v to shape %v", len(tensor.data()), newShape)
		return tensor
	}
	sh := types.Shape(newShape)
	tensor.shape = newShape
	tensor.strides = sh.GetStrides()
	tensor.dim_order = sh.InitDimOrder()
	return tensor
}

// Transposes tensor to given axes. If no axes are set, default transposing applied
func (tensor *Tensor[T]) T(axes ...uint) *Tensor[T] {
	if tensor.Err != nil {
		return tensor
	}
	n_dims := len(tensor.shape)
	if n_dims == 1 {
		return tensor
	}

	switch len(axes) {
	case 0:
		axes = make([]uint, n_dims)
		for i := range axes {
			axes[i] = uint(len(axes) - i - 1)
		}
	default:
		unique_axes := make(map[uint]bool)
		for _, a := range axes {
			if unique_axes[a] {
				tensor.Err = errors.New("repeatable axis in transpose")
				return tensor
			} else {
				unique_axes[a] = true
			}
		}

		if len(axes) < n_dims {
			tensor.Err = fmt.Errorf("too few axes provided. Expected %v, got %v", n_dims, len(axes))
			return tensor
		} else if len(axes) > n_dims {
			tensor.Err = fmt.Errorf("too many axes provided. Expected %v, got %v", n_dims, len(axes))
			return tensor
		}
	}

	outTensor := CreateTensor(tensor.data(), tensor.shape)

	for i, axis := range axes {
		outTensor.shape[i] = tensor.shape[axis]
		outTensor.strides[i] = tensor.strides[axis]
		outTensor.dim_order[i] = tensor.dim_order[axis]
	}
	return outTensor
}

// alias for .T(axes).AsContiguous()
func (tensor *Tensor[T]) TrC(axes ...uint) *Tensor[T] {
	return tensor.T(axes...).AsContiguous()
}

// TrC for 2D matrix
//
// [1,2,3][4,5,6] => [1,4][2,5][3,6]
func (tensor *Tensor[T]) TrC2D() *Tensor[T] {
	if tensor.Err != nil {
		return tensor
	}
	if len(tensor.Shape()) != 2 {
		tensor.Err = errors.New("tensor must be 2D")
		return tensor
	}
	if !tensor.IsContiguous() {
		tensor.Err = errors.New("tensor must be contiguous")
		return tensor
	}
	sh := tensor.shape
	rows := int(sh[0])
	cols := int(sh[1])
	transposed := make([]T, len(tensor.data()))
	data := tensor.data()

	var wg sync.WaitGroup
	for i := 0; i < rows; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			transpose_cont2D_loop(data, transposed, i, cols, rows)
		}(i)
	}
	wg.Wait()
	return CreateTensorNoCopy[T](transposed, types.Shape{sh[1], sh[0]})
}

// stacks tensors together. All tensors should have the same shape
func Stack[T types.TensorType](tensors ...*Tensor[T]) (*Tensor[T], error) {
	if len(tensors) < 1 {
		return nil, errors.New("at least 1 tensor is required")
	}
	if len(tensors) == 1 {
		return tensors[0], nil
	}

	// validate shapes
	other_shapes := make([]types.Shape, len(tensors))
	for i, tensor := range tensors {
		other_shapes[i] = tensor.Shape()
	}
	united_shape, err := types.StackShapes(0, other_shapes...)
	if err != nil {
		return nil, err
	}

	united := CreateEmptyTensor[T](united_shape...)
	prev_position := 0
	for i := 0; i < len(tensors); i++ {
		tensor := tensors[i]
		size := len(tensor.data())
		copy(united.data()[prev_position:prev_position+size], tensor.data())
		prev_position += size
	}
	return united, nil
}

// tensor list logic
type TensorList[T types.TensorType] struct {
	StackedTensors *Tensor[T]
}

func (tlist *TensorList[T]) Append(new_tensor *Tensor[T]) {
	new_tensor.MustAssert()
	if tlist.StackedTensors == nil {
		tlist.StackedTensors = new_tensor.Copy()
		return
	}
	upd, err := Stack[T](tlist.StackedTensors, new_tensor)
	if err != nil {
		new_tensor.Err = err
		return
	}
	tlist.StackedTensors = upd
}
