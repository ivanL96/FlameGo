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

func (tensor *Tensor[T]) Squeeze(out *Tensor[T]) *Tensor[T] {
	if tensor.Err != nil {
		return tensor
	}
	if tensor.shape.IsScalarLike() {
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
	outTensor.shape = tensor.shape.Squeeze()
	outTensor.strides = outTensor.shape.GetStrides()
	outTensor.dim_order = outTensor.shape.InitDimOrder()
	return outTensor
}

func (tensor *Tensor[T]) Unsqueeze(axis uint, out *Tensor[T]) *Tensor[T] {
	if tensor.Err != nil {
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
	outTensor.shape = tensor.shape.AddDim(axis)
	outTensor.strides = outTensor.shape.GetStrides()
	outTensor.dim_order = outTensor.shape.InitDimOrder()
	return outTensor
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

// alias for .T(axes).AsContinuous()
func (tensor *Tensor[T]) TrC(axes ...uint) *Tensor[T] {
	return tensor.T(axes...).AsContinuous()
}

// TrC for 2D matrix
// [1,2,3][4,5,6] => [1,4][2,5][3,6]
func (tensor *Tensor[T]) TrC2D() *Tensor[T] {
	if tensor.Err != nil {
		return tensor
	}
	if len(tensor.Shape()) != 2 {
		tensor.Err = errors.New("tensor must be 2D")
		return tensor
	}
	if !tensor.IsContinuous() {
		tensor.Err = errors.New("tensor must be continuous")
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
	if len(tensors) < 2 {
		return nil, errors.New("at least 2 tensors is required")
	}

	// validate shapes
	var prev_shape types.Shape
	for _, tensor := range tensors {
		if prev_shape == nil {
			prev_shape = tensor.shape
			continue
		}
		if !tensor.shape.Equals(prev_shape) {
			return nil, errors.New("all shapes must be equal")
		}
	}

	united_shape := make(types.Shape, len(tensors[0].shape)+1)
	united_shape[0] = types.Dim(len(tensors))
	copy(united_shape[1:], tensors[0].shape)

	united := CreateEmptyTensor[T](united_shape...)
	var wg sync.WaitGroup
	for i := 0; i < len(tensors); i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			tensor := tensors[i]
			size := len(tensor.data())
			copy(united.data()[size*i:size*(i+1)], tensor.data())
		}(i)
	}
	wg.Wait()
	return united, nil
}
