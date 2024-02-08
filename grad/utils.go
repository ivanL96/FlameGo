package grad

import (
	"fmt"
	"gograd/tensor"
	"gograd/tensor/types"
	"sync"
)

func ToOneHot[T types.TensorType](y *tensor.Tensor[T], classes uint) *tensor.Tensor[T] {
	ndims := len(y.Shape().Squeeze())
	if ndims > 1 {
		panic(fmt.Sprintf("Var y must have a vector-like shape, got %v", y.Shape()))
	}
	size := y.Shape()[0]
	oneHotData := make([]T, int(size)*int(classes))

	var wg sync.WaitGroup
	for i := 0; i < int(size); i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			classidx := int(y.Data()[i])
			if classidx >= int(classes) {
				panic(fmt.Sprintf(
					"too few classes. tensor has element '%v' which must be less than 'classes' %v", classidx, classes,
				))
			}
			oneHotData[i*int(classes)+classidx] = 1
		}(i)
	}
	wg.Wait()
	return tensor.CreateTensor[T](oneHotData, types.Shape{size, types.Dim(classes)})
}

// number of classes is auto detected
func ToOneHotAuto[T types.TensorType](y *tensor.Tensor[T]) *tensor.Tensor[T] {
	_max := uint(y.Max(false).Item() + 1)
	return ToOneHot(y, _max)
}
