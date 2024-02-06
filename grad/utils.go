package grad

import (
	"fmt"
	"gograd/tensor/types"
	"sync"
)

func (y *Var[T]) ToOneHot(classes uint) *Var[T] {
	ndims := len(y.Value.Shape().Squeeze())
	if ndims > 1 {
		panic(fmt.Sprintf("Var y must have a vector-like shape, got %v", y.Value.Shape()))
	}
	size := y.Value.Shape()[0]
	oneHotData := make([]T, int(size)*int(classes))

	var wg sync.WaitGroup
	for i := 0; i < int(size); i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			classidx := int(y.Value.Data()[i])
			if classidx >= int(classes) {
				panic(fmt.Sprintf("too few classes. tensor has element '%v' which must be less than 'classes' %v", classidx, classes))
			}
			oneHotData[i*int(classes)+classidx] = 1
		}(i)
	}
	wg.Wait()
	return VarFrom[T](oneHotData, types.Shape{size, types.Dim(classes)})
}

// number of classes is auto detected
func (y *Var[T]) ToOneHotAuto() *Var[T] {
	_max := uint(y.Value.Max(false).Item() + 1)
	return y.ToOneHot(_max)
}
