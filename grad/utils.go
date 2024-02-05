package grad

import (
	"fmt"
	"gograd/tensor/types"
)

func (y *variable[T]) ToOneHot(classes uint) *variable[T] {
	ndims := len(y.Value.Shape().Squeeze())
	if ndims > 1 {
		panic(fmt.Sprintf("variable y must have a vector-like shape, got %v", y.Value.Shape()))
	}
	size := y.Value.Shape()[0]
	oneHotData := make([]T, int(size)*int(classes))
	for i := 0; i < int(size); i++ {
		classidx := int(y.Value.Data()[i])
		if classidx >= int(classes) {
			panic(fmt.Sprintf("too few classes. tensor has element '%v' which must be less than 'classes' %v", classidx, classes))
		}
		oneHotData[i*int(classes)+classidx] = 1
	}
	return VarFrom[T](oneHotData, types.Shape{size, types.Dim(classes)})
}

// number of classes is auto detected
func (y *variable[T]) ToOneHotAuto() *variable[T] {
	_max := uint(y.Value.Max(false).Data()[0] + 1)
	return y.ToOneHot(_max)
}
