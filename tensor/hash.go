package tensor

import "gograd/tensor/types"

// https://golangprojectstructure.com/hash-functions-go-code/

func djb2[T types.TensorType](data []T) uint64 {
	hash := uint64(5381)

	for _, b := range data {
		hash += uint64(b) + hash + hash<<5
	}
	return hash
}

func sdbmHash[T types.TensorType](data []T) uint64 {
	var hash uint64

	for _, b := range data {
		hash = uint64(b) + (hash << 6) + (hash << 16) - hash
	}

	return hash
}

func (tensor *Tensor[T]) Hash() uint64 {
	return djb2(tensor.data)
}
