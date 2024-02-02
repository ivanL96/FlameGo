package tensor

import (
	"crypto/md5"
	"encoding/hex"
)

// https://golangprojectstructure.com/hash-functions-go-code/

// returns unique hash based on tensors data, shape and dim order
func (tensor *Tensor[T]) Id() (string, error) {
	if tensor.Err != nil {
		return "", tensor.Err
	}
	data := tensor.data()
	shape := tensor.Shape()
	dimord := tensor.dim_order

	input := make([]byte, len(data)+len(shape)+len(dimord))
	for i := 0; i < len(data); i++ {
		input[i] = byte(data[i])
	}
	for i, v := range shape {
		input[len(data)+i] = byte(v)
	}
	for i, v := range dimord {
		input[len(data)+len(dimord)+i] = byte(v)
	}
	hash := md5.Sum(input)
	str := hex.EncodeToString(hash[:])
	return str, nil
}
