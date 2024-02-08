package tensor

import (
	"bytes"
	"encoding/gob"
	"gograd/tensor/types"
	"log"
)

func (tensor *Tensor[T]) EncodeToBytes() []byte {
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)

	m := make(map[string][]T)

	ndims := len(tensor.Shape())
	shape := make([]T, ndims)
	strides := make([]T, ndims)
	dimord := make([]T, ndims)

	for i := 0; i < ndims; i++ {
		shape[i] = T(tensor.Shape()[i])
		strides[i] = T(tensor.Strides()[i])
		dimord[i] = T(tensor.Order()[i])
	}
	m["shape"] = shape
	m["strides"] = strides
	m["dimord"] = dimord
	m["data"] = tensor.Data()

	if err := enc.Encode(m); err != nil {
		log.Fatal(err)
	}

	return buf.Bytes()
}

func DecodeBytes[T types.TensorType](input []byte) *Tensor[T] {
	buf := bytes.NewBuffer(input)
	dec := gob.NewDecoder(buf)

	m := make(map[string][]T)

	if err := dec.Decode(&m); err != nil {
		log.Fatal(err)
	}

	shape := make(types.Shape, len(m["shape"]))
	strides := make([]int, len(m["shape"]))
	dimord := make([]uint16, len(m["shape"]))
	for i := 0; i < len(m["shape"]); i++ {
		shape[i] = types.Dim(m["shape"][i])
		strides[i] = int(m["strides"][i])
		dimord[i] = uint16(m["dimord"][i])
	}

	return &Tensor[T]{
		data_buff: m["data"],
		shape:     shape,
		strides:   strides,
		dim_order: dimord,
	}
}
