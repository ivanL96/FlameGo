package tensor

import (
	"math/rand"
	"time"
)

func create_rand(seed int64) *rand.Rand {
	// use -1 for non deterministic rand
	if seed == -1 {
		seed = time.Now().UnixNano()
	}
	rand_source := rand.NewSource(seed)
	return rand.New(rand_source)
}

func CreateRandFloat32Slice(length int, seed int64) []float32 {
	_rand := create_rand(seed)
	slice := make([]float32, length)
	for i := range slice {
		slice[i] = _rand.Float32()
	}
	return slice
}

func CreateRandFloat64Slice(length int, seed int64) []float64 {
	_rand := create_rand(seed)
	slice := make([]float64, length)
	for i := range slice {
		slice[i] = _rand.Float64()
	}
	return slice
}

func RandomFloat64Tensor(shape Shape, seed int64) *Tensor[float64] {
	rand_tensor := InitEmptyTensor[float64](shape...)
	value := CreateRandFloat64Slice(int(rand_tensor.len), seed)
	rand_tensor.data = value
	return rand_tensor
}

func RandomFloat32Tensor(shape Shape, seed int64) *Tensor[float32] {
	rand_tensor := InitEmptyTensor[float32](shape...)
	value := CreateRandFloat32Slice(int(rand_tensor.len), seed)
	rand_tensor.data = value
	return rand_tensor
}
