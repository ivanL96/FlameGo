package tensor

import (
	types "gograd/tensor/types"
	"math/rand"
	"time"
)

type RNG struct {
	Seed int64
}

// use seed -1 for non deterministic rand
func NewRNG(seed int64) *RNG {
	return &RNG{seed}
}

func createRand(seed int64) *rand.Rand {
	if seed == -1 {
		seed = time.Now().UnixNano()
	}
	randSource := rand.NewSource(seed)
	return rand.New(randSource)
}

func createRandFloat32Slice(length int, seed int64) []float32 {
	_rand := createRand(seed)
	slice := make([]float32, length)
	for i := range slice {
		slice[i] = _rand.Float32()
	}
	return slice
}

func createRandFloat64Slice(length int, seed int64) []float64 {
	_rand := createRand(seed)
	slice := make([]float64, length)
	for i := range slice {
		slice[i] = _rand.Float64()
	}
	return slice
}

func (rng *RNG) RandomFloat64(shape ...types.Dim) *Tensor[float64] {
	randTensor := CreateEmptyTensor[float64](shape...)
	value := createRandFloat64Slice(len(randTensor.data()), rng.Seed)
	randTensor.SetData(value)
	return randTensor
}

func (rng *RNG) RandomFloat32(shape ...types.Dim) *Tensor[float32] {
	randTensor := CreateEmptyTensor[float32](shape...)
	value := createRandFloat32Slice(len(randTensor.data()), rng.Seed)
	randTensor.SetData(value)
	return randTensor
}

func ShuffleData(rng *RNG, data []int) {
	_rand := createRand(rng.Seed)
	_rand.Shuffle(len(data), func(i, j int) { data[i], data[j] = data[j], data[i] })
}
