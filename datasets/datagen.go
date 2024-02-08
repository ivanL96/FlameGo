package datasets

import "math"

func GenerateFunction(
	f func(a, b float32) float32,
	min, max float32,
	amount int,
	seed int64,
) *DataSet[float32] {

	if min >= max {
		panic("min must be less than max")
	}

	_rand := createRand(seed)
	x := make([][]float32, amount)
	result := make([][]float32, amount)

	for i := 0; i < amount; i++ {
		a := math.Floor(float64(min + (_rand.Float32() * (max - min))))
		b := math.Floor(float64(min + (_rand.Float32() * (max - min))))

		result[i] = []float32{f(float32(a), float32(b))}
		x[i] = []float32{float32(a), float32(b)}
	}

	return &DataSet[float32]{
		x, result,
	}
	// return x, result
}
