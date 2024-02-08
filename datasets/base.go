package datasets

import (
	"gograd/tensor"
	"gograd/tensor/types"
	"math/rand"
	"time"
)

type DataSet[T types.TensorType] struct {
	X [][]T
	Y [][]T
}

func createRand(seed int64) *rand.Rand {
	// use -1 for non deterministic rand
	if seed == -1 {
		seed = time.Now().UnixNano()
	}
	randSource := rand.NewSource(seed)
	return rand.New(randSource)
}

func (dset *DataSet[T]) Shuffle(rng *tensor.RNG) *DataSet[T] {
	recordsX := dset.X
	recordsY := dset.Y

	// shuffled_idx := make([]int, len(recordsX))
	// tensor.ShuffleData(rng, shuffled_idx)
	_shuffle := func(i, j int) {
		recordsX[i], recordsX[j] = recordsX[j], recordsX[i]
		recordsY[i], recordsY[j] = recordsY[j], recordsY[i]
	}
	_rand := createRand(rng.Seed)
	_rand.Shuffle(len(recordsX), _shuffle)
	return dset
}

func (dset *DataSet[T]) Split(test_ratio float32) ([]T, []T, []T, []T) {
	if test_ratio > 1 || test_ratio < 0 {
		panic("test_ratio must be < 1 and > 0")
	}
	if len(dset.X) != len(dset.Y) {
		panic("Y and X slices have different length")
	}
	features := len(dset.X[0]) // 4
	classes := len(dset.Y[0])  // 1

	test_records_len := int(float32(len(dset.X)) * test_ratio)
	train_records_len := len(dset.X) - test_records_len
	X_train := make([]T, train_records_len*features) // 120*4
	Y_train := make([]T, train_records_len*classes)  // 120*1
	X_test := make([]T, test_records_len*features)   // 30*4
	Y_test := make([]T, test_records_len*classes)    //30*1

	for j := 0; j < len(dset.X); j++ { // 150
		x := dset.X[j]
		y := dset.Y[j]
		if j < train_records_len {
			copy(X_train[j*features:(j+1)*features], x)
			copy(Y_train[j*classes:(j+1)*classes], y)
			continue
		}
		k := j - train_records_len
		copy(X_test[k*features:(k+1)*features], x)
		copy(Y_test[k*classes:(k+1)*classes], y)
	}
	return X_train, Y_train, X_test, Y_test
}

// func Flatten() {

// }
