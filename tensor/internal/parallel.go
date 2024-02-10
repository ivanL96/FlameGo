package internal

import (
	"gograd/tensor/types"
	"runtime"
	"sync"
)

var numCPU int = runtime.NumCPU()

func parallel[T types.TensorType](
	f func(int, int, []T, []T, []T, *sync.Mutex),
	a, b, out []T,
) {
	la := len(a)
	if len(a) == 1 {
		la = len(b)
	}
	chunk_size := (la + numCPU - 1) / numCPU

	var wg sync.WaitGroup
	var mu sync.Mutex
	wg.Add(numCPU)

	for i := 0; i < numCPU; i++ {
		start := i * chunk_size
		end := (i + 1) * chunk_size
		if end > la {
			end = la
		}

		go func(start, end int) {
			defer wg.Done()
			// runtime.LockOSThread()
			// defer runtime.UnlockOSThread()
			f(start, end, a, b, out, &mu)
		}(start, end)
	}
	wg.Wait()
}
