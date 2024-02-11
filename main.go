package main

import (
	_ "net/http/pprof"
)

func main() {
	// var wg sync.WaitGroup
	// go func() {
	// 	fmt.Println(http.ListenAndServe("localhost:6060", nil))
	// }()
	// wg.Add(1)
	pprof_main(classifier_iris)
	// pprof_main(classifier_gen)
	// wg.Wait()
	// classifier_gen()
}

// go build -gcflags="-m -l"
