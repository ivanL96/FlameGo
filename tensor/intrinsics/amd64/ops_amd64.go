package amd64

import "unsafe"

// func Add(a, b int64) int64 {
// 	return add(a, b)
// }

func Dot_mm256(a, b []float32) float32 {
	var ret float32
	_mm256_dot(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(uintptr(len(a))), unsafe.Pointer(&ret))
	return ret
}
