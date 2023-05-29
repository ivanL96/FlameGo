//go:build !noasm

package cpu

import (
	"gograd/tensor/intrinsics/amd64"
	"gograd/tensor/intrinsics/noasm"
)

// var impl = Default

// func init() {
// 	if cpuid.CPU.Supports(cpuid.AVX512F, cpuid.AVX512DQ) {
// 		impl = AVX512
// 	} else if cpuid.CPU.Supports(cpuid.AVX) {
// 		impl = AVX
// 	}
// }

type implementation int

const (
	Default implementation = iota
	AVX
	// AVX512
)

func (i implementation) String() string {
	switch i {
	case AVX:
		return "avx"
	// case AVX512:
	// 	return "avx512"
	default:
		return "default"
	}
}

func (i implementation) Dot(a, b []float32) float32 {
	switch i {
	case AVX:
		return amd64.Dot_mm256(a, b)
	// case AVX512:
	// 	var ret float32
	// 	_mm512_dot(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(uintptr(len(a))), unsafe.Pointer(&ret))
	// 	return ret
	default:
		var c float32
		return noasm.Dot(a, b, c)
	}
}
