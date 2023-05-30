//go:build !noasm

package cpu

import (
	"gograd/tensor/intrinsics/amd64"
	"gograd/tensor/intrinsics/noasm"

	"github.com/klauspost/cpuid/v2"
)

// auto-detection of various cpu instructions
// is nothing is supported falls back to pure go implementation

type Implementation int

const (
	Default Implementation = iota
	AVX
	AVX512
)

// finds possible accelerations instructions
func DetectImpl() Implementation {
	var impl Implementation = 0
	if cpuid.CPU.Supports(cpuid.AVX512F, cpuid.AVX512DQ) {
		impl = AVX512
	} else if cpuid.CPU.Supports(cpuid.AVX) {
		impl = AVX
	}
	return impl
}

func IsImplAvailable(impl Implementation) bool {
	if impl == AVX512 && !cpuid.CPU.Supports(cpuid.AVX512F, cpuid.AVX512DQ) {
		return false
		// panic(fmt.Sprintf("Implementation %v is not supported", impl.String()))
	} else if impl == AVX && !cpuid.CPU.Supports(cpuid.AVX) {
		return false
	} else if impl != Default {
		return false
	}
	return true
}

func (i Implementation) String() string {
	switch i {
	case AVX:
		return "avx"
	case AVX512:
		return "avx512"
	default:
		return "default"
	}
}

func (i Implementation) Dot(a, b []float32) float32 {
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
