//go:build !noasm

package cpu

import (
	"flamego/tensor/internal/intrinsics/amd64"
	"flamego/tensor/internal/noasm"
	"flamego/tensor/types"

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

func Dot(i Implementation, a, b []float32) float32 {
	switch i {
	case AVX:
		// afl := types.Any(a).([]float32)
		// bfl := types.Any(a).([]float32)
		ret := amd64.Dot_mm256(a, b)
		// return types.Any(ret).(T)
		return ret
	// case AVX512:
	// 	var ret float32
	// 	_mm512_dot(unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(uintptr(len(a))), unsafe.Pointer(&ret))
	// 	return ret
	default:
		var c float32
		return noasm.Dot(a, b, c)
	}
}

func Mul[T types.TensorType](i Implementation, a, b, c []T) {
	switch i {
	case AVX:
		afl := types.Any(a).([]float32)
		bfl := types.Any(b).([]float32)
		cfl := types.Any(c).([]float32)
		amd64.Mul_mm256(afl, bfl, cfl)
	default:
		noasm.MulMatx(a, b, c)
	}
}

func MulToConst[T types.TensorType](i Implementation, a []T, b T, c []T) {
	switch i {
	case AVX:
		afl := types.Any(a).([]float32)
		bfl := types.Any(b).(float32)
		cfl := types.Any(c).([]float32)
		amd64.Mul_to_const_mm256(afl, bfl, cfl)
	default:
		noasm.MulMatxToConst(a, b, c)
	}
}

func Div[T types.TensorType](i Implementation, a, b, c []T) {
	noasm.DivMatx(a, b, c)
}

func Add[T types.TensorType](i Implementation, a, b, c []T) {
	// switch i {
	// case AVX:
	// 	amd64.Mul_mm256(a, b, c)
	// default:
	noasm.AddMatx(a, b, c)
	// }
}

func Sub[T types.TensorType](i Implementation, a, b, c []T) {
	noasm.SubMatx(a, b, c)
}

func Pow[T types.TensorType](i Implementation, a, b, c []T) {
	noasm.PowMatx(a, b, c)
}
