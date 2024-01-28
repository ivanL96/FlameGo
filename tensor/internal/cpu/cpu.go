//go:build !noasm

package cpu

import (
	"gograd/tensor/internal/intrinsics/amd64"
	"gograd/tensor/internal/noasm"
	"gograd/tensor/types"

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
		var c float32
		amd64.Dot_mm256(a, b, &c)
		return c
	case AVX512: // temporary fallback to avx256
		var c float32
		amd64.Dot_mm256(a, b, &c)
		return c
	default:
		var c float32
		return noasm.Dot(a, b, c)
	}
}

func input_b_scalar_to_float32[T types.TensorType](a []T, b T, out []T) ([]float32, float32, []float32) {
	afl, ok_a := any(a).([]float32)
	bfl, ok_b := any(b).(float32)
	cfl, ok_c := any(out).([]float32)
	if !ok_a || !ok_b || !ok_c {
		return nil, 0, nil
	}
	return afl, bfl, cfl
}

func input_to_float32[T types.TensorType](a, b, out []T) ([]float32, []float32, []float32) {
	afl, ok_a := any(a).([]float32)
	bfl, ok_b := any(b).([]float32)
	cfl, ok_c := any(out).([]float32)
	if !ok_a || !ok_b || !ok_c {
		return nil, nil, nil
	}
	return afl, bfl, cfl
}

func Mul[T types.TensorType](i Implementation, a, b, c []T) {
	afl, bfl, cfl := input_to_float32(a, b, c)
	if i == AVX && afl != nil {
		amd64.Mul_mm256(afl, bfl, cfl)
	} else if i == AVX512 && afl != nil { // temporary fallback to avx256
		amd64.Mul_mm256(afl, bfl, cfl)
	} else {
		noasm.MulMatx(a, b, c)
	}
}

func MulToConst[T types.TensorType](i Implementation, a []T, b []T, c []T) {
	afl, bfl, cfl := input_b_scalar_to_float32(a, b[0], c)
	if i == AVX && afl != nil {
		amd64.Mul_to_const_mm256(afl, bfl, cfl)
	} else {
		noasm.MulMatxToConst(a, b[0], c)
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
