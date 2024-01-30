//go:build !noasm

package device

import (
	"fmt"
	"gograd/tensor/internal/intrinsics/amd64"
	"gograd/tensor/internal/noasm"
	"gograd/tensor/types"

	"github.com/klauspost/cpuid/v2"
)

// auto-detection of various cpu instructions
// is nothing is supported falls back to pure go implementation

type Implementation struct {
	impl         int
	all_suppored []string
}

const (
	Default int = iota
	AVX
	AVX512
	CUDA // To be implemented
)

// finds possible accelerations instructions
func DetectImpl() Implementation {
	var impl Implementation
	if cpuid.CPU.Supports(cpuid.AVX512F, cpuid.AVX512DQ) {
		impl.all_suppored = append(impl.all_suppored, "AVX512")
	}
	if cpuid.CPU.Supports(cpuid.AVX) {
		impl.all_suppored = append(impl.all_suppored, "AVX")
	}
	// select the best cpu impl
	if cpuid.CPU.Supports(cpuid.AVX512F, cpuid.AVX512DQ) {
		impl.impl = AVX512
	} else if cpuid.CPU.Supports(cpuid.AVX) {
		impl.impl = AVX
	}
	return impl
}

func (i Implementation) ShowDebugInfo() Implementation {
	if len(i.all_suppored) > 0 {
		fmt.Println("CPU acceleration:", i.all_suppored, "available.")
	}
	return i
}

func IsImplAvailable(i Implementation) bool {
	if i.impl == AVX512 && !cpuid.CPU.Supports(cpuid.AVX512F, cpuid.AVX512DQ) {
		return false
		// panic(fmt.Sprintf("Implementation %v is not supported", impl.String()))
	} else if i.impl == AVX && !cpuid.CPU.Supports(cpuid.AVX) {
		return false
	} else if i.impl != Default {
		return false
	}
	return true
}

// binary
func Dot(i Implementation, a, b []float32) float32 {
	switch i.impl {
	case AVX:
		var c float32
		amd64.Dot_mm256(a, b, &c)
		return c
	case AVX512:
		var c float32
		amd64.Dot_mm512(a, b, &c)
		return c
	default:
		var c float32
		return noasm.Dot(a, b, c)
	}
}

func Mul[T types.TensorType](i Implementation, a, b, c []T) {
	afl, bfl, cfl := types.Input_to_float32(a, b, c)
	if i.impl == AVX && afl != nil {
		amd64.Mul_mm256(afl, bfl, cfl)
	} else if i.impl == AVX512 && afl != nil {
		amd64.Mul_mm512(afl, bfl, cfl)
	} else {
		noasm.MulMatx(a, b, c)
	}
}

func MulToConst[T types.TensorType](i Implementation, a []T, b []T, c []T) {
	afl, bfl, cfl := types.Input_b_scalar_to_float32(a, b[0], c)
	if i.impl == AVX && afl != nil {
		amd64.Mul_to_const_mm256(afl, bfl, cfl)
	} else if i.impl == AVX512 && afl != nil {
		amd64.Mul_to_const_mm256(afl, bfl, cfl)
	} else {
		noasm.MulMatxToConst(a, b[0], c)
	}
}

func Div[T types.TensorType](i Implementation, a, b, c []T) {
	noasm.DivMatx(a, b, c)
}

func Add[T types.TensorType](i Implementation, a, b, c []T) {
	afl, _, _ := types.Input_to_float32(a, b, c)
	if i.impl == AVX && afl != nil {
		noasm.AddMatx(a, b, c, amd64.Add_mm256)
	} else if i.impl == AVX512 && afl != nil {
		noasm.AddMatx(a, b, c, amd64.Add_mm256)
	} else {
		noasm.AddMatx(a, b, c, nil)
	}
}

func Sub[T types.TensorType](i Implementation, a, b, c []T) {
	noasm.SubMatx(a, b, c)
}

func Pow[T types.TensorType](i Implementation, a, b, c []T) {
	noasm.PowMatx(a, b, c)
}

// unary
func Sigmoid[T types.TensorType](i Implementation, a, c []T) {
	noasm.SigmoidMatx(a, c)
}

func Neg[T types.TensorType](i Implementation, a, c []T) {
	noasm.NegMatx(a, c)
}

// reduce
func Sum[T types.TensorType](i Implementation, a, c []T) {
	// afl, cfl := reduce_input_to_float32(a, c)
	// if i.impl == AVX && afl != nil {
	// 	amd64.Sum_mm256(afl, cfl)
	// } else if i.impl == AVX512 && afl != nil {
	// 	amd64.Sum_mm256(afl, cfl)
	// } else {
	noasm.SumMatx(a, c)
	// }
}
