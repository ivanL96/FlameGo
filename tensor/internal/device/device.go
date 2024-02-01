package device

import (
	"fmt"
	"gograd/tensor/internal/intrinsics/amd64"
	"gograd/tensor/internal/ops"
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
func DetectImpl() *Implementation {
	var impl Implementation
	impl.impl = Default
	if cpuid.CPU.Supports(cpuid.AVX512F, cpuid.AVX512DQ) {
		impl.all_suppored = append(impl.all_suppored, "AVX512")
	}
	if cpuid.CPU.Supports(cpuid.AVX) {
		impl.all_suppored = append(impl.all_suppored, "AVX")
	}
	if cpuid.CPU.Supports(cpuid.AVX512VNNI) {
		impl.all_suppored = append(impl.all_suppored, "AVX512VNNI")
	}
	// select the best cpu impl
	if cpuid.CPU.Supports(cpuid.AVX512F, cpuid.AVX512DQ) {
		impl.impl = AVX512
	} else if cpuid.CPU.Supports(cpuid.AVX) {
		impl.impl = AVX
	}
	return &impl
}

func (i *Implementation) ShowDebugInfo() *Implementation {
	if len(i.all_suppored) > 0 {
		fmt.Println("CPU acceleration:", i.all_suppored, "available.")
	}
	return i
}

func IsImplAvailable(i *Implementation) bool {
	if i.impl == AVX512 && !cpuid.CPU.Supports(cpuid.AVX512F, cpuid.AVX512DQ) {
		return false
	} else if i.impl == AVX && !cpuid.CPU.Supports(cpuid.AVX) {
		return false
	} else if i.impl != Default {
		return false
	}
	return true
}

// binary
func MatMul[T types.TensorType](
	i Implementation,
	a, b, out []T,
	a_shape, b_shape types.Shape,
	a_strides, b_strides, out_strides []int,
) {
	af, bf, outf := types.Input_to_float32(a, b, out)
	switch i.impl {
	case AVX:
		ops.MatMulMatx(af, bf, outf, a_shape, b_shape, a_strides, b_strides, out_strides, amd64.Dot_mm256)
	case AVX512:
		ops.MatMulMatx(af, bf, outf, a_shape, b_shape, a_strides, b_strides, out_strides, amd64.Dot_mm512)
	default:
		ops.MatMulMatx(af, bf, outf, a_shape, b_shape, a_strides, b_strides, out_strides, ops.Dot[T])
	}
}

func Mul[T types.TensorType](i Implementation, a, b, c []T) {
	// afl, _, _ := types.Input_to_float32(a, b, c)
	switch i.impl {
	case AVX:
		ops.MulMatx(a, b, c, amd64.Mul_mm256)
	case AVX512:
		ops.MulMatx(a, b, c, amd64.Mul_mm256)
	default:
		ops.MulMatx(a, b, c, nil)
	}
}

func MulToConst[T types.TensorType](i Implementation, a, b, c []T) {
	// afl, _, _ := types.Input_b_scalar_to_float32(a, b[0], c)
	switch i.impl {
	case AVX:
		ops.MulMatxToConst(a, b, c, amd64.Mul_to_const_mm256)
	case AVX512:
		ops.MulMatxToConst(a, b, c, amd64.Mul_to_const_mm256)
	default:
		ops.MulMatxToConst(a, b, c, nil)
	}
}

func Div[T types.TensorType](i Implementation, a, b, c []T) {
	ops.DivMatx(a, b, c)
}

func Add[T types.TensorType](i Implementation, a, b, c []T) {
	// afl, _, _ := types.Input_to_float32(a, b, c)
	switch i.impl {
	case AVX:
		ops.AddMatx(a, b, c, amd64.Add_mm256)
	case AVX512:
		ops.AddMatx(a, b, c, amd64.Add_mm256)
	default:
		ops.AddMatx(a, b, c, nil)
	}
}

func Sub[T types.TensorType](i Implementation, a, b, c []T) {
	ops.SubMatx(a, b, c)
}

func Pow[T types.TensorType](i Implementation, a, b, c []T) {
	ops.PowMatx(a, b, c)
}

// unary
func Sigmoid[T types.TensorType](i Implementation, a, c []T) {
	ops.SigmoidMatx(a, c)
}

func Neg[T types.TensorType](i Implementation, a, c []T) {
	ops.NegMatx(a, c)
}

func Relu[T types.TensorType](i Implementation, a, c []T) {
	ops.ReluMatx(a, c)
}

// masking
func ApplyFunc[T types.TensorType](i Implementation, a []T, expr func(T) T, out []T) {
	ops.ApplyFuncMatx(a, expr, out)
}

// reduce
func Sum[T types.TensorType](i Implementation, a, c []T) {
	// afl, cfl := reduce_input_to_float32(a, c)
	// if i.impl == AVX && afl != nil {
	// 	amd64.Sum_mm256(afl, cfl)
	// } else if i.impl == AVX512 && afl != nil {
	// 	amd64.Sum_mm256(afl, cfl)
	// } else {
	ops.SumMatx(a, c)
	// }
}

func Max[T types.TensorType](i Implementation, a, c []T) {
	ops.MaxMatx(a, c)
}
