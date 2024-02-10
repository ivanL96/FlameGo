package device

import (
	"fmt"
	"gograd/tensor/internal"
	"gograd/tensor/internal/intrinsics/amd64"
	"gograd/tensor/internal/intrinsics/src"
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
	if cpuid.CPU.AnyOf(cpuid.AMXBF16, cpuid.AMXBF16, cpuid.AMXINT8, cpuid.AMXTILE) {
		impl.all_suppored = append(impl.all_suppored, "AMX")
	}
	// select the best cpu impl
	if cpuid.CPU.Supports(cpuid.AVX512F, cpuid.AVX512DQ) {
		impl.impl = AVX512
	} else if cpuid.CPU.Supports(cpuid.AVX) {
		impl.impl = AVX
	}
	return &impl
}

func (i *Implementation) SetImpl(newimpl int) {
	i.impl = newimpl
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
		internal.MatMulMatx(af, bf, outf, a_shape, b_shape, a_strides, b_strides, out_strides, amd64.Dot_mm256)
	case AVX512:
		internal.MatMulMatx(af, bf, outf, a_shape, b_shape, a_strides, b_strides, out_strides, amd64.Dot_mm512)
	default:
		internal.MatMulMatx(af, bf, outf, a_shape, b_shape, a_strides, b_strides, out_strides, internal.Dot[T])
	}
}

func Mul[T types.TensorType](i Implementation, a, b, c []T) {
	switch i.impl {
	case AVX:
		src.Mul_mm256(a, b, c)
	case AVX512:
		src.Mul_mm512(a, b, c)
	default:
		internal.ElementwiseNoSimd(a, b, c, internal.MulAtomic)
	}
}

func Div[T types.TensorType](i Implementation, a, b, c []T) {
	switch i.impl {
	case AVX:
		internal.RunSimdImpl(a, b, c, src.Div_mm256)
	case AVX512:
		internal.RunSimdImpl(a, b, c, src.Div_mm256)
	default:
		internal.ElementwiseNoSimd(a, b, c, internal.DivAtomic)
	}
}

func Add[T types.TensorType](i Implementation, a, b, c []T) {
	switch i.impl {
	case AVX:
		src.Add_mm256(a, b, c)
	case AVX512:
		src.Add_mm256(a, b, c)
	default:
		internal.ElementwiseNoSimd(a, b, c, internal.AddAtomic)
	}
}

func Sub[T types.TensorType](i Implementation, a, b, c []T) {
	switch i.impl {
	case AVX:
		src.Sub_mm256(a, b, c)
	case AVX512:
		src.Sub_mm256(a, b, c)
	default:
		internal.ElementwiseNoSimd(a, b, c, internal.SubAtomic)
	}
}

func Pow[T types.TensorType](i Implementation, a, b, c []T) {
	// internal.ElementwiseNoSimd(a, b, c, internal.PowAtomic)
	internal.PowMatx(a, b, c)
}

// unary
func Sigmoid[T types.TensorType](i Implementation, a, c []T) {
	internal.SigmoidMatx(a, c)
}

func Neg[T types.TensorType](i Implementation, a, c []T) {
	internal.NegMatx(a, c)
}

func LnNeg[T types.TensorType](i Implementation, a, c []T) {
	internal.LnNegMatx(a, c)
}

func Exp[T types.TensorType](i Implementation, a, c []T) {
	internal.ExpMatx(a, c)
}

func Relu[T types.TensorType](i Implementation, a, c []T) {
	internal.ReluMatx(a, c)
}

func Softmax[T types.TensorType](i Implementation, a, c []T, strides []int) {
	internal.SoftmaxMatx(a, c, strides)
}

// masking
func ApplyFunc[T types.TensorType](i Implementation, a []T, expr func(T) T, out []T) {
	internal.ApplyFuncMatx(a, expr, out)
}

// reduce
func Sum[T types.TensorType](i Implementation, a, c []T) {
	// afl, cfl := reduce_input_to_float32(a, c)
	// if i.impl == AVX && afl != nil {
	// 	amd64.Sum_mm256(afl, cfl)
	// } else if i.impl == AVX512 && afl != nil {
	// 	amd64.Sum_mm256(afl, cfl)
	// } else {
	internal.SumMatx(a, c)
	// }
}
func SumAxis[T types.TensorType](i Implementation, a, c []T, shape types.Shape, axis int) {
	internal.SumAxisMatx(a, c, shape, axis)
}

func Max[T types.TensorType](i Implementation, a, c []T) {
	internal.MaxMatx(a, c)
}

func Min[T types.TensorType](i Implementation, a, c []T) {
	internal.MinMatx(a, c)
}
