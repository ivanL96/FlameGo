package src

/*
#cgo CFLAGS: -Wall -mavx2 -O3 -fopenmp -fPIC
#cgo LDFLAGS: -lm -fopenmp
#include "avx_float.h"
#include "avx_int.h"
*/
import "C"
import (
	"gograd/tensor/types"
	"reflect"
	"unsafe"
)

func identical[T types.TensorType](s1, s2 []T) bool {
	if len(s1) != len(s2) {
		return false
	}
	return len(s1) == 0 || &s1[0] == &s2[0]
}

func Add_mm256[T types.TensorType](a, b, c []T) {
	if len(a) == 0 || len(b) == 0 || len(c) == 0 {
		return
	}

	t := reflect.TypeOf(a[0]).Kind()
	switch t {
	case reflect.Int, reflect.Int32:
		if len(b) == 1 {
			C._mm256i_add_to_const((*C.int)(unsafe.Pointer(&a[0])), C.int(int32(b[0])), (*C.int)(unsafe.Pointer(&c[0])), C.longlong(len(a)))
		} else if len(a) == 1 {
			C._mm256i_add_to_const((*C.int)(unsafe.Pointer(&b[0])), C.int(int32(a[0])), (*C.int)(unsafe.Pointer(&c[0])), C.longlong(len(b)))
		} else {
			C._mm256i_add_to((*C.int)(unsafe.Pointer(&a[0])), (*C.int)(unsafe.Pointer(&b[0])), (*C.int)(unsafe.Pointer(&c[0])), C.longlong(len(a)))
		}
	case reflect.Float32:
		if len(b) == 1 {
			C._mm256_add_to_const((*C.float)(unsafe.Pointer(&a[0])), C.float(float32(b[0])), (*C.float)(unsafe.Pointer(&c[0])), C.longlong(len(a)))
		} else if len(a) == 1 {
			C._mm256_add_to_const((*C.float)(unsafe.Pointer(&b[0])), C.float(float32(a[0])), (*C.float)(unsafe.Pointer(&c[0])), C.longlong(len(b)))
		} else {
			C._mm256_add_to((*C.float)(unsafe.Pointer(&a[0])), (*C.float)(unsafe.Pointer(&b[0])), (*C.float)(unsafe.Pointer(&c[0])), C.longlong(len(a)))
		}
	// case reflect.Float64:
	default:
		not_implemented_err("Add", t)
	}
}

func Mul_mm256[T types.TensorType](a, b, c []T) {
	if len(a) == 0 || len(b) == 0 || len(c) == 0 {
		return
	}

	t := reflect.TypeOf(a[0]).Kind()
	switch t {
	case reflect.Int, reflect.Int32:
		out := (*C.int)(unsafe.Pointer(&c[0]))
		if len(b) == 1 {
			C._mm256i_mul_to_const((*C.int)(unsafe.Pointer(&a[0])), C.int(int32(b[0])), out, C.longlong(len(a)))
		} else if len(a) == 1 {
			C._mm256i_mul_to_const((*C.int)(unsafe.Pointer(&b[0])), C.int(int32(a[0])), out, C.longlong(len(b)))
		} else {
			C._mm256i_mul_to((*C.int)(unsafe.Pointer(&a[0])), (*C.int)(unsafe.Pointer(&b[0])), out, C.longlong(len(a)))
		}
	case reflect.Float32:
		out := (*C.float)(unsafe.Pointer(&c[0]))
		if len(b) == 1 {
			C._mm256_mul_to_const((*C.float)(unsafe.Pointer(&a[0])), C.float(float32(b[0])), out, C.longlong(len(a)))
		} else if len(a) == 1 {
			C._mm256_mul_to_const((*C.float)(unsafe.Pointer(&b[0])), C.float(float32(a[0])), out, C.longlong(len(b)))
		} else {
			C._mm256_mul_to((*C.float)(unsafe.Pointer(&a[0])), (*C.float)(unsafe.Pointer(&b[0])), out, C.longlong(len(a)))
		}
	// case reflect.Float64:
	default:
		not_implemented_err("Mul", t)
	}
}

func Sub_mm256[T types.TensorType](a, b, c []T) {
	if len(a) == 0 || len(b) == 0 || len(c) == 0 {
		return
	}

	t := reflect.TypeOf(a[0]).Kind()
	switch t {
	case reflect.Int, reflect.Int32:
		out := (*C.int)(unsafe.Pointer(&c[0]))
		if len(b) == 1 {
			C._mm256i_sub_to_const_b((*C.int)(unsafe.Pointer(&a[0])), C.int(int32(b[0])), out, C.longlong(len(a)))
		} else if len(a) == 1 {
			C._mm256i_sub_to_const_a(C.int(int32(a[0])), (*C.int)(unsafe.Pointer(&b[0])), out, C.longlong(len(b)))
		} else {
			C._mm256i_sub_to((*C.int)(unsafe.Pointer(&a[0])), (*C.int)(unsafe.Pointer(&b[0])), out, C.longlong(len(a)))
		}
	case reflect.Float32:
		out := (*C.float)(unsafe.Pointer(&c[0]))
		if len(b) == 1 {
			C._mm256_sub_to_const_b((*C.float)(unsafe.Pointer(&a[0])), C.float(float32(b[0])), out, C.longlong(len(a)))
		} else if len(a) == 1 {
			C._mm256_sub_to_const_a(C.float(float32(a[0])), (*C.float)(unsafe.Pointer(&b[0])), out, C.longlong(len(b)))
		} else {
			C._mm256_sub_to((*C.float)(unsafe.Pointer(&a[0])), (*C.float)(unsafe.Pointer(&b[0])), out, C.longlong(len(a)))
		}
	default:
		not_implemented_err("Sub", t)
	}
}

func Div_mm256[T types.TensorType](a, b, c []T) {
	if len(a) == 0 || len(b) == 0 || len(c) == 0 {
		return
	}

	t := reflect.TypeOf(a[0]).Kind()
	switch t {
	case reflect.Float32:
		af, bf, cf := any(a).([]float32), any(b).([]float32), any(c).([]float32)
		if len(b) == 1 {
			C._mm256_div_to_const_b((*C.float)(&af[0]), C.float(bf[0]), (*C.float)(&cf[0]), C.longlong(len(a)))
		} else if len(a) == 1 {
			C._mm256_div_to_const_a(C.float(af[0]), (*C.float)(&bf[0]), (*C.float)(&cf[0]), C.longlong(len(b)))
		} else {
			C._mm256_div_to((*C.float)(&af[0]), (*C.float)(&bf[0]), (*C.float)(&cf[0]), C.longlong(len(a)))
		}
	default:
		not_implemented_err("Div", t)
	}
}

func Relu_mm256[T types.TensorType](a, c []T) {
	if len(a) == 0 || len(c) == 0 {
		return
	}

	t := reflect.TypeOf(a[0]).Kind()
	switch t {
	case reflect.Float32:
		C._mm256_relu((*C.float)(unsafe.Pointer(&a[0])), (*C.float)(unsafe.Pointer(&c[0])), C.longlong(len(a)))
	default:
		not_implemented_err("Relu", t)
	}
}

func GradientStep_mm256[T types.TensorType](a, b []T, c T) {
	if len(a) == 0 || len(b) == 0 {
		return
	}

	t := reflect.TypeOf(a[0]).Kind()
	switch t {
	case reflect.Float32:
		C._mm256_gradientstep(
			(*C.float)(unsafe.Pointer(&a[0])), (*C.float)(unsafe.Pointer(&b[0])), C.float(float32(c)), C.longlong(len(a)))
	default:
		not_implemented_err("GradientStep", t)
	}
}

// func Pow_mm256[T types.TensorType](a, b, c []T) {
// 	if len(a) == 0 || len(b) == 0 || len(c) == 0 {
// 		return
// 	}

// 	t := reflect.TypeOf(a[0]).Kind()
// 	switch t {
// 	case reflect.Float32:
// 		af, bf, cf := any(a).([]float32), any(b).([]float32), any(c).([]float32)
// 		n := C.longlong(int32(len(a)))
// 		if len(b) == 1 {
// 			C._mm256_pow_to_const_b((*C.float)(&af[0]), C.float(bf[0]), (*C.float)(&cf[0]), n)
// 		} else if len(a) == 1 {
// 			C._mm256_pow_to_const_a(C.float(af[0]), (*C.float)(&bf[0]), (*C.float)(&cf[0]), n)
// 		} else {
// 			C._mm256_pow_to((*C.float)(&af[0]), (*C.float)(&bf[0]), (*C.float)(&cf[0]), n)
// 		}
// 	default:
// 		panic("not implemented")
// 	}
// }
