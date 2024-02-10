package src

/*
#cgo CFLAGS: -mavx -O3 -mavx512f -mavx512dq -fopenmp
#cgo LDFLAGS: -lm -fopenmp
#include "avx2_float.h"
#include "avx2_int.h"
*/
import "C"
import (
	"fmt"
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

func not_implemented_err(msg string, t reflect.Kind) {
	panic(fmt.Sprintf("%v is not implemented for %v", msg, t))
}

func Add_mm256[T types.TensorType](a, b, c []T) {
	if len(a) == 0 || len(b) == 0 || len(c) == 0 {
		return
	}
	n := C.longlong(len(a))

	t := reflect.TypeOf(a[0]).Kind()
	switch t {
	case reflect.Int, reflect.Int32:
		// ai, bi, ci := any(a).([]int32), any(b).([]int32), any(c).([]int32)
		if len(b) == 1 {
			C._mm256i_add_to_const((*C.int)(unsafe.Pointer(&a[0])), C.int(int32(b[0])), (*C.int)(unsafe.Pointer(&c[0])), n)
		} else if len(a) == 1 {
			C._mm256i_add_to_const((*C.int)(unsafe.Pointer(&b[0])), C.int(int32(a[0])), (*C.int)(unsafe.Pointer(&c[0])), n)
		} else {
			C._mm256i_add_to((*C.int)(unsafe.Pointer(&a[0])), (*C.int)(unsafe.Pointer(&b[0])), (*C.int)(unsafe.Pointer(&c[0])), n)
		}
	case reflect.Float32:
		// af, bf, cf := any(a).([]float32), any(b).([]float32), any(c).([]float32)
		if len(b) == 1 {
			C._mm256_add_to_const((*C.float)(unsafe.Pointer(&a[0])), C.float(float32(b[0])), (*C.float)(unsafe.Pointer(&c[0])), n)
		} else if len(a) == 1 {
			C._mm256_add_to_const((*C.float)(unsafe.Pointer(&b[0])), C.float(float32(a[0])), (*C.float)(unsafe.Pointer(&c[0])), n)
		} else {
			C._mm256_add_to((*C.float)(unsafe.Pointer(&a[0])), (*C.float)(unsafe.Pointer(&b[0])), (*C.float)(unsafe.Pointer(&c[0])), n)
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
	n := C.longlong(len(a))

	t := reflect.TypeOf(a[0]).Kind()
	switch t {
	case reflect.Int, reflect.Int32:
		if len(b) == 1 {
			C._mm256i_mul_to_const((*C.int)(unsafe.Pointer(&a[0])), C.int(int32(b[0])), (*C.int)(unsafe.Pointer(&c[0])), n)
		} else if len(a) == 1 {
			C._mm256i_mul_to_const((*C.int)(unsafe.Pointer(&b[0])), C.int(int32(a[0])), (*C.int)(unsafe.Pointer(&c[0])), n)
		} else {
			C._mm256i_mul_to((*C.int)(unsafe.Pointer(&a[0])), (*C.int)(unsafe.Pointer(&b[0])), (*C.int)(unsafe.Pointer(&c[0])), n)
		}
	case reflect.Float32:
		if len(b) == 1 {
			C._mm256_mul_to_const((*C.float)(unsafe.Pointer(&a[0])), C.float(float32(b[0])), (*C.float)(unsafe.Pointer(&c[0])), n)
		} else if len(a) == 1 {
			C._mm256_mul_to_const((*C.float)(unsafe.Pointer(&b[0])), C.float(float32(a[0])), (*C.float)(unsafe.Pointer(&c[0])), n)
		} else {
			C._mm256_mul_to((*C.float)(unsafe.Pointer(&a[0])), (*C.float)(unsafe.Pointer(&b[0])), (*C.float)(unsafe.Pointer(&c[0])), n)
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
	case reflect.Float32:
		af, bf, cf := any(a).([]float32), any(b).([]float32), any(c).([]float32)
		n := C.longlong(len(a))
		if len(b) == 1 {
			C._mm256_sub_to_const((*C.float)(&af[0]), C.float(bf[0]), (*C.float)(&cf[0]), n)
		} else if len(a) == 1 {
			C._mm256_sub_to_const((*C.float)(&bf[0]), C.float(af[0]), (*C.float)(&cf[0]), n)
		} else {
			C._mm256_sub_to((*C.float)(&af[0]), (*C.float)(&bf[0]), (*C.float)(&cf[0]), n)
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
		n := C.longlong(len(a))
		if len(b) == 1 {
			C._mm256_div_to_const_b((*C.float)(&af[0]), C.float(bf[0]), (*C.float)(&cf[0]), n)
		} else if len(a) == 1 {
			C._mm256_div_to_const_a(C.float(af[0]), (*C.float)(&bf[0]), (*C.float)(&cf[0]), n)
		} else {
			C._mm256_div_to((*C.float)(&af[0]), (*C.float)(&bf[0]), (*C.float)(&cf[0]), n)
		}
	default:
		not_implemented_err("Div", t)
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

func Neg_mm256[T types.TensorType](a, c []T) {
	if len(a) == 0 || len(c) == 0 {
		return
	}

	t := reflect.TypeOf(a[0]).Kind()
	switch t {
	case reflect.Float32:
		af, cf := any(a).([]float32), any(c).([]float32)
		n := C.longlong(len(a))
		C._mm256_neg_to((*C.float)(&af[0]), (*C.float)(&cf[0]), n)
	default:
		panic("not implemented")
	}
}
