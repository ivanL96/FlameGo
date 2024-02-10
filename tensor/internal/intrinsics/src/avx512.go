package src

/*
#cgo CFLAGS: -mavx2 -O3 -fopenmp -mavx512f -mavx512dq
#cgo LDFLAGS: -lm -fopenmp
#include "avx512_float.h"
#include "avx_int.h"
*/
import "C"
import (
	"gograd/tensor/types"
	"reflect"
	"unsafe"
)

func Mul_mm512[T types.TensorType](a, b, c []T) {
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
			C._mm512_mul_to_const((*C.float)(unsafe.Pointer(&a[0])), C.float(float32(b[0])), out, C.longlong(len(a)))
		} else if len(a) == 1 {
			C._mm512_mul_to_const((*C.float)(unsafe.Pointer(&b[0])), C.float(float32(a[0])), out, C.longlong(len(b)))
		} else {
			C._mm512_mul_to((*C.float)(unsafe.Pointer(&a[0])), (*C.float)(unsafe.Pointer(&b[0])), out, C.longlong(len(a)))
		}
	// case reflect.Float64:
	default:
		not_implemented_err("Mul", t)
	}
}
