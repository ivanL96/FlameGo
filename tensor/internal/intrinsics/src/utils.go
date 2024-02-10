package src

import (
	"fmt"
	"reflect"
)

func not_implemented_err(msg string, t reflect.Kind) {
	panic(fmt.Sprintf("%v is not implemented for %v", msg, t))
}
