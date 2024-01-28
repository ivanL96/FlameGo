package main

import (
	"fmt"
	"gograd/tensor"
	types "gograd/tensor/types"
	"testing"

	"golang.org/x/exp/constraints"
)

// go test ./tests -v

func assertEqualSlices[DT types.TensorType](t *testing.T, slice1 []DT, slice2 []DT) {
	if !tensor.EqualSlices(slice1, slice2) {
		t.Errorf("Slices must be equal. Got %v and %v", slice1, slice2)
	}
}

func assertNotEqualSlices[DT types.TensorType](t *testing.T, slice1 []DT, slice2 []DT) {
	if tensor.EqualSlices(slice1, slice2) {
		t.Errorf("Slices must be not equal. Got %v and %v", slice1, slice2)
	}
}

func assert(t *testing.T, stmt bool) {
	if !stmt {
		t.Errorf("Statement must be true.")
	}
}

const (
	Equals    = 0
	NotEquals = 1
)

func assertStatement[T constraints.Ordered](t *testing.T, value1 T, operator int, value2 T) {
	stmt := false
	operator_alias := ""
	switch operator {
	case Equals:
		stmt = value1 == value2
		operator_alias = "=="
	case NotEquals:
		stmt = value1 != value2
		operator_alias = "!="
	default:
		panic(fmt.Sprintf("Unknown operator %v", operator))
	}
	if !stmt {
		t.Errorf(fmt.Sprintf("False statement %v %v %v.", value1, operator_alias, value2))
	}
}
