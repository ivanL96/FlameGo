package main

import (
	"fmt"
	"gograd/tensor"
	"testing"

	"golang.org/x/exp/constraints"
)

func assertEqualSlices[DT tensor.Number](t *testing.T, slice1 []DT, slice2 []DT) {
	if !tensor.Compare_slices(slice1, slice2) {
		t.Errorf("Slices must be equal. Got %v and %v", slice1, slice2)
	}
}

func assertNotEqualSlices[DT tensor.Number](t *testing.T, slice1 []DT, slice2 []DT) {
	if tensor.Compare_slices(slice1, slice2) {
		t.Errorf("Slices must be not equal. Got %v and %v", slice1, slice2)
	}
}

func assert(t *testing.T, stmt bool) {
	if !stmt {
		t.Errorf("Statement must be true.")
	}
}

const (
	Equals    int = 0
	NotEquals     = 1
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
