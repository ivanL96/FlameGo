package main

import (
	"gograd/tensor"
	"testing"
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
