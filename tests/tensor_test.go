package main

import (
	"gograd/tensor"
	types "gograd/tensor/types"
	"reflect"
	"testing"
)

func TestAsType(t *testing.T) {
	a := tensor.CreateTensor([]int32{1, 2, 3}, types.Shape{3, 1})
	b := tensor.AsType[int32, int32](a)
	assert(t, a.DType() == b.DType())

	c := tensor.AsType[int32, float32](a)
	assertStatement(t, a.DType().Kind(), NotEquals, c.DType().Kind())
	assertStatement(t, a.DType().Kind(), Equals, reflect.Int32)
	assertStatement(t, c.DType().Kind(), Equals, reflect.Float32)
}

func TestCopy(t *testing.T) {
	a := tensor.CreateTensor([]int32{1, 2, 3}, types.Shape{3, 1})
	b := a.Copy().SetData([]int32{7, 8, 9})
	a_data := a.Data()
	b_data := b.Data()
	assertNotEqualSlices(t, a_data, b_data)
	assert(t, &a != &b)
	assert(t, &(a_data) != &(b_data))
}

func TestCompare(t *testing.T) {
	a := tensor.CreateTensor([]int32{1, 2, 3}, types.Shape{3, 1})
	b := tensor.CreateTensor([]int32{1, 2, 3}, types.Shape{1, 3})
	c := tensor.CreateTensor([]int32{1, 2, 3, 4}, types.Shape{1, 4})
	d := tensor.CreateTensor([]int32{1, 2, 3}, types.Shape{1, 3})
	ab, err := a.IsEqual(b)
	assert(t, !ab)
	assert(t, err == nil)
	ac, err2 := a.IsEqual(c)
	assert(t, !ac)
	assert(t, err2 == nil)
	bd, err3 := b.IsEqual(d)
	assert(t, bd)
	assert(t, err3 == nil)

	// IsEqual is dim order aware
	a1 := tensor.Range[int32](4).Reshape(2, 2).T()
	a2 := a1.AsContiguous()
	a1a2, err4 := a1.IsEqual(a2)
	assert(t, a1a2)
	assert(t, err4 == nil)
}

func TestRange(t *testing.T) {
	var a *tensor.Tensor[int32] = nil
	var b *tensor.Tensor[int32] = nil

	a = tensor.Range[int32](6).MustAssert()
	b = tensor.CreateTensor([]int32{0, 1, 2, 3, 4, 5}, types.Shape{6})
	assertEqualSlices(t, a.Data(), b.Data())
	assertEqualSlices(t, a.Shape(), b.Shape())

	a = tensor.Range[int32](2, 6, 1).MustAssert()
	b = tensor.CreateTensor([]int32{2, 3, 4, 5}, types.Shape{4})
	assertEqualSlices(t, a.Data(), b.Data())
	assertEqualSlices(t, a.Shape(), b.Shape())

	a = tensor.Range[int32](1, 6, 2).MustAssert()
	b = tensor.CreateTensor([]int32{1, 3, 5}, types.Shape{3})
	assertEqualSlices(t, a.Data(), b.Data())
	assertEqualSlices(t, a.Shape(), b.Shape())
}

// test -timeout 30s -run ^TestFill$ gograd/tests
func TestFill(t *testing.T) {
	a := tensor.CreateEmptyTensor[int32](12)
	b := a.Copy()
	a.Fill(2).MustAssert()
	// fmt.Println(b.ToString(), a.ToString())
	assertEqualSlices(t, a.Shape(), b.Shape())
	assertNotEqualSlices(t, a.Data(), b.Data())
}

func TestEye(t *testing.T) {
	a := tensor.Eye[float32](5, 4)
	a.MustAssert()
	for i := 0; i < 4; i++ {
		a1, err := a.Get(i, i)
		assert(t, a1 == 1)
		assert(t, err == nil)
	}
	// fmt.Println(a.ToString())
}

func TestSerializer(t *testing.T) {
	rng := tensor.NewRNG(-1)
	a := rng.RandomFloat32(2, 3)
	enc := a.EncodeToBytes()
	decoded := tensor.DecodeBytes[float32](enc)
	isequal, err := a.IsEqual(decoded)
	if err != nil {
		panic(err)
	}
	assert(t, isequal)
}
