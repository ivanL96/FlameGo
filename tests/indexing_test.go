package main

import (
	"gograd/tensor"
	types "gograd/tensor/types"
	"testing"
)

func TestIndex(t *testing.T) {
	a := tensor.Range[int32](8).Reshape(2, 2, 2)
	sub1 := a.Index(1).MustAssert()
	assertEqualSlices(t, sub1.Data(), []int32{4, 5, 6, 7})
	assertEqualSlices(t, sub1.Shape(), types.Shape{2, 2})

	sub2 := sub1.Index(1).MustAssert()
	assertEqualSlices(t, sub2.Data(), []int32{6, 7})
	assertEqualSlices(t, sub2.Shape(), types.Shape{2})

	b := tensor.CreateTensor([]int32{0, 1, 2, 3}, types.Shape{2, 2})
	bsub := b.Index(-2).MustAssert()
	assertEqualSlices(t, bsub.Data(), []int32{0, 1})
	assertEqualSlices(t, bsub.Shape(), types.Shape{2})

	c := tensor.Range[int32](100).Reshape(10, 10).Fill(7)
	c.Set([]int{0, 0}, 1)
	csub := c.Index(0).MustAssert()
	// fmt.Println(csub.ToString())
	assertEqualSlices(t, csub.Shape(), types.Shape{10})
}

func TestTransposeAndIndex(t *testing.T) {
	a := tensor.Range[int32](8).Reshape(4, 2)
	// fmt.Println(a)
	a = a.T()
	// fmt.Println(a)
	a = a.Index(1).MustAssert()
	// fmt.Println(a, a.ToString())
	assertEqualSlices(t, a.Data(), []int32{1, 3, 5, 7})
	assertEqualSlices(t, a.Shape(), types.Shape{4})

	b := tensor.Range[int32](8).Reshape(4, 1, 1, 2)
	// b.ToString()
	b = b.T().Index(1).MustAssert()
	assertEqualSlices(t, b.Data(), []int32{1, 3, 5, 7})
	assertEqualSlices(t, b.Shape(), types.Shape{1, 1, 4})

	c := tensor.Range[int32](3*4*2).Reshape(3, 4, 2)
	c = c.T().Index(0).MustAssert()
	assertEqualSlices(t,
		c.Data(), []int32{0, 8, 16, 2, 10, 18, 4, 12, 20, 6, 14, 22})
	assertEqualSlices(t, c.Shape(), types.Shape{4, 3})

	d := tensor.Range[int32](8).Reshape(4, 2, 1)
	d = d.T().Index(0).MustAssert()
	// d.ToString()
	assertEqualSlices(t, d.Data(), []int32{0, 2, 4, 6, 1, 3, 5, 7})
	assertEqualSlices(t, d.Shape(), types.Shape{2, 4})

	e := tensor.CreateTensor([]int32{3}, types.Shape{1, 1, 1, 1})
	e = e.T().Index(0).MustAssert()
	assertEqualSlices(t, e.Data(), []int32{3})
	assertEqualSlices(t, e.Shape(), types.Shape{1, 1, 1})
}

func TestTransposeAndIndexFast(t *testing.T) {
	a := tensor.Range[int32](8).Reshape(4, 2).T()
	// fmt.Println(a.ToString())
	out := a.Get_fast(1, 1)
	// a.ToString()
	assert(t, out == 3)
}

func TestAdvIndexing(t *testing.T) {
	a := tensor.Range[int32](2*2*2*2).Reshape(2, 2, 2, 2)
	// arr[:,:,:,0]
	c := a.TrC(3, 0, 1, 2).Index(0).MustAssert()
	assertEqualSlices(t, c.Data(), []int32{0, 2, 4, 6, 8, 10, 12, 14})
	// arr[:,:,0,:]
	c = a.TrC(2, 0, 1, 3).Index(0).MustAssert()
	assertEqualSlices(t, c.Data(), []int32{0, 1, 4, 5, 8, 9, 12, 13})
	// arr[:,:,0,0]
	c = c.TrC(2, 0, 1).Index(0).MustAssert()
	assertEqualSlices(t, c.Data(), []int32{0, 4, 8, 12})
}

func TestAdvIndexing2(t *testing.T) {
	a := tensor.Range[int32](2*2*2*2).Reshape(2, 2, 2, 2)
	c := a.IndexAdv(":,:,:,:").MustAssert()
	assertEqualSlices(t, c.Data(), a.Data())
	c = a.IndexAdv(":,:,:,0").MustAssert()
	assertEqualSlices(t, c.Data(), []int32{0, 2, 4, 6, 8, 10, 12, 14})
	c = a.IndexAdv(":,:,0,:").MustAssert()
	assertEqualSlices(t, c.Data(), []int32{0, 1, 4, 5, 8, 9, 12, 13})
	c = a.IndexAdv(":,:,0").MustAssert()
	assertEqualSlices(t, c.Data(), []int32{0, 1, 4, 5, 8, 9, 12, 13})
	c = a.IndexAdv(":,:,0,0").MustAssert()
	assertEqualSlices(t, c.Data(), []int32{0, 4, 8, 12})
	c = a.IndexAdv("0,1,0,1").MustAssert()
	assertEqualSlices(t, c.Data(), []int32{5})
}

func TestIndexMask(t *testing.T) {
	a := tensor.Range[int32](9).Reshape(3, 3)
	// 0 1 2
	// 3 4 5
	// 6 7 8
	mask := tensor.CreateTensor([]int32{1, 0, 2}, types.Shape{3})
	masked := a.IndexMask(mask, false).MustAssert()
	assertEqualSlices(t, a.Shape(), types.Shape{3, 3})
	assertEqualSlices(t, masked.Data(), []int32{3, 4, 5, 0, 1, 2, 6, 7, 8})
	assertEqualSlices(t, masked.Shape(), types.Shape{3, 3})

	mask1 := tensor.CreateTensor([]int32{
		0, 2,
		1, 0,
		2, 1}, types.Shape{3, 2})
	masked1 := a.IndexMask(mask1, false).MustAssert()
	assertEqualSlices(t, masked1.Data(), []int32{2, 3, 7})
}

func TestIndexMask2(t *testing.T) {
	a := tensor.Range[int32](8).Reshape(2, 2, 2)
	mask := tensor.CreateTensor([]int32{1, 0}, types.Shape{2})
	masked := a.IndexMask(mask, false).MustAssert()
	assertEqualSlices(t, masked.Data(), []int32{4, 5, 6, 7, 0, 1, 2, 3})
	assertEqualSlices(t, masked.Shape(), types.Shape{2, 2, 2})

	b := tensor.Range[int32](8).Reshape(2, 2, 2)
	maskb := tensor.CreateTensor([]int32{1, 0, 1, 0}, types.Shape{2, 2})
	maskedb := b.IndexMask(maskb, false).MustAssert()
	assertEqualSlices(t, maskedb.Data(), []int32{4, 5, 4, 5})
	assertEqualSlices(t, maskedb.Shape(), types.Shape{2, 2})
}

func TestIndexMask3(t *testing.T) {
	a := tensor.Range[int32](8).Reshape(2, 4)
	mask := tensor.CreateTensor([]int32{2, 1}, types.Shape{2})
	a.SetByIndexMask(mask, true, 77)
	assertEqualSlices(t, a.Data(), []int32{0, 1, 77, 3, 4, 77, 6, 7})
}
