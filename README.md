A simple framework for linear algebra and machine learning, inspired by numpy/pytorch.

Supports:
1. Tensor operations such as Add, Mul, Dot, etc
   ```
   a := tensor.CreateTensor[int32]([]int32{1,2,3}, types.Shape{3,1})
   b := tensor.CreateTensor[int32]([]int32{3,4,5}, types.Shape{1,3})
   c := a.Mul(b.T()) // transposes b and applies elementwise multiplication

   // Matrix multiplication
   a := tensor.CreateTensor[int32]([]int32{1,2,3,4,5,6}, types.Shape{3,2})
   b := tensor.CreateTensor[int32]([]int32{1,2,3,4,5,6}, types.Shape{2,3})
   c := a.MatMul(b) // result shape will be (3,3)
   ```
2. Reshaping
   ```
   a := tensor.Range[int32](10) // creates a vec from 0 to 9
   a = a.Reshape(5,2,1)
   ```
3. Numpy-like indexing
   ```
   a := tensor.Range[int32](8).Reshape(2,2,2)
   a = a.Index(0,1) // normal indexing
   a.IndexAdv(":,:,1") // indexing along axes
   ```
4. Broadcasting
   ```
   a := tensor.Range[int32](8).Reshape(2,2,2)
   b := tensor.Scalar[int32](1) // scalar with value 1
   c := a.Add(b) // will work because b is auto-broadcasted
   ```
5. Auto differentiation. Grad sub-module implements reverse-mode auto grad logic.
    ```
    a := grad.Variable[float32](tensor.Scalar[float32](4))
  	b := grad.Variable[float32](tensor.Scalar[float32](5))
  	z := a.Mul(b)
  	z.Backward(nil) // Two gradients will be calculated for a & b vars
    fmt.Println(a.Grad.ToString()) // dz/da 5
    fmt.Println(b.Grad.ToString()) // dz/db 4
   ```
