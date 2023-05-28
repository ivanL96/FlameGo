package tensor

import (
	"fmt"
	types "gograd/tensor/types"
	"reflect"
	"strconv"
	"strings"
)

type printSettings struct {
	whitespaceOffset int
	lastCharClosed   bool
}

func joinData[T types.TensorType](sb *strings.Builder, data []T) {
	stringData := make([]string, len(data))
	dtype := getTypeArray(data)
	switch dtype.Kind() {
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64,
		reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		for i, val := range data {
			stringData[i] = string(strconv.FormatInt(int64(val), 10))
		}
	case reflect.Float32:
		for i, val := range data {
			stringData[i] = strconv.FormatFloat(float64(val), 'f', 4, 32)
		}
	case reflect.Float64:
		for i, val := range data {
			stringData[i] = strconv.FormatFloat(float64(val), 'f', 4, 64)
		}
	default:
		panic("Got unknown type for stringify data")
	}
	sb.WriteString(strings.Join(stringData, ", "))
}

func stringRepr[T types.TensorType](sb *strings.Builder, tensor *Tensor[T], ps *printSettings) {
	// TODO shrink height of printed array
	if ps.lastCharClosed {
		for j := 0; j < ps.whitespaceOffset; j++ {
			sb.WriteRune(' ')
		}
		ps.lastCharClosed = false
	}
	sb.WriteRune('[')
	if len(tensor.shape) == 1 { // at this point tensor is subset created from mainTensor.Index()
		lastDim := tensor.shape[0]
		if lastDim > 30 {
			joinData(sb, tensor.data[:5])
			sb.WriteString("...")
			joinData(sb, tensor.data[len(tensor.data)-5:])
		} else {
			joinData(sb, tensor.data[:lastDim])
		}
		sb.WriteRune(']')
		ps.whitespaceOffset -= 1
		ps.lastCharClosed = true
		return
	}

	nrows := int(tensor.shape[0])
	for i := 0; i < nrows; i++ {
		if i > 0 {
			sb.WriteRune('\n')
		}
		ps.whitespaceOffset += 1
		stringRepr(sb, tensor.Index(i), ps)
		if i == nrows-1 {
			sb.WriteRune(']')
			ps.whitespaceOffset -= 1
			ps.lastCharClosed = true
		}
	}
}

func (tensor *Tensor[T]) ToString() string {
	var shape_prod types.Dim = 1
	for _, dim := range tensor.shape {
		shape_prod *= dim
	}
	var (
		sb strings.Builder
		ps printSettings
	)
	stringRepr(&sb, tensor, &ps)
	strData := sb.String()

	if len(tensor.shape) > 1 {
		strData = "\n" + strData
	}
	return fmt.Sprintf(
		"Tensor(%v, shape=%v, dtype=%v, order=%v, strides=%v)",
		strData,
		tensor.shape,
		getTypeArray(tensor.data).Kind().String(),
		tensor.dim_order,
		tensor.strides,
	)
}
