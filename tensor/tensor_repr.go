package tensor

import (
	types "flamego/tensor/types"
	"fmt"
	"reflect"
	"strconv"
	"strings"
)

const _LINES_LIMIT int = 100
const _HEAD_SIZE int = 10

type printSettings struct {
	whitespaceOffset int
	linesAdded       int
	linesSkipped     int
	total_lines      int
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

func printIndent(sb *strings.Builder, ps *printSettings) {
	if ps.lastCharClosed && ps.linesSkipped == 0 {
		for j := 0; j < ps.whitespaceOffset; j++ {
			sb.WriteRune(' ')
		}
		ps.lastCharClosed = false
	}
}

func stringRepr[T types.TensorType](sb *strings.Builder, tensor *Tensor[T], ps *printSettings) {
	printIndent(sb, ps)

	// write line
	if len(tensor.shape) == 1 { // at this point tensor is subset created from mainTensor.Index()
		if ps.total_lines >= _LINES_LIMIT {
			// if tensor is too big then should shrink lines after some limit (_HEAD_SIZE)
			if ps.linesAdded >= _HEAD_SIZE {
				// shrink height
				if ps.linesSkipped == 0 {
					sb.WriteString("...\n")
					ps.linesSkipped++
					return
				} else if ps.linesSkipped < ps.total_lines-_HEAD_SIZE*2 {
					ps.linesSkipped++
					return
				} else {
					// stop skipping lines, reset counters
					ps.linesAdded = 0
					ps.linesSkipped = 0
					ps.lastCharClosed = true
					printIndent(sb, ps)
				}
			}
			ps.linesAdded++
		}

		sb.WriteRune('[')
		lastDim := tensor.shape[0]
		if lastDim > 30 {
			// shrink row
			joinData(sb, tensor.data()[:5])
			sb.WriteString("...")
			joinData(sb, tensor.data()[len(tensor.data())-5:])
		} else {
			joinData(sb, tensor.data()[:lastDim])
		}
		sb.WriteRune(']')
		ps.whitespaceOffset -= 1
		ps.lastCharClosed = true
		return
	}

	// write sub tensor
	if ps.linesSkipped == 0 {
		sb.WriteRune('[')
	}
	nrows := int(tensor.shape[0])
	for i := 0; i < nrows; i++ {
		if ps.linesSkipped == 0 {
			if i > 0 {
				sb.WriteRune('\n')
			}
			ps.whitespaceOffset += 1
		}
		stringRepr(sb, tensor.Index(i), ps)
		if i == nrows-1 && ps.linesSkipped == 0 {
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
	total_lines := int(shape_prod / tensor.shape[len(tensor.shape)-1])
	var sb strings.Builder
	ps := printSettings{
		total_lines: total_lines,
	}
	stringRepr(&sb, tensor, &ps)
	strData := sb.String()

	if len(tensor.shape) > 1 {
		strData = "\n" + strData
	}
	return fmt.Sprintf(
		"Tensor(%v, shape=%v, dtype=%v, order=%v, strides=%v)",
		strData,
		tensor.shape,
		getTypeArray(tensor.data()).Kind().String(),
		tensor.dim_order,
		tensor.strides,
	)
}
