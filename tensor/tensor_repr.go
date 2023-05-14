package tensor

import (
	"fmt"
	"reflect"
	"strconv"
	"strings"
)

type PrintSettings struct {
	whitespace_offset int
	last_char_closed  bool
}

func join_data[T TensorType](sb *strings.Builder, data []T) {
	string_data := make([]string, len(data))
	dtype := get_type_array(data)
	switch dtype.Kind() {
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64,
		reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		for i, val := range data {
			string_data[i] = string(strconv.FormatInt(int64(val), 10))
		}
	case reflect.Float32:
		for i, val := range data {
			string_data[i] = strconv.FormatFloat(float64(val), 'E', -1, 32)
		}
	case reflect.Float64:
		for i, val := range data {
			string_data[i] = strconv.FormatFloat(float64(val), 'E', -1, 64)
		}
	default:
		panic("Got unknown type for stringify data")
	}
	sb.WriteString(strings.Join(string_data, ", "))
}

func _string_repr[T TensorType](sb *strings.Builder, tensor *Tensor[T], ps *PrintSettings) {
	// TODO shrink heigth of printed array
	if ps.last_char_closed {
		for j := 0; j < ps.whitespace_offset; j++ {
			sb.WriteRune(' ')
		}
		ps.last_char_closed = false
	}
	sb.WriteRune('[')
	if len(tensor.shape) == 1 {
		if len(tensor.data) > 30 {
			join_data(sb, tensor.data[:5])
			sb.WriteString("...")
			join_data(sb, tensor.data[len(tensor.data)-5:])
		} else {
			join_data(sb, tensor.data)
		}
		sb.WriteRune(']')
		ps.whitespace_offset -= 1
		ps.last_char_closed = true
		return
	}

	nrows := int(tensor.shape[0])
	for i := 0; i < nrows; i++ {
		if i > 0 {
			sb.WriteRune('\n')
		}
		ps.whitespace_offset += 1
		_string_repr(sb, tensor.Index(i), ps)
		if i == nrows-1 {
			sb.WriteRune(']')
			ps.whitespace_offset -= 1
			ps.last_char_closed = true
		}
	}
}

func (tensor *Tensor[T]) ToString() string {
	var sb strings.Builder
	var ps PrintSettings
	_string_repr(&sb, tensor, &ps)
	str_data := sb.String()

	if len(tensor.shape) > 1 {
		str_data = "\n" + str_data
	}
	str := fmt.Sprintf("Tensor(data=%v, shape=%v, dtype=%v)",
		str_data,
		tensor.shape,
		tensor.dtype.String(),
	)
	return str
}
