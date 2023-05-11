package tensor

func create_slice[T any](n int, value T) []T {
	slice := make([]T, n)
	for i := range slice {
		slice[i] = value
	}
	return slice
}

func reverse_slice_inplace[T any](slice []T) []T {
	for i := len(slice)/2 - 1; i >= 0; i-- {
		opp := len(slice) - 1 - i
		slice[i], slice[opp] = slice[opp], slice[i]
	}
	return slice
}

func reverse_slice_copy[T any](slice []T) []T {
	rev_slice := make([]T, len(slice))
	for i := range slice {
		rev_slice[i] = slice[len(slice)-1-i]
	}
	return rev_slice
}

func Compare_slices[T Number](slice1, slice2 []T) bool {
	if len(slice1) != len(slice2) {
		return false
	}
	for i, element := range slice1 {
		if element != slice2[i] {
			return false
		}
	}
	return true
}
