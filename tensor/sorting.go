package tensor

func bubbleSort[T TensorType](slice []T) {
	n := len(slice)
	for i := 0; i < n-1; i++ {
		for j := 0; j < n-i-1; j++ {
			if slice[j] > slice[j+1] {
				slice[j+1], slice[j] = slice[j], slice[j+1]
			}
		}
	}
}

func quickSort[T TensorType](arr []T, low int, high int) {
	if low < high {
		/* pi is partitioning index, arr[pi] is now at right place */
		pi := qsPartition(arr, low, high)
		quickSort(arr, low, pi-1)
		quickSort(arr, pi+1, high)
	}
}

/*
This function takes last element as pivot, places the pivot element
at its correct position in sorted array, and places all smaller
(smaller than pivot) to left of pivot and all greater elements to right of pivot
*/
func qsPartition[T TensorType](arr []T, low int, high int) int {
	// pivot (Element to be placed at right position)
	pivot := arr[high]
	i := (low - 1) // Index of smaller element and indicates the
	// right position of pivot found so far

	for j := low; j <= high-1; j++ {
		// If current element is smaller than the pivot
		if arr[j] < pivot {
			i++ // increment index of smaller element
			arr[i], arr[j] = arr[j], arr[i]
		}
	}
	arr[i+1], arr[high] = arr[high], arr[i+1]
	return i + 1
}

func (tensor *Tensor[T]) Sort() *Tensor[T] {
	data := tensor.data
	if len(data) <= 1 {
		return tensor.Copy()
	}
	outTensor := tensor.Copy()
	quickSort(outTensor.data, 0, len(tensor.data)-1)
	return outTensor
}
