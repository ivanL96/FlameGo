package datasets

import (
	"encoding/csv"
	"os"
	"strconv"
)

func LoadIris(path string) *DataSet[float32] {

	file, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()

	if err != nil {
		panic(err)
	}
	X_train := make([][]float32, 0, len(records)*4)
	Y_train := make([][]float32, 0, len(records))
	// X_test := make([]float32, 0, test_records_len*4)
	// Y_test := make([]float32, 0, test_records_len)

	for j := 0; j < len(records); j++ {
		record := records[j]
		f1, _ := strconv.ParseFloat(record[1], 32)
		f2, _ := strconv.ParseFloat(record[2], 32)
		f3, _ := strconv.ParseFloat(record[3], 32)
		f4, _ := strconv.ParseFloat(record[4], 32)
		f5, _ := strconv.ParseFloat(record[5], 32)
		r := []float32{
			float32(f1),
			float32(f2),
			float32(f3),
			float32(f4),
		}
		// if j < train_records_len {
		X_train = append(X_train, r)
		Y_train = append(Y_train, []float32{float32(f5)})
		// 	continue
		// }
		// X_test = append(X_test, r...)
		// Y_test = append(Y_test, float32(f5))
	}
	// return X_train, Y_train, X_test, Y_test
	return &DataSet[float32]{X_train, Y_train}
}
