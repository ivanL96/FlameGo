package datasets

import (
	"encoding/csv"
	"os"
	"strconv"
)

func LoadIris(path string) ([]float32, []float32) {

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
	X := make([]float32, 0, len(records)*4)
	Y := make([]float32, 0, len(records))
	for _, record := range records {
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
		X = append(X, r...)
		Y = append(Y, float32(f5))
	}
	return X, Y
}
