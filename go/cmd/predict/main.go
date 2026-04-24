package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"
	"time"
)

type Metrics struct {
	Rows             int     `json:"rows"`
	GoSeconds        float64 `json:"go_seconds"`
	GoRowsPerSecond  float64 `json:"go_rows_per_second"`
	MatchRateVsPython float64 `json:"match_rate_vs_python"`
	Note             string  `json:"note"`
}

func readCSV(path string) ([][]float64, error) {
	file, err := os.Open(path)
	if err != nil { return nil, err }
	defer file.Close()
	rows, err := csv.NewReader(file).ReadAll()
	if err != nil { return nil, err }
	data := make([][]float64, len(rows))
	for i, row := range rows {
		data[i] = make([]float64, len(row))
		for j, raw := range row {
			v, err := strconv.ParseFloat(raw, 64)
			if err != nil { return nil, fmt.Errorf("row %d col %d: %w", i, j, err) }
			data[i][j] = v
		}
	}
	return data, nil
}

func main() {
	modelPath := flag.String("model", "../models/quality_risk_pipeline.onnx", "path to ONNX model")
	inputPath := flag.String("input", "../data/X_test.csv", "CSV test input")
	outPath := flag.String("out", "../results/go_metrics.json", "metrics output JSON")
	flag.Parse()

	data, err := readCSV(*inputPath)
	if err != nil { log.Fatal(err) }

	start := time.Now()
	// Assignment integration point:
	// Use github.com/gomlx/onnx-gomlx/onnx to read modelPath and execute through GoMLX.
	// See README for the exact GoMLX API pattern. This starter keeps CLI, timing,
	// validation, and file structure ready while isolating inference in one place.
	_ = *modelPath
	predictions := make([]int, len(data))
	for i := range predictions { predictions[i] = 0 }
	elapsed := time.Since(start).Seconds()

	metrics := Metrics{
		Rows: len(data),
		GoSeconds: elapsed,
		GoRowsPerSecond: float64(len(data))/elapsed,
		MatchRateVsPython: 0.0,
		Note: "Wire predictions to GoMLX/ONNX output, then compute match rate against data/python_predictions.csv.",
	}
	b, _ := json.MarshalIndent(metrics, "", "  ")
	if err := os.WriteFile(*outPath, b, 0644); err != nil { log.Fatal(err) }
	fmt.Println(string(b))
}
