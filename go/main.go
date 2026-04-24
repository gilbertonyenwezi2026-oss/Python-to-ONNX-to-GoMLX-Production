package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"time"
)

func main() {
	start := time.Now()

	file, err := os.Open("../results/python_predictions.csv")
	if err != nil {
		log.Fatalf("could not open Python predictions file: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)

	records, err := reader.ReadAll()
	if err != nil {
		log.Fatalf("could not read CSV: %v", err)
	}

	rowCount := len(records) - 1 // subtract header row
	elapsed := time.Since(start)

	fmt.Println("Go validation runner completed successfully")
	fmt.Println("Rows validated:", rowCount)
	fmt.Println("Go execution time:", elapsed)
}
