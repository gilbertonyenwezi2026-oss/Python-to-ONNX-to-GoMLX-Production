# Python-to-ONNX-to-GoMLX Production Inference Experiment

## Management problem
A technology startup wants Python's flexibility during model development, but is concerned about Python's cost and performance in production. This project tests a practical deployment pattern: train a model in Python, export it to ONNX, then run inference in Go using the GoMLX/ONNX ecosystem.

## Model and use case
The model predicts high-defect-risk production batches from synthetic operational features such as line speed, temperature, humidity, pressure, operator experience, material age, shift load, and sensor noise. This maps the assignment to an Operational Excellence and quality-control setting.

## Repository structure
```text
python/train_export.py          Train Scikit-Learn model, benchmark Python, export ONNX
python/requirements.txt         Python dependencies
go/cmd/predict/main.go          Go CLI timing scaffold for ONNX/GoMLX inference
go/go.mod                       Go module definition
data/                           Generated test data and Python predictions
models/                         Generated ONNX and joblib models
results/                        Python and Go benchmark JSON files
llm_dialogs/                    Plain-text AI assistant dialog notes
```

## Run Python training and export
```bash
cd onnx-gomlx-startup
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r python/requirements.txt
python python/train_export_onnx.py
```

Outputs:
- `models/quality_risk_pipeline.onnx`
- `data/X_test.csv`
- `data/python_predictions.csv`
- `results/python_metrics.json`

## Run Go timing application
```bash
cd go
go mod tidy
go run ./cmd/predict -model ../models/quality_risk_pipeline.onnx -input ../data/X_test.csv -out ../results/go_metrics.json
```

Build executable:
```bash
cd go
go build -o ../bin/qc_predict.exe ./cmd/predict   # Windows
# or
go build -o ../bin/qc_predict ./cmd/predict       # macOS/Linux
```

## GoMLX/ONNX integration note
GoMLX is an accelerated ML framework for Go, and `onnx-gomlx` converts ONNX models into GoMLX graphs for inference or fine-tuning. The production implementation should place the ONNX execution logic behind one Go function so the CLI, metrics, and validation stay unchanged.

Expected GoMLX flow:
1. Read the `.onnx` file with `onnx.ReadFile()`.
2. Move ONNX variables into a GoMLX context.
3. Execute `model.CallGraph()` through a GoMLX backend.
4. Convert model outputs into class predictions.
5. Compare Go predictions with `python_predictions.csv`.

## Benchmark table
After running both sides, update this table:

| Runtime               | Test Rows | Execution Time | Rows/Second | Accuracy |
|---|---:|---:|---:|---:|
| Python / Scikit-Learn | 100,000   | 0.005618 sec   | 17,800,246 | 0.85678   |
| Go validation runner  | 100,000   | 0.008938 sec   | 11,188,686 | N/A       |



| Runtime               | Rows    | Execution time                    | Rows/sec | Match rate vs Python |
|---|---:|---:|---:|---:|
| Python / Scikit-Learn | 100,000 | see `results/python_metrics.json` | see JSON | baseline             |
| Go / GoMLX ONNX       | 100,000 | see `results/go_metrics.json`     | see JSON | target: 99.9–100%     |

## Consultant recommendation
Python continues to be the optimal environment for experimentation, model training, and enhancing analyst productivity. ONNX offers a portability layer for transferring learned models into production environments. Go is appealing for production due to its straightforward deployment, rapid startup, generated binaries, and robust concurrency capabilities for handling thousands of requests. The suggested architecture entails training in Python, exporting to ONNX, validating numerical equivalence, and deploying the Go executable behind an API service.

## References
- ONNX supported tools documentation.
- GoMLX GitHub documentation.
- ONNX-GoMLX GitHub documentation.

## Author
Gilbert Onyenwezi

MSDS 431 - GO & AI Programming
- Northwestern University
- School of Professional Studies
- Masters in Data Science - Data Engineering Specialization
- 2026

