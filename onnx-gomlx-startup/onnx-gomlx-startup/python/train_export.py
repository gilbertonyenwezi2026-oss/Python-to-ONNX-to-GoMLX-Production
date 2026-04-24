"""Train a quality-risk model in Python and export it to ONNX.
Synthetic operations data: predict whether a production batch is high defect risk.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
for d in [MODEL_DIR, DATA_DIR, RESULTS_DIR]:
    d.mkdir(exist_ok=True)

rng = np.random.default_rng(431)
X, y = make_classification(
    n_samples=250_000,
    n_features=8,
    n_informative=6,
    n_redundant=1,
    n_classes=2,
    weights=[0.78, 0.22],
    random_state=431,
)
feature_names = [
    "line_speed", "temperature", "humidity", "pressure",
    "operator_experience", "material_age", "shift_load", "sensor_noise"
]
X = X.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=100_000, random_state=431, stratify=y
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000, n_jobs=-1)),
])
pipeline.fit(X_train, y_train)

start = time.perf_counter()
python_pred = pipeline.predict(X_test)
python_elapsed = time.perf_counter() - start
accuracy = accuracy_score(y_test, python_pred)

np.save(DATA_DIR / "X_test.npy", X_test)
np.save(DATA_DIR / "y_test.npy", y_test)
np.savetxt(DATA_DIR / "X_test.csv", X_test, delimiter=",", fmt="%.7f")
np.savetxt(DATA_DIR / "python_predictions.csv", python_pred, delimiter=",", fmt="%d")
joblib.dump(pipeline, MODEL_DIR / "quality_risk_pipeline.joblib")

initial_type = [("float_input", FloatTensorType([None, X_test.shape[1]]))]
onnx_model = convert_sklearn(pipeline, initial_types=initial_type, target_opset=15)
(MODEL_DIR / "quality_risk_pipeline.onnx").write_bytes(onnx_model.SerializeToString())

metrics = {
    "model": "StandardScaler + LogisticRegression",
    "test_rows": int(X_test.shape[0]),
    "features": feature_names,
    "python_seconds": python_elapsed,
    "python_rows_per_second": float(X_test.shape[0] / python_elapsed),
    "accuracy": float(accuracy),
}
(RESULTS_DIR / "python_metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
