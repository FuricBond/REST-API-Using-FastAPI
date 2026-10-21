# Iris Classification API — REST API Using FastAPI

A production-grade RESTful machine learning service that serves predictions from a trained Random Forest model on the Iris flower dataset using FastAPI, Pydantic, scikit-learn, and Joblib.

---

## 1. Project Objective

The objective of this assignment is to develop an end-to-end machine learning web service:
1. Train a classification model on the Iris dataset and save a serialized artifact bundle containing the estimator and its metadata.
2. Build a high-performance REST API with FastAPI that loads the model bundle once at application startup via a lifespan context manager.
3. Validate client requests using strict Pydantic schemas, rejecting malformed or biologically impossible inputs.
4. Provide diagnostics, health monitoring, metadata inspection, and inference endpoints with interactive Swagger documentation.
5. Validate the service through automated unit and integration tests using `pytest` and `httpx`.

---

## 2. Technologies Used

- **Python (3.13)**: Core programming language.
- **FastAPI**: Modern, asynchronous web framework for building REST APIs with automatic OpenAPI schema generation.
- **Pydantic (v2)**: Data parsing, type validation, and constraint enforcement.
- **scikit-learn**: Machine learning library used for dataset loading, train-test splitting, Random Forest training, and accuracy evaluation.
- **Random Forest**: Supervised ensemble learning algorithm composed of multiple decision trees.
- **NumPy**: Fast array operations for formatting input feature tensors.
- **Joblib**: Efficient serialization and deserialization of the trained model and associated metadata.
- **pytest**: Framework for writing and executing automated test suites.
- **HTTPX / TestClient**: In-process HTTP testing client for FastAPI applications.

---

## 3. Project Directory Structure

```text
Assignment 8/
├── artifacts/
│   └── iris_model.joblib       # Serialized model bundle (estimator & metadata)
├── tests/
│   └── test_api.py             # Automated pytest suite (6 test cases)
├── main.py                     # FastAPI application, schemas, lifespan, and endpoints
├── requirements.txt            # Project dependencies
├── train_model.py              # ML training pipeline script
└── README.md                   # Project documentation
```

---

## 4. Setup Instructions (Windows PowerShell)

Follow these steps from the root project directory (`D:\SEM 5\MLAO\Assignment 8`):

### 1. Create a Virtual Environment
```powershell
python -m venv venv
```

### 2. Activate the Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```
*(If script execution is disabled on your system, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` first.)*

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Train the Model and Create Artifacts
```powershell
python train_model.py
```
This trains the Random Forest classifier, evaluates it on the test set, prints training metrics, and exports the model bundle to `artifacts/iris_model.joblib`.

---

## 5. Starting the API

Launch the local development server using Uvicorn:

```powershell
python -m uvicorn main:app --reload
```

- The API server will start at: `http://127.0.0.1:8000`
- The `--reload` flag enables hot reloading whenever code changes are saved.

---

## 6. API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Root endpoint returning a welcome message and links to docs/endpoints. |
| `GET` | `/health` | Reports service status and whether the ML model is currently loaded in memory. |
| `GET` | `/model-info` | Returns model type, version, feature names, target names, and test accuracy. |
| `POST` | `/predict` | Validates 4 flower measurements and returns species prediction and probabilities. |

---

## 7. Sample Prediction Request & Response

### Request (`POST /predict`)
- **URL**: `http://127.0.0.1:8000/predict`
- **Headers**: `Content-Type: application/json`
- **Body**:
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

### Response (`HTTP 200 OK`)
```json
{
  "predicted_class": 0,
  "predicted_species": "setosa",
  "confidence": 1.0,
  "probabilities": {
    "setosa": 1.0,
    "versicolor": 0.0,
    "virginica": 0.0
  },
  "model_version": "1.0.0"
}
```

---

## 8. Input Validation & Error Handling

### Pydantic Validation Constraints
Input data is strictly validated against the `PredictionRequest` schema:
- `sepal_length`: `float`, `gt=0` (greater than 0), `le=20.0` (less than or equal to 20 cm)
- `sepal_width`: `float`, `gt=0`, `le=20.0`
- `petal_length`: `float`, `gt=0`, `le=20.0`
- `petal_width`: `float`, `gt=0`, `le=20.0`

### HTTP 422 Unprocessable Entity
If a client sends non-positive values (e.g., `-5.1` or `0`), values exceeding 20 cm, non-numeric strings, or omits a required field, FastAPI automatically responds with `HTTP 422 Unprocessable Entity` containing detailed field-level error messages.

### Explanation of Prediction Response Fields
- `predicted_class`: Integer class index (`0` for setosa, `1` for versicolor, `2` for virginica).
- `predicted_species`: Human-readable species string mapped directly from the saved bundle.
- `confidence`: The highest probability among all classes, rounded to 4 decimal places (e.g., `1.0` or `0.9700`).
- `probabilities`: Dictionary mapping each Iris species name to its individual predicted probability, each rounded to 4 decimal places.
- `model_version`: Semantic version string (`"1.0.0"`) of the active model bundle.

### Controlled Internal Error Masking (HTTP 500)
Unexpected runtime inference errors are caught by an internal error barrier. The server returns a clean `HTTP 500 Internal Server Error` with `{"detail": "Prediction failed due to an internal server error."}` so that internal stack traces and implementation details are never leaked to clients.

---

## 9. Interactive API Documentation (Swagger UI)

FastAPI automatically generates interactive OpenAPI documentation:

- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

You can explore endpoints, view JSON schemas, and test requests directly in your browser.

---

## 10. Automated Testing

The automated test suite in `tests/test_api.py` utilizes `pytest` and `fastapi.testclient.TestClient`. It contains 6 distinct test cases:

1. `test_root_endpoint`: Confirms `GET /` returns HTTP 200 with navigation links.
2. `test_health_endpoint`: Confirms `GET /health` returns HTTP 200 and `model_loaded: true`.
3. `test_model_info_endpoint`: Confirms `GET /model-info` returns HTTP 200 with model version, features, and accuracy.
4. `test_predict_valid_input`: Confirms `POST /predict` returns HTTP 200, class `0`, species `"setosa"`, and probability distribution.
5. `test_predict_negative_measurement`: Confirms `POST /predict` rejects negative values with HTTP 422.
6. `test_predict_missing_measurement`: Confirms `POST /predict` rejects payloads missing required fields with HTTP 422.

### Running Tests
Execute the following command from the root project directory:
```powershell
python -m pytest -v
```

### Execution Results
```text
tests/test_api.py::test_root_endpoint PASSED                             [ 16%]
tests/test_api.py::test_health_endpoint PASSED                           [ 33%]
tests/test_api.py::test_model_info_endpoint PASSED                       [ 50%]
tests/test_api.py::test_predict_valid_input PASSED                       [ 66%]
tests/test_api.py::test_predict_negative_measurement PASSED              [ 83%]
tests/test_api.py::test_predict_missing_measurement PASSED               [100%]

======================== 6 passed, 1 warning in 3.02s =========================
```

---

## 11. Trained Model Details

- **Model Type**: `RandomForestClassifier` (`sklearn.ensemble`)
- **Number of Estimators (Trees)**: `100` (`random_state=42`)
- **Total Dataset Size**: 150 samples across 3 classes (Setosa, Versicolor, Virginica)
- **Feature Set (4 features)**:
  1. `sepal length (cm)`
  2. `sepal width (cm)`
  3. `petal length (cm)`
  4. `petal width (cm)`
- **Data Splitting**: Stratified 80/20 train-test split (`random_state=42`, `stratify=y`)
  - **Training Records**: 120 samples
  - **Testing Records**: 30 samples (10 of each species)
- **Actual Test Accuracy**: `0.9000` (90.00%)
- **Model Version**: `1.0.0`
- **Saved Artifact**: `artifacts/iris_model.joblib`

---

## 12. Limitations & Conclusion

### Limitations
1. **Dataset Scope**: The model is trained on Fisher's Iris dataset (150 samples), which is a small benchmark dataset suitable for academic demonstration rather than diverse field botany.
2. **Feature Range Constraints**: The API rejects measurements $> 20\text{ cm}$, which prevents out-of-distribution inputs but reflects fixed boundary assumptions.
3. **Serialization Portability**: Model artifacts are stored using Joblib, which requires matching Python and scikit-learn major versions between training and inference environments.
4. **Single-Node Execution**: Inference runs in-process on the local host CPU. Large-scale enterprise deployments would require containerization and distributed inference workers.

### Conclusion
This project successfully establishes an end-to-end Machine Learning deployment pipeline. By combining scikit-learn's modeling capabilities with FastAPI's asynchronous architecture, Pydantic's data validation, and Joblib's serialization, the resulting service is robust, self-documenting, and verifiable through automated tests.
