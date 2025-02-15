# Air Pollution Prediction System

## Project Overview
The **Air Pollution Prediction System** is a machine learning application designed to predict air quality using various environmental features. The project includes:
- **Data Pipeline**: For loading, transforming, and preprocessing data.
- **Model Training and Inference Pipelines**: For training and deploying machine learning models.
- **Monitoring and Metrics**: Powered by **Prometheus**, **Grafana**, and **MLflow** for real-time tracking of metrics and model performance.
- **API**: A **FastAPI-based RESTful API** for health checks, training pipelines, and inference.

---

## Features
- **Preprocessing**: Scalable data transformation and encoding using configurable pipelines.
- **Model Training**: Supports `DecisionTree` and `LogisticRegression` models.
- **Monitoring**:
  - Prometheus and Grafana for visualizing metrics.
  - Alert rules to identify pipeline failures or low model performance.
- **Testing**: Comprehensive test suite using **pytest** for all components.
- **MLOps**:
  - MLflow integration for experiment tracking, artifact management, and model registry.
  - CI/CD workflows for automated testing, linting, and deployment via **GitHub Actions**.

---

## Dataset
- **Source**: [Air Quality and Pollution Assessment](https://www.kaggle.com/datasets/mujtabamatin/air-quality-and-pollution-assessment) from Kaggle.
- **Description**:
  - Contains air quality data across various regions.
  - Features include particulate matter (PM2.5, PM10), CO, NO2, SO2, temperature, humidity, proximity to industrial areas, and population density.

---

## Project Structure
```
.
├── .github/
│   ├── actions/
│   │   └── setup/
│   ├── workflows/
│   │   ├── ci.yml
├── config/
│   ├── config_dev.yml
├── data/
│   └── updated_pollution_dataset.csv
├── mlruns/  # MLflow experiment storage
├── src/
│   ├── air_pollution/
│   │   ├── data_loader/
│   │   ├── data_pipeline/
│   │   ├── endpoints/
│   │   ├── model/
│   │   ├── scripts/
│   │   ├── main.py
├── tests/
│   ├── test_data_loader.py
│   ├── test_data_transformer.py
│   ├── test_model.py
│   ├── test_preprocessing.py
│   ├── test_label_encoder.py
├── docker-compose.yml
├── Dockerfile
├── README.md
├── poetry.lock
├── pyproject.toml 
...
```

---

### Setup Instructions

#### Prerequisites
- Python 3.9 or above
- Poetry for dependency management
- Docker and Docker Compose for containerized deployment

#### Local Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```

2. Install dependencies with Poetry:
   ```bash
   poetry install
   ```

3. Run the API locally:
   ```bash
   poetry run uvicorn src.air_pollution.main:app --reload
   ```

4. Access the API at [http://127.0.0.1:8000](http://127.0.0.1:8000)

#### Using Docker

1. Build and start all services:
   ```bash
   docker-compose up --build
   ```

2. Access:
   - API: [http://127.0.0.1:8000](http://127.0.0.1:8000)
   - MLflow: [http://127.0.0.1:5000](http://127.0.0.1:5000)
   - Grafana: [http://127.0.0.1:3000](http://127.0.0.1:3000)
   - Prometheus: [http://127.0.0.1:9090](http://127.0.0.1:9090)

#### Training

To train the model using mlflow, use the following command:
```bash
poetry run air_pollution_train --config config/config_dev.yaml
```

#### Inference

For interactive inference:
```bash
poetry run air_pollution_inference --interactive
```

For inference with a file:
```bash
poetry run air_pollution_inference --file input_data.json
```

For inference with data:
```bash
poetry run air_pollution_inference --data "{\"Temperature\": 28.3, \"Humidity\": 75.6, \"PM2.5\": 35.2, \"CO2\": 420, \"NO2\": 23.5}"
```
---

## Key Components

### 1. **API Endpoints**
- **Health Check**: `/api/health`
  - Returns system status.
- **Train Pipeline**: `/api/train`
  - Triggers model training using a specified configuration file.
- **Inference Pipeline**: `/api/predict`
  - Accepts input data and returns predictions.

### 2. **Configurations**
Defined in `config/config_dev.yml`:
- **Data Loader**:
  - File Path: Path to the dataset file, e.g., data/updated_pollution_dataset.csv.
  - File Type: Format of the dataset (e.g., csv, json).
- **Transformation**:
  - Normalization:  Flag indicating if data normalization should be applied (true or false).
  - Scaling Method:  The scaling method used, such as minmax, or standard .
- **Model**:
  - Type: Type of machine learning model used (decisiontree, logistic).
  - Parameters: A dictionary containing hyperparameters and configuration specific to the model, e.g., {} for default parameters.

### 3. **Monitoring and Alerting**
- **Prometheus**:
  - Tracks pipeline and model performance.
  - Some Metrics include:
    - `predict_requests_total`
    - `training_accuracy`
    - `training_f1_score`
- **Grafana**:
  - Visualizes Prometheus metrics.
- **Alert Rules** (defined in `alert_rules.yml`):
  - Alerts for low accuracy or F1 score.

### 4. **CI/CD**
GitHub Actions for:
- Testing (`pytest`).
- Linting (`ruff`).
- Type Checking (`mypy`).

### 5. **Docs** 
To know more about the files and their usage, you can view the documentation generated in the doc directory. This includes detailed explanations and guides

---

## Testing
Run tests with:
```bash
poetry invoke test
```
Test files:
- `test_data_loader.py`: Tests for CSV and JSON data loading.
- `test_data_transformer.py`: Tests for scaling transformations.
- `test_model.py`: Tests for logistic regression and decision tree models.
- `test_preprocessing.py`: Tests preprocessing pipelines.
- `test_label_encoder.py`: Tests label encoding functionality.

---

## Dependencies
Key libraries:
- **FastAPI**: RESTful API framework.
- **MLflow**: Experiment tracking and model management.
- **Prometheus & Grafana**: Monitoring and visualization.
- **Scikit-learn**: ML model training.
- **Pandas**: Data manipulation.

Dev tools:
- **pytest**: Testing.
- **mypy**: Type checking.
- **ruff**: Linting.

---

## Acknowledgments
- Dataset by Mujtaba Matin: [Kaggle Link](https://www.kaggle.com/datasets/mujtabamatin/air-quality-and-pollution-assessment).

---
