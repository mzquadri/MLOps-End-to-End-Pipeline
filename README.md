# MLOps End-to-End Pipeline

Production-grade MLOps pipeline demonstrating experiment tracking, model versioning, automated validation, and deployment-ready packaging. Built with MLflow, DVC concepts, and CI/CD-ready architecture.

## Project Overview

This project implements a complete MLOps lifecycle for a text classification task (sentiment analysis), showcasing industry best practices for:

- **Experiment Tracking**: MLflow-based logging of parameters, metrics, and artifacts
- **Data Versioning**: DVC-inspired data pipeline with hash-based tracking
- **Model Registry**: Version-controlled model storage with stage management (staging/production)
- **Automated Validation**: Data quality checks, model performance gates, and drift detection
- **Deployment Packaging**: Docker-ready FastAPI inference service
- **CI/CD Pipeline**: GitHub Actions workflow for automated testing and deployment

## Project Structure

```
MLOps-End-to-End-Pipeline/
├── README.md
├── requirements.txt
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── configs/
│   ├── train_config.yaml        # Training hyperparameters
│   └── deploy_config.yaml       # Deployment configuration
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py         # Data loading, validation, versioning
│   ├── feature_store.py         # Feature computation and caching
│   ├── train.py                 # Training with MLflow experiment tracking
│   ├── evaluate.py              # Model evaluation and performance gates
│   ├── model_registry.py        # Model versioning and stage management
│   └── serve.py                 # FastAPI inference service
├── tests/
│   ├── test_data_pipeline.py    # Data pipeline unit tests
│   ├── test_model.py            # Model training/inference tests
│   └── test_api.py              # API endpoint tests
├── notebooks/
│   └── 01_MLOps_Walkthrough.ipynb
├── data/                        # Dataset directory
├── models/                      # Saved model artifacts
└── results/                     # Evaluation reports
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/mzquadri/MLOps-End-to-End-Pipeline.git
cd MLOps-End-to-End-Pipeline

# Install dependencies
pip install -r requirements.txt

# Run data pipeline
python src/data_pipeline.py --config configs/train_config.yaml

# Train with experiment tracking
python src/train.py --config configs/train_config.yaml --experiment "sentiment-v1"

# Evaluate and validate
python src/evaluate.py --model_path models/latest --threshold 0.85

# Register model
python src/model_registry.py --register --stage production

# Serve model (local)
python src/serve.py --model_path models/production
# API available at http://127.0.0.1:8000

# Docker deployment
docker-compose up --build
```

## MLOps Components

| Component | Tool/Approach | Description |
|-----------|--------------|-------------|
| Experiment Tracking | MLflow | Log params, metrics, artifacts per run |
| Data Versioning | Hash-based (DVC-like) | Track dataset versions via content hashing |
| Model Registry | MLflow + custom | Stage management: None → Staging → Production |
| Validation Gates | Custom | Min accuracy, max drift thresholds |
| Serving | FastAPI | REST API with health checks, batch prediction |
| Containerization | Docker | Reproducible deployment packaging |
| Testing | pytest | Unit + integration tests |
| CI/CD | GitHub Actions | Automated test → train → validate → deploy |

## Pipeline Architecture

```
[Raw Data] → [Data Pipeline] → [Feature Store] → [Training]
                    ↓                                  ↓
            [Data Validation]              [MLflow Experiment Tracking]
                                                       ↓
                                             [Model Evaluation]
                                                       ↓
                                              [Performance Gate]
                                                   ↓     ↓
                                               PASS      FAIL → Alert
                                                 ↓
                                         [Model Registry]
                                                 ↓
                                    [Staging → Production]
                                                 ↓
                                       [FastAPI Service]
                                                 ↓
                                         [Docker Deploy]
```

## Key Features

- **Reproducible Experiments**: Every training run is logged with full parameter and metric history
- **Data Drift Detection**: Statistical tests (KS-test, PSI) to detect distribution shifts
- **Performance Gates**: Automatic promotion/rejection based on accuracy and fairness thresholds
- **Model Lineage**: Full traceability from data version → training run → deployed model
- **Health Monitoring**: API health checks and prediction latency tracking

## Technical Stack

- **ML**: scikit-learn, TF-IDF
- **MLOps**: MLflow, DVC concepts
- **API**: FastAPI, Uvicorn, Pydantic
- **Containerization**: Docker, docker-compose
- **Testing**: pytest
- **Configuration**: PyYAML, Hydra-style configs

## Alignment with Experience

This project reflects MLOps engineering practices from **BP-ITCS**, including:
- Experiment tracking and model versioning for production ML systems
- CI/CD pipeline design for automated model training and deployment
- Containerized model serving for enterprise applications
- Data validation and monitoring for reliability

## Author

**Mohd Zamin Quadri** - M.Sc. Mathematics in Science and Engineering, Technical University of Munich

[![LinkedIn](https://img.shields.io/badge/LinkedIn-mohd--zamin-blue)](https://www.linkedin.com/in/mohd-zamin/)
[![GitHub](https://img.shields.io/badge/GitHub-mzquadri-black)](https://github.com/mzquadri)
