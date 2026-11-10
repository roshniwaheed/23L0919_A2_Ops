# VisionGuard — AI-Generated vs Real Art Detection

[![VisionGuard CI](https://github.com/roshniwaheed/23L0919_A2_Ops/actions/workflows/visionguard-ci.yml/badge.svg)](https://github.com/roshniwaheed/23L0919_A2_Ops/actions/workflows/visionguard-ci.yml)

VisionGuard is an end-to-end Machine Learning Operations (MLOps) project that classifies artwork as **AI-generated** or **real** using deep learning.

The project demonstrates dataset versioning, experiment tracking, model evaluation, web application development, containerization, database integration, and automated continuous integration.

## Technology Stack

| Component | Technology |
|---|---|
| Machine learning | TensorFlow, Keras, MobileNetV2 |
| Experiment tracking | MLflow, DagsHub |
| Dataset and model versioning | DVC |
| Web application | Flask |
| Database | SQLite (local), PostgreSQL (Docker) |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Code quality and testing | Ruff, pytest |
| Programming language | Python 3.11 |

## Dataset

The dataset contains artwork from two classes: AI-generated and real.

| Split | Images |
|---|---:|
| Training | 681 |
| Validation | 146 |
| Testing | 147 |
| **Total** | **974** |

The dataset is versioned using DVC. Raw images are excluded from ordinary Git history.

## Model Development

VisionGuard evaluates a baseline CNN and a transfer-learning model based on MobileNetV2.

The selected champion model is stored at:

`models/mobilenet_best.keras`

The trained models are tracked through DVC and stored in DagsHub artifact storage.

### Champion Model Performance

| Metric | Value |
|---|---:|
| Test accuracy | 73.47% |
| Test ROC-AUC | 0.8204 |
| Classification threshold | 0.653947 |

Model experiments and performance metrics are tracked using MLflow.

## Application Features

- Upload an artwork image for classification.
- Predict whether the image is AI-generated or real.
- Display prediction results and confidence information.
- Store prediction history in a database.
- View previous predictions.
- Provide a health-check endpoint for monitoring.

### Flask Endpoints

| Endpoint | Purpose |
|---|---|
| `/health` | Application health check |
| `/predict` | Image classification |
| `/history` | Prediction history |

## Docker and Docker Compose

VisionGuard uses a Dockerfile to package the Flask application and its TensorFlow dependencies.

Docker Compose runs two services:

- **Web:** Flask application served using Gunicorn.
- **Database:** PostgreSQL with a persistent named volume.

The application communicates with PostgreSQL through Docker Compose's internal network.

### Running the Application

Create a local `.env` file containing the PostgreSQL credentials and required application configuration.

Then run:

```bash
docker compose up -d --build
```

Access the application at:

`http://localhost:5002`

To stop the services:

```bash
docker compose down
```

The PostgreSQL named volume preserves stored prediction history between container restarts.

## Data and Model Versioning

DVC is used to track the dataset and trained models.

Tracked metadata files include:

- `data/raw.dvc`
- `models.dvc`

DagsHub provides remote DVC artifact storage.

To retrieve the versioned trained models with authorized DagsHub access:

```bash
dvc pull -r dagshub models.dvc
```

The actual model binaries are not committed to Git.

## CI/CD — GitHub Actions

VisionGuard uses GitHub Actions to automate code quality checks and Docker build validation.

The CI workflow is located at:

`.github/workflows/visionguard-ci.yml`

### CI Pipeline

1. Check out the source code.
2. Configure Python 3.11.
3. Run Ruff linting.
4. Run automated smoke tests using pytest.
5. Authenticate with DagsHub through the `DAGSHUB_TOKEN` GitHub Actions secret.
6. Download the versioned trained models using DVC.
7. Build the Docker image containing the actual champion model.

The pipeline runs on pushes and pull requests targeting `main` and can also be triggered manually.

### Security

- Secrets are excluded from Git.
- DagsHub authentication is configured using GitHub Actions Secrets.
- Trained model binaries are managed using DVC.
- The Docker application runs as a non-root user.

### CI Status

The badge at the top of this README displays the latest GitHub Actions workflow status.

The pipeline validates the Docker image build but does not automatically deploy it.

## Project Repositories

**GitHub:** https://github.com/roshniwaheed/23L0919_A2_Ops

**DagsHub:** https://dagshub.com/roshniwaheed/23L0919_A2_Ops

## Future Work

- Deploy the trained model as a public cloud inference API.
- Integrate the cloud API into the Flask application.
- Extend automated tests to cover API behavior and inference responses.