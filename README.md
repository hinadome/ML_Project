# Capstone Project: Proactive Cloud Operations Engine
## ML-Driven Auto-Scaling

### 1. Project Vision
An end-to-end MLOps solution that transforms reactive cloud infrastructure into a proactive, self-scaling system. By analyzing historical traffic logs, the system predicts upcoming traffic and gives signal/trigger to scaling system.

---

### 2. Core Functional Pillars
* **Proactive Auto-Scaling:** Uses Time-Series Forecasting (XGBoost & LSTM) to anticipate traffic spikes and adjust cloud capacity ( instance ).

---

### 3. Technology Stack
| Domain | Technologies |
| :--- | :--- |
| **Machine Learning** | Python, Scikit-learn, XGBoost, PyTorch (LSTM) |
| **Cloud Infrastructure** | GCP (Cloud Run, Artifact Registry) |
| **DevOps & MLOps** | Docker, Terraform (IaC), GitHub Actions (CI/CD) |
| **Application Layer** | FastAPI (REST API), Uvicorn |

---

### 4. Implementation Phases (Milestones)

#### Phase 1: Data Engineering & EDA (2d)
* **Ingestion:** Parse [raw Apache/NASA logs](https://ita.ee.lbl.gov/html/contrib/NASA-HTTP.html)  using Regex into structured Pandas DataFrames.
* **Validation:** Clean data, handle missing values, and ensure time-series continuity.
* **EDA:** Visualize seasonal traffic patterns and error rate distributions.

#### Phase 2: Feature Engineering & Modeling(2d)
* **Temporal Features:** Implement Lag features, rolling statistics, and cyclical encoding (Sin/Cos).
* **Training:** Develop and compare RandomForest, GradientBoostingRegressor(XGBoost), and LSTM models.
* **Evaluation:** Measure performance using MAE and RMSE metrics to minimize over-provisioning.

#### Phase 3: System Architecture(3d) 
* **API Development:** Wrap models into a FastAPI service for real-time inference.
* **Containerization:** Build optimized Docker images including the trained model artifacts.
* **Infrastructure:** Provision cloud resources (AWS/GCP) using Terraform.

#### Phase 4: Automation (CI/CD)(2d)
* **Pipeline:** Orchestrate a GitHub Actions workflow to automate training, building, and deployment.

#### Phase 5: Monitoring(2d)
* **Monitoring:** Implement health checks and logging for the deployed API.

