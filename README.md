# ⚡ End-to-End Energy Forecasting Pipeline with MLOps

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-GPU-green?logo=xgboost&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Fast-orange)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker&logoColor=white)

An end-to-end Machine Learning pipeline designed to forecast energy consumption of buildings based on the **ASHRAE - Great Energy Predictor III** dataset.

This project demonstrates a **production-ready approach** to handling large-scale tabular data (20M+ rows) on consumer hardware (6GB VRAM) by implementing a **Hybrid CPU/GPU Pipeline**, **Advanced Feature Engineering**, and **Model Ensembling**.

---

## 🚀 Key Engineering Highlights

### 1. Hybrid CPU/GPU Architecture (Solving the OOM Bottleneck)

- **Challenge:** The dataset (20M rows) and feature engineering steps exceeded the 6GB VRAM limit of the GPU (RTX 3050), causing `CUDA Out Of Memory` errors during ETL.
- **Solution:** Decoupled the pipeline into two stages:
  - **ETL & Preprocessing:** Executed on **CPU (Pandas)** with 16GB RAM to handle heavy transformations (Timezone alignment, Weather interpolation).
  - **Training:** Executed on **GPU (XGBoost/LightGBM)** using optimized memory structures (Quantile DMatrix / Histogram methods).

### 2. Winning Solution Strategies (Data Cleaning)

Implemented critical preprocessing steps derived from the Kaggle competition winning solutions:

- **🔧 Unit Correction:** Fixed the "Site 0" meter reading error where electricity was recorded in kBTU instead of kWh (Scaling factor: 0.2931).
- **🌍 Timezone Alignment:** Aligned UTC timestamps to local solar time for each site to capture accurate daily patterns.
- **🧹 Anomaly Detection:** Removed zero-reading artifacts and extreme outliers to prevent model confusion.
- **☁️ Weather Imputation:** Applied linear interpolation to fill missing air temperature data, preserving temporal continuity.

### 3. Weighted Ensemble (Meta-Modeling)

Instead of relying on a single model, an ensemble strategy was used to minimize variance and improve generalization:

- **Model 1:** XGBoost (GPU Hist) - High performance on Public LB.
- **Model 2:** LightGBM (Leaf-wise) - Better generalization on Private LB.
- **Strategy:** Inverse Error Weighting based on Leaderboard scores (**51.6% XGBoost + 48.4% LightGBM**).

---

## 🛠️ Tech Stack

| Component           | Technology         | Purpose                                                |
| :------------------ | :----------------- | :----------------------------------------------------- |
| **Orchestration**   | Python Modules     | Modular & decoupled architecture (`src/`)              |
| **Data Processing** | Pandas / NumPy     | High-performance ETL on CPU                            |
| **Modeling**        | XGBoost & LightGBM | Gradient Boosting Machines                             |
| **Tracking**        | MLflow             | Experiment logging, metrics, and artifact storage      |
| **Inference**       | Batch Processing   | Chunk-based prediction engine to avoid memory overflow |

---

## 📊 Performance Results

The pipeline was rigorously tested and optimized. The ensemble model provided the most stable results across unknown data (Private LB).

| Model Strategy          | Public LB Score | Private LB Score | Insight                                                 |
| :---------------------- | :-------------- | :--------------- | :------------------------------------------------------ |
| **XGBoost (Single)**    | **1.129**       | 1.458            | High performance, slightly overfit to public test set.  |
| **LightGBM (Single)**   | 1.206           | 1.397            | Better generalization, lower variance.                  |
| **Ensemble (Weighted)** | 1.199           | **1.382** 🏆     | **Best Private Score.** Optimal balance for production. |

---

## 📂 Project Structure

```bash
energy-forecasting-mlops/
├── .github/workflows/    # CI/CD Pipelines (Tests)
├── data/                 # Raw and processed data
├── mlruns/               # MLflow experiment logs
├── src/                  # Source code modules
│   ├── data_loader.py    # Data ingestion logic
│   ├── preprocessing.py  # Feature engineering & Cleaning
│   ├── train.py          # XGBoost training pipeline
│   ├── train_lgbm.py     # LightGBM training pipeline
│   ├── inference.py      # Shared inference logic
│   └── utils.py          # Hardware compatibility utils
├── notebooks/            # EDA and Analysis notebooks
├── tests/                # Unit tests for pipeline integrity
├── predict_xgb.py        # Inference script (XGBoost Model)
├── predict_lgbm.py       # Inference script (LightGBM Model)
├── ensemble.py           # Meta-model inference script
└── requirements.txt      # Project dependencies
```

🔧 How to Run

1. Environment Setup

```bash
conda create -n mlops-energy python=3.10
conda activate mlops-energy
pip install -r requirements.txt
```

2. Training (with MLflow Tracking)
   Train both models to generate artifacts in mlruns/.

```bash
# Train XGBoost (GPU)
python -m src.train_xgb

# Train LightGBM (CPU/GPU)
python -m src.train_lgbm
```

3. Ensemble Inference
   Generate the final submission file using the weighted average of the trained models. (Note: Update Run IDs in ensemble.py before running)

```bash
python ensemble.py
```

### 🐳 Running with Docker (Recommended)

To ensure reproducibility across different OS and hardware, you can run the entire pipeline within a Docker container.

**1. Build the Image:**

```bash
docker build -t energy-predictor:v1 .

👨‍💻 Author
Ozan Bozyel Biomedical Engineer & Deep Learning Practitioner

Passionate about building scalable AI systems, bridging the gap between research models and production engineering. Experienced in MLOps, Medical AI, and Resource-Constrained ML.
```
### 📫 Connect with Me

[![LinkedIn](https://img.shields.io/badge/linkedin-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ozan-bozyel)
[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:ozanbozyel.job@gmail.com)
