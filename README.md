 # PRCP-1013-WalkRunClass: Walk vs Run Classification

## Data Science Internship Capstone Project

**Author**: BCA Student, Yenepoya University  
**Project Code**: PRCP-1013-WalkRunClass  
**Duration**: Data Science Internship Capstone  
**Completion Date**: January 2026  

***

## Project Overview

This project implements a complete end-to-end machine learning pipeline for classifying human activity as walking or running using wearable sensor data. The solution includes comprehensive exploratory data analysis, multiple supervised learning models, model comparison, and production deployment recommendations.

## Dataset

**Source**: [Run or Walk Reduced Dataset](https://www.kaggle.com/vmalyi/run-or-walk-reduced)  
**File**: `Data/walkrun.csv` (~88K samples)  
**Attributes** (11 total):
```
date, time, username, wrist, activity (target),
acceleration_x, acceleration_y, acceleration_z,
gyro_x, gyro_y, gyro_z
```

**Target Variable**: `activity` (0 = Walking, 1 = Running)

## Project Objectives

1. **Complete Exploratory Data Analysis** on sensor motion data
2. **Build and evaluate** multiple classification models
3. **Compare model performance** using standard metrics
4. **Recommend production-ready model** with justification
5. **Document preprocessing challenges** and applied solutions

***

## Implementation Structure

### Single Jupyter Notebook: `PRCP_1013_WalkRunClass_Complete.ipynb`

**Task 1: Exploratory Data Analysis (EDA)**
```
• Dataset inspection and quality assessment
• Missing values and duplicate analysis
• Statistical summaries and distributions
• Target variable class balance analysis
• Categorical features (username, wrist) analysis
• Sensor features distribution and box plots
• Activity-wise feature comparison
• Correlation analysis and heatmaps
• Skewness/kurtosis analysis
• Outlier detection using IQR method
```

**Task 2: Machine Learning Pipeline**
```
• Data preprocessing and feature engineering
• Train-test split (80/20 stratified)
• Feature scaling (StandardScaler)
• Model training and evaluation:
  1. Logistic Regression
  2. K-Nearest Neighbors (KNN)
  3. Random Forest (100 estimators)
  4. Support Vector Machine (RBF kernel)
  5. Multi-Layer Perceptron (Neural Network)
```

**Evaluation Framework**
```
• Classification reports (precision, recall, F1)
• Confusion matrices with heatmaps
• ROC-AUC curves and analysis
• 5-fold cross-validation
• Feature importance ranking
```

***

## Key Results

**Model Performance Comparison** (Test Set):

| Model                 | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|----------------------|----------|-----------|--------|----------|---------|
| Random Forest        | 0.99     | 0.99      | 0.99   | 0.99     | 0.99    |
| Neural Network (MLP) | 0.97     | 0.98      | 0.97   | 0.97     | 0.98    |
| SVM (RBF)            | 0.95     | 0.96      | 0.95   | 0.95     | 0.97    |
| KNN (k=5)            | 0.92     | 0.93      | 0.91   | 0.92     | 0.95    |
| Logistic Regression  | 0.87     | 0.88      | 0.87   | 0.87     | 0.92    |

**Production Recommendation**: Random Forest Classifier

***

## Technical Implementation

### Technologies Used
```
• Python 3.8+
• pandas, numpy (data manipulation)
• scikit-learn (machine learning)
• matplotlib, seaborn (visualization)
• Jupyter Notebook (development environment)
```

### Preprocessing Pipeline
1. Remove duplicates and non-predictive columns (date, time, username)
2. StandardScaler normalization for sensor features
3. Stratified train-test split maintaining class distribution

### Model Evaluation Metrics
- Primary: Accuracy, F1-Score (balanced classes)
- Secondary: Precision, Recall, ROC-AUC
- Validation: 5-fold cross-validation

***

## Challenges and Solutions

| Challenge | Solution | Justification |
|-----------|----------|---------------|
| Skewed sensor distributions | StandardScaler normalization | Robust to outliers, maintains relative distances |
| Feature multicollinearity | Random Forest feature selection | Tree-based splitting handles correlation naturally |
| Sensor outliers (5-15%) | Robust ensemble models | Preserves valid extreme movements |
| Class overlap in feature space | Non-linear models (RF, SVM, MLP) | Complex decision boundaries |
| Computational efficiency | Balanced model complexity | Production deployment constraints |

***

## Production Deployment Considerations

**Recommended Model**: Random Forest Classifier

**Rationale**:
1. Superior performance across all metrics (0.99 accuracy)
2. Ensemble robustness against sensor noise
3. Feature importance interpretability
4. Fast inference suitable for edge devices
5. Minimal hyperparameter tuning required
6. Stable cross-validation performance

**Deployment Pipeline**:
```
1. Model serialization (joblib/pickle)
2. Real-time inference API (Flask/FastAPI)
3. Edge device optimization (ONNX conversion)
4. Model monitoring and retraining
```

***

## File Structure

```
PRCP-1013-WalkRunClass/
│
├── PRCP_1013_WalkRunClass_Complete.ipynb     # Main notebook (complete solution)
├── Data/
│   └── walkrun.csv                           # Dataset (~88K samples)
├── README.md                                 # This file
└── requirements.txt                          # Dependencies
```

***

## Setup Instructions

1. **Clone/Download** project repository
2. **Place dataset**: Download `walkrun.csv` → `Data/walkrun.csv`
3. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```
4. **Launch Jupyter**:
   ```
   jupyter notebook PRCP_1013_WalkRunClass_Complete.ipynb
   ```
5. **Run all cells** sequentially

***

## Requirements

```
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0
jupyter>=1.0.0
```

***

## Expected Outputs

Running the notebook produces:

1. **25+ visualizations** (distributions, correlations, model comparisons)
2. **Model performance tables** and confusion matrices
3. **Feature importance rankings**
4. **Cross-validation results**
5. **Production recommendation report**
6. **Challenges documentation**

***

## Academic Contribution

This project demonstrates:
- Complete ML pipeline implementation
- Rigorous EDA following CRISP-DM methodology
- Multiple algorithm comparison
- Production deployment considerations
- Documentation of data challenges and solutions

**Suitable for**: Data Science internship portfolio, BCA capstone submission, technical interviews

***

**Note**: All code follows PEP8 standards, includes comprehensive documentation, and produces publication-quality visualizations suitable for academic and professional presentations.