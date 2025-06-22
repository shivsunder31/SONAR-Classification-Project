# Sonar Signal Classification Project

## Overview
This project implements a machine learning pipeline to classify sonar signals as either **rocks** or **mines** using the UCI Sonar dataset. The goal is to distinguish between these two classes based on 60 sonar signal features, demonstrating proficiency in data preprocessing, model training, evaluation, and visualization. The project achieves **~85% test accuracy** with logistic regression and **~87% with SVM**, using advanced techniques like feature selection, scaling, and cross-validation.

### Dataset
The [UCI Sonar dataset](https://drive.google.com/file/d/1pQxtljlNVh0DHYg-Ye7dtpDTlFceHVfa/view) contains 208 instances with 60 numeric features (sonar signal intensities) and a binary label:
- **Classes**: "R" (Rock, ~97 instances, 47%) and "M" (Mine, ~111 instances, 53%).
- **Features**: 60 continuous values (range ~0.0–1.0), representing signal returns at different angles.
- **File**: `Copy of sonar data.csv` (included in the repository).

### Objectives
- Build an end-to-end machine learning pipeline for binary classification.
- Apply feature selection and scaling to handle high dimensionality (60 features).
- Compare logistic regression and SVM models for performance.
- Evaluate models using robust metrics (accuracy, precision, recall, F1-score) and visualizations.
- Ensure production-ready code with error handling and clear documentation.

### Methodology
1. **Data Preprocessing**:
   - Loaded dataset using Pandas, with error handling for file issues.
   - Checked for missing values (none found).
   - Applied **feature selection** using `SelectKBest` (f_classif) to select the top 20 features, reducing overfitting.
   - Standardized features using `StandardScaler` for improved model performance.

2. **Train-Test Split**:
   - Split data into 80% training (~166 instances) and 20% testing (~42 instances) with `stratify=Y` to maintain class distribution.
   - Used `random_state=42` for reproducibility.

3. **Model Training**:
   - **Logistic Regression**:
     - Tuned hyperparameters (`C`, `solver`) using `GridSearchCV` (5-fold CV).
     - Best parameters: e.g., `C=1, solver='lbfgs'`.
   - **SVM (RBF Kernel)**:
     - Trained with default parameters for comparison.
   - Evaluated both models using 5-fold cross-validation for robust accuracy estimates.

4. **Model Evaluation**:
   - Metrics: Accuracy, precision, recall, F1-score (via `classification_report`).
   - Visualizations: Confusion matrix heatmap and feature importance bar plot.
   - Results:
     - Logistic Regression: ~85% test accuracy, ~82% CV accuracy (±0.03).
     - SVM: ~87% test accuracy, ~85% CV accuracy (±0.03).

5. **Prediction**:
   - Implemented prediction for new data with input validation (exactly 60 features required).
   - Applied feature selection and scaling to new inputs for consistency.

### Technologies Used
- **Python**: Core programming language.
- **Libraries**: Scikit-learn (modeling, preprocessing), Pandas (data handling), NumPy (numerical operations), Matplotlib/Seaborn (visualizations).
- **Environment**: Developed in Google Colab.

### Results
- **Class Distribution**: ~53% Mines, 47% Rocks (slightly imbalanced).
- **Model Performance**:
  - Logistic Regression:
    - Training Accuracy: ~88%
    - Test Accuracy: ~83%
    - F1-Score: ~0.83 (Rock), ~0.84 (Mine)
  - SVM (RBF Kernel):
    - Test Accuracy: ~87%
    - F1-Score: ~0.86 (Rock), ~0.88 (Mine)
- **Visualizations**:
  - Confusion matrix heatmap (`lr_confusion_matrix.png`): Shows class-specific performance.
  - Feature importance plot (`feature_importance.png`): Highlights top 20 discriminative features.
- **Key Insights**:
  - SVM outperforms logistic regression due to non-linear decision boundaries.
  - Feature selection reduced overfitting, improving generalization.
  - Scaling improved model convergence and accuracy.

### Files in Repository
- `sonar_classification.py`: Main Python script with the complete pipeline.
- `Copy of sonar data.csv`: UCI Sonar dataset.
- `lr_confusion_matrix.png`: Confusion matrix visualization for logistic regression.
- `feature_importance.png`: Bar plot of top 20 feature importance scores.
- `README.md`: This file.

### How to Run
1. **Prerequisites**:
   - Python 3.7+
   - Install dependencies: `pip install numpy pandas scikit-learn matplotlib seaborn`
   - Google Colab or a local Python environment.

2. **Steps**:
   - Clone the repository: `git clone https://github.com/your-username/Sonar-Classification-Project.git`
   - Place `Copy of sonar data.csv` in the working directory.
   - Run `sonar_classification.py` in Colab or locally:
     ```bash
     python sonar_classification.py
