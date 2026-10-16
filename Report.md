# Disease Prediction from Medical Data
## Internship Experiment Report

### Objective
This project evaluates the performance of four machine learning classifiers on structured medical datasets:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

The goal is to compare their effectiveness for binary disease classification using standard clinical-style tabular data.

### Datasets Used
- Breast Cancer Wisconsin Diagnostic: binary classification of benign vs. malignant tumors
- UCI Cleveland Heart Disease: binary classification of disease presence vs. absence
- Pima Indians Diabetes: binary classification of diabetes outcome

### Methodology
- Stratified 80:20 train-test split with `random_state=42`
- Median imputation and feature scaling were performed inside pipeline preprocessing
- Five-fold stratified cross-validation on the training set
- Evaluation metrics: Accuracy, Precision, Recall, F1-score, and ROC-AUC

### Comparative Results

#### 1. Breast Cancer Wisconsin Diagnostic

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | Training Time (s) | CV Accuracy | CV ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9750 | 0.9286 | 0.9512 | 0.9960 | 0.2187 | 0.9736 | 0.9958 |
| SVM | 0.9737 | 1.0000 | 0.9286 | 0.9630 | 0.9947 | 0.1893 | 0.9714 | 0.9949 |
| Random Forest | 0.9737 | 1.0000 | 0.9286 | 0.9630 | 0.9944 | 2.5063 | 0.9604 | 0.9880 |
| XGBoost | 0.9649 | 1.0000 | 0.9048 | 0.9500 | 0.9940 | 0.6189 | 0.9670 | 0.9938 |

#### 2. Heart Disease

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | Training Time (s) | CV Accuracy | CV ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8689 | 0.8125 | 0.9286 | 0.8667 | 0.9513 | 0.0046 | 0.8264 | 0.8960 |
| SVM | 0.8525 | 0.8065 | 0.8929 | 0.8475 | 0.9437 | 0.0094 | 0.8223 | 0.8832 |
| Random Forest | 0.9016 | 0.8438 | 0.9643 | 0.9000 | 0.9551 | 0.2457 | 0.8139 | 0.8837 |
| XGBoost | 0.8852 | 0.8387 | 0.9286 | 0.8814 | 0.9383 | 0.0255 | 0.8058 | 0.8680 |

#### 3. Diabetes

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | Training Time (s) | CV Accuracy | CV ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7078 | 0.6000 | 0.5000 | 0.5455 | 0.8130 | 0.0046 | 0.7882 | 0.8434 |
| SVM | 0.7403 | 0.6522 | 0.5556 | 0.6000 | 0.7964 | 0.0356 | 0.7801 | 0.8333 |
| Random Forest | 0.7468 | 0.6596 | 0.5741 | 0.6139 | 0.8165 | 0.2622 | 0.7736 | 0.8195 |
| XGBoost | 0.7532 | 0.6739 | 0.5741 | 0.6200 | 0.8250 | 0.0251 | 0.7589 | 0.8227 |

### Key Observations
- The breast cancer dataset showed the strongest classification performance across models.
- Random Forest and XGBoost were often competitive for heart disease and diabetes tasks, with XGBoost achieving the highest test accuracy in the diabetes dataset.
- Model performance varied noticeably by dataset, indicating that no single algorithm is uniformly best across all clinical prediction tasks.
- Training times were low for logistic regression and SVM in most cases, while tree-based models were slower but still manageable for the evaluated datasets.

### Limitations
This report reflects educational experiments conducted on public benchmark datasets. The results are not sufficient to establish clinical validity or suitability for patient diagnosis. Independent external validation and domain expert review are necessary before any clinical deployment.

### Model Artifact Note
Saved pipelines were refitted on each complete dataset after evaluation. These fitted parameters are retained as reproducibility artifacts and are not the held-out evaluation models used to generate the reported test metrics.
