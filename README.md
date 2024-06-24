
---

# Chronic Kidney Disease Prediction Model

This repository contains a Jupyter Notebook for building and evaluating a machine learning model to predict chronic kidney disease based on medical attributes. The model uses multiple classifiers and combines their predictions to provide a final diagnosis.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Model Building](#model-building)
- [Model Evaluation](#model-evaluation)
- [Saving the Model](#saving-the-model)
- [Contributing](#contributing)
- [License](#license)

## Overview

The notebook builds and evaluates several machine learning models to predict chronic kidney disease. The steps include:

1. Loading and exploring the dataset.
2. Data preprocessing.
3. Splitting the data into training and testing sets.
4. Building multiple classifiers.
5. Evaluating the models using various metrics.
6. Saving the best model.

## Dataset

The dataset used in this notebook consists of medical attributes related to chronic kidney disease. Each row represents a patient with various attributes and the diagnosis result. The dataset can be found [here](https://www.kaggle.com/datasets/mansoordaku/ckdisease).

### Dataset Description

The dataset contains the following columns:

- **Bp**: Blood pressure
- **Sg**: Specific gravity
- **Al**: Albumin
- **Su**: Sugar
- **Rbc**: Red blood cells
- **Bu**: Blood urea
- **Sc**: Serum creatinine
- **Sod**: Sodium
- **Pot**: Potassium
- **Hemo**: Hemoglobin
- **Wbcc**: White blood cell count
- **Rbcc**: Red blood cell count
- **Htn**: Hypertension
- **Class**: Diagnosis of chronic kidney disease (1 = yes, 0 = no)

## Installation

To run this notebook, you need to have Python installed along with the following packages:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install these packages using pip:

```sh
pip install pandas numpy matplotlib seaborn scikit-learn
```

Alternatively, you can use the provided `requirements.txt` file:

```sh
pip install -r requirements.txt
```

## Usage

1. Clone this repository:

```sh
git clone https://github.com/Sahiru2007/Chronic-Kidney-Disease-preidction-Model.git
cd Chronic-Kidney-Disease-preidction-Model
```

2. Open the Jupyter Notebook:

```sh
jupyter notebook Chronic_Kidney_Disease.ipynb
```

3. Run all cells in the notebook to see the complete analysis and model evaluation.

## Data Preprocessing

### Handling Missing Values

Missing values in the dataset are handled by dropping duplicate values and normalizing the data:

```python
data = data.drop_duplicates()

for x in data.columns:
    data[x] = (data[x] - data[x].min()) / (data[x].max() - data[x].min())
```

### Splitting the Data

The dataset is split into training and testing sets using an 80-20 split:

```python
from sklearn.model_selection import train_test_split

y = data['Class']
X = data.drop(['Class'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
```

## Model Building

### Models Used

The notebook evaluates a Support Vector Classifier (SVC) model:

- **Support Vector Machine (SVM)**

### Training the Model

Example: Training a Support Vector Classifier

```python
from sklearn.svm import SVC

svc_model = SVC(kernel='linear')
svc_model.fit(X_train, y_train)
predictions = svc_model.predict(X_test)
```

## Model Evaluation

### Metrics

The model is evaluated using the following metrics:

- **Accuracy**: The ratio of correctly predicted instances to the total instances.
- **Confusion Matrix**: A summary of prediction results on a classification problem.

### Example: Evaluating Support Vector Classifier

```python
from sklearn.metrics import accuracy_score, confusion_matrix

train_predictions = svc_model.predict(X_train)
test_predictions = svc_model.predict(X_test)

train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

train_cm = confusion_matrix(y_train, train_predictions)
test_cm = confusion_matrix(y_test, test_predictions)

print(f'Training Accuracy: {train_accuracy}')
print(f'Testing Accuracy: {test_accuracy}')
print(f'Training Confusion Matrix:\n{train_cm}')
print(f'Testing Confusion Matrix:\n{test_cm}')
```

### Evaluation Results

- **Support Vector Classifier**: 
  - Training Accuracy ~ 100%
  - Testing Accuracy ~ 98.75%

## Unique Aspects of the Notebook

- **Correlation Matrix**: A heatmap to visualize the correlations between features and the target variable.
- **Normalization**: Feature scaling to bring all features to a similar scale.
- **Stratified Split**: Ensuring the class distribution remains the same in both training and testing sets.

### Correlation Matrix

```python
import seaborn as sns
import matplotlib.pyplot as plt

corr = data.corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr, annot=True, fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
```

## Saving the Model

The best-performing model is saved using the `pickle` module for future use:

```python
import pickle

filename = 'cronic_kidney.pkl'
with open(filename, 'wb') as file:
    pickle.dump(svc_model, file)

print(f"Model saved to {filename}")
```

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

---

