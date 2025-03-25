# Scikit-Learn (sklearn) Cheat Sheet

## 1. ✅ Import Convention

```python
python
CopyEdit
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

```

---

## 2. 🧩 Core Functions / Classes / Concepts

| Name | Example Usage | Short Description |
| --- | --- | --- |
| `train_test_split` | `train_test_split(X, y, test_size=0.2)` | Splits dataset into training and test sets |
| `LinearRegression()` | `model = LinearRegression()` | Linear model for regression |
| `LogisticRegression()` | `model = LogisticRegression()` | Logistic model for binary classification |
| `fit()` | `model.fit(X_train, y_train)` | Train the model on data |
| `predict()` | `model.predict(X_test)` | Predict using trained model |
| `accuracy_score()` | `accuracy_score(y_test, y_pred)` | Evaluates classification accuracy |
| `confusion_matrix()` | `confusion_matrix(y_test, y_pred)` | Shows TP, TN, FP, FN breakdown |
| `StandardScaler()` | `scaler = StandardScaler()` | Scales features to mean=0 and std=1 |
| `Pipeline()` | `Pipeline([("scaler", StandardScaler()), ...])` | Sequentially apply transformers and estimators |
| `cross_val_score()` | `cross_val_score(model, X, y, cv=5)` | Evaluate model with cross-validation |
| `GridSearchCV()` | `GridSearchCV(model, param_grid, cv=5)` | Search best hyperparameters |
| `classification_report()` | `classification_report(y_test, y_pred)` | Detailed precision/recall/F1 report |
| `RandomForestClassifier()` | `RandomForestClassifier(n_estimators=100)` | Ensemble method using decision trees |
| `KNeighborsClassifier()` | `KNeighborsClassifier(n_neighbors=3)` | k-NN classification algorithm |
| `make_classification()` (utils) | `make_classification(n_samples=1000)` | Generate synthetic classification data |

---

## 3. ⚙️ Common Operations & Their Usage

### 🔄 Train/Test Split

```python
python
CopyEdit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

```

### 🔍 Standardizing Features

```python
python
CopyEdit
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

```

### 🧠 Model Training

```python
python
CopyEdit
model = LogisticRegression()
model.fit(X_train, y_train)

```

### 📈 Prediction & Evaluation

```python
python
CopyEdit
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

```

### 🔁 Cross-Validation

```python
python
CopyEdit
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
print("Cross-validation scores:", scores)

```

### 🔧 Grid Search for Hyperparameter Tuning

```python
python
CopyEdit
from sklearn.model_selection import GridSearchCV
params = {'C': [0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(), params, cv=5)
grid.fit(X_train, y_train)
print("Best Params:", grid.best_params_)

```

### 📊 Classification Report

```python
python
CopyEdit
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

```

---

## 4. 💡 Useful Tips / Best Practices

- **Always scale features** for algorithms sensitive to magnitude (e.g., SVM, Logistic Regression, k-NN).
- **Use pipelines** to bundle preprocessing + model:
    
    ```python
    python
    CopyEdit
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression())
    ])
    pipeline.fit(X_train, y_train)
    
    ```
    
- **Start simple** with models like LogisticRegression or DecisionTree before using complex ones.
- **Check for class imbalance** before training (e.g., use `value_counts()` in Pandas).
- **Use `RandomState`** for reproducibility when splitting data or training models.
- **Interpretability**: Use `.coef_` or `feature_importances_` to understand feature impact.

---

## 5. 🔗 Integration with NumPy, Pandas, PyTorch

- Scikit-Learn integrates natively with **NumPy** and **Pandas**:
    - Accepts NumPy arrays and Pandas DataFrames as inputs.
    - Outputs can be directly converted to NumPy or Pandas objects.
- To use it with **PyTorch**:
    - Use Scikit-Learn for preprocessing and PyTorch for modeling:
        
        ```python
        python
        CopyEdit
        X_scaled = StandardScaler().fit_transform(X_np)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        ```
        

---

## 6. 🛠 Mini Project / Small Exercises

### 💼 Tasks to Practice

1. **Train a Linear Regression model** on `sklearn.datasets.load_boston()` (or similar).
2. **Train a classifier** using `make_classification()` and evaluate it with accuracy and confusion matrix.
3. **Build a pipeline** to standardize data and use Logistic Regression on breast cancer dataset.
4. **Tune hyperparameters** of a k-NN model using GridSearchCV.
5. **Train a RandomForestClassifier** and plot feature importances.
6. **Compare models** (Logistic Regression vs. Decision Tree) on the same dataset using cross_val_score.
7. **Visualize decision boundaries** using `matplotlib` for a 2D synthetic dataset.
8. **Perform multi-class classification** using `LogisticRegression(multi_class='multinomial')`.
9. **Use OneHotEncoder** on categorical variables and train a model.
10. **Save and load model** using `joblib`:
    
    ```python
    python
    CopyEdit
    import joblib
    joblib.dump(model, "model.pkl")
    model = joblib.load("model.pkl")
    
    ```
