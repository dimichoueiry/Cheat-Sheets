# Numpy Cheat Sheet

---

# NumPy Daily Use Cheatsheet

> Import NumPy:
> 
> 
> Always start by importing NumPy (usually as `np`):
> 
> ```python
> import numpy as np
> 
> ```
> 

## 1. Creating Arrays

- **From Python Lists:**
    
    ```python
    arr = np.array([1, 2, 3, 4])
    
    ```
    
- **Zeros and Ones:**
    
    ```python
    zeros = np.zeros((3, 4))       # 3x4 array of zeros
    ones = np.ones((2, 5))         # 2x5 array of ones
    
    ```
    
- **Range and Linspace:**
    
    ```python
    ar = np.arange(0, 10, 2)       # [0, 2, 4, 6, 8]
    lin = np.linspace(0, 1, 5)       # 5 evenly spaced numbers between 0 and 1
    
    ```
    
- **Identity Matrix:**
    
    ```python
    eye = np.eye(4)                # 4x4 identity matrix
    
    ```
    

---

## 2. Inspecting Arrays

- **Shape, Dimension, and Size:**
    
    ```python
    arr.shape     # returns tuple of dimensions, e.g., (4,)
    arr.ndim      # returns number of dimensions, e.g., 1
    arr.size      # total number of elements
    arr.dtype     # data type of elements
    
    ```
    

---

## 3. Basic Operations and Broadcasting

- **Element-wise Arithmetic:**
    
    ```python
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    sum_arr = a + b             # [5, 7, 9]
    product = a * b             # [4, 10, 18]
    
    ```
    
- **Broadcasting:**
NumPy automatically applies arithmetic on arrays of different shapes if compatible.
    
    ```python
    a = np.array([[1, 2, 3]])
    b = np.array([10, 20, 30])
    result = a + b              # Broadcasts b across a's rows
    
    ```
    

---

## 4. Slicing and Indexing

- **Basic Slicing:**
    
    ```python
    sub = arr[1:3]              # elements from index 1 to 2
    
    ```
    
- **Multidimensional Slicing:**
    
    ```python
    matrix = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])
    sub_matrix = matrix[0:2, 1:3]  # rows 0-1, cols 1-2
    
    ```
    
- **Boolean Indexing:**
    
    ```python
    mask = matrix > 5
    filtered = matrix[mask]       # returns elements greater than 5
    
    ```
    

---

## 5. Aggregation Functions

- **Sum, Mean, Standard Deviation:**
    
    ```python
    total = np.sum(matrix)           # sum of all elements
    col_mean = np.mean(matrix, axis=0) # mean along columns
    row_std = np.std(matrix, axis=1)   # std deviation along rows
    
    ```
    
- **Min, Max, Median, Percentile:**
    
    ```python
    minimum = np.min(matrix)
    maximum = np.max(matrix)
    median = np.median(matrix)
    percentile_90 = np.percentile(matrix, 90)
    
    ```
    

---

## 6. Reshaping and Transposing

- **Reshape:**
    
    ```python
    arr = np.arange(12)
    reshaped = arr.reshape((3, 4))   # convert 1D to 3x4 2D array
    
    ```
    
- **Flattening:**
    
    ```python
    flat = reshaped.flatten()          # returns a copy as a 1D array
    view = reshaped.ravel()            # returns a flattened view if possible
    
    ```
    
- **Transpose:**
    
    ```python
    transposed = reshaped.T           # or np.transpose(reshaped)
    
    ```
    

---

## 7. Stacking and Splitting Arrays

- **Concatenation:**
    
    ```python
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    combined = np.concatenate((a, b), axis=0)  # vertical concatenation
    
    ```
    
- **Vertical & Horizontal Stacking:**
    
    ```python
    v_stack = np.vstack((a, b))   # stacks arrays vertically
    h_stack = np.hstack((a, b))   # stacks arrays horizontally
    
    ```
    
- **Splitting:**
    
    ```python
    split_arrays = np.split(combined, 2, axis=0)  # split into 2 arrays along rows
    
    ```
    

---

## 8. Sorting and Searching

- **Sort and Argsort:**
    
    ```python
    sorted_arr = np.sort(arr)
    sort_indices = np.argsort(arr)
    
    ```
    
- **Where:**
    
    ```python
    indices = np.where(arr > 5)    # returns indices where condition is True
    
    ```
    

---

## 9. Linear Algebra

- **Dot Product and Matrix Multiplication:**
    
    ```python
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    dot_product = np.dot(a, b)
    matrix_mult = np.matmul(a, b)   # equivalent to a @ b
    
    ```
    
- **Other Linalg Functions:**
    
    ```python
    inv = np.linalg.inv(a)          # matrix inverse
    det = np.linalg.det(a)          # determinant
    eig_vals, eig_vecs = np.linalg.eig(a)  # eigenvalues and eigenvectors
    
    ```
    

---

## 10. Random Number Generation

- **Random Arrays:**
    
    ```python
    rand_arr = np.random.rand(3, 4)       # uniform random numbers in [0,1)
    randn_arr = np.random.randn(3, 4)       # standard normal distribution
    rand_int = np.random.randint(0, 10, size=(2, 3))  # random integers between 0 and 10
    
    ```
    
- **Random Choice:**
    
    ```python
    sample = np.random.choice([10, 20, 30], size=5, replace=True)
    
    ```
    

---

## 11. Other Useful Functions

- **Unique Values:**
    
    ```python
    unique_vals = np.unique(arr)
    
    ```
    
- **Clip and Round:**
    
    ```python
    clipped = np.clip(arr, 0, 5)      # limits values between 0 and 5
    rounded = np.round(arr, decimals=2)  # round to 2 decimal places
    
    ```
    
- **Cumulative Sum and Differences:**
    
    ```python
    cum_sum = np.cumsum(arr)
    diff = np.diff(arr)
    
    ```
    
- **Absolute Value:**
    
    ```python
    abs_arr = np.abs(arr)
    
    ```
    

---

## Summary

- **Array Creation:** Use functions like `np.array`, `np.zeros`, `np.ones`, `np.arange`, and `np.linspace` for building arrays.
- **Inspection:** Check dimensions, shape, size, and type.
- **Arithmetic & Broadcasting:** Perform element-wise operations and leverage broadcasting for mixed shapes.
- **Slicing & Indexing:** Access, modify, and filter arrays using slices and boolean indexing.
- **Aggregation & Statistics:** Quickly compute sums, means, standard deviations, and other statistics along desired axes.
- **Reshaping & Transposing:** Change array dimensions with `reshape`, `flatten`, and transpose using `.T`.
- **Stacking & Splitting:** Combine and partition arrays with functions like `concatenate`, `vstack`, `hstack`, and `split`.
- **Linear Algebra:** Utilize `np.dot`, `np.matmul`, and `np.linalg` for matrix operations.
- **Random Generation:** Create random arrays using `np.random` functions.
- **Miscellaneous Tools:** Use `np.sort`, `np.where`, `np.unique`, and more for everyday tasks.

# **🟢NumPy Cheat Sheet – Essentials for ML/DL/DS**

### 📌 **Import Convention**

```python

import numpy as np
```

---

### 📊 **Array Creation**

| Function | Example | Description |
| --- | --- | --- |
| `np.array()` | `np.array([1, 2, 3])` | Create array from list/tuple |
| `np.zeros(shape)` | `np.zeros((3, 4))` | Array of zeros |
| `np.ones(shape)` | `np.ones((2, 3))` | Array of ones |
| `np.full(shape, fill_value)` | `np.full((2,2), 5)` | Filled with a constant value |
| `np.eye(N)` | `np.eye(3)` | Identity matrix |
| `np.arange(start, stop, step)` | `np.arange(0, 10, 2)` | Range of values |
| `np.linspace(start, stop, num)` | `np.linspace(0, 1, 5)` | Evenly spaced numbers over interval |
| `np.random.rand(shape)` | `np.random.rand(2, 3)` | Random floats [0, 1), uniform |
| `np.random.randn(shape)` | `np.random.randn(3, 3)` | Random samples from standard normal (Gaussian) |
| `np.random.randint(low, high, shape)` | `np.random.randint(0, 10, (3,3))` | Random integers |
| `np.random.seed(seed)` | `np.random.seed(42)` | Fix randomness for reproducibility |

---

### 🔄 **Array Reshaping & Manipulation**

| Function | Example | Description |
| --- | --- | --- |
| `reshape(new_shape)` | `a.reshape(2, 3)` | Change shape without changing data |
| `ravel()` / `flatten()` | `a.ravel()` | Flatten array (1D view) |
| `transpose()` / `.T` | `a.T` | Transpose (swap rows and columns) |
| `concatenate()` | `np.concatenate([a, b], axis=0)` | Concatenate arrays |
| `stack()` | `np.stack([a, b], axis=1)` | Stack arrays along new axis |
| `split()` | `np.split(a, 3, axis=0)` | Split array |
| `expand_dims()` | `np.expand_dims(a, axis=0)` | Add a new dimension |
| `squeeze()` | `np.squeeze(a)` | Remove single-dimensional entries |

---

### 🔢 **Basic Operations**

| Operation | Example | Description |
| --- | --- | --- |
| `+`, `-`, `*`, `/` | `a + b` | Element-wise addition, subtraction, etc. |
| `np.dot(a, b)` | `np.dot(a, b)` | Matrix multiplication |
| `@` operator | `a @ b` | Matrix multiplication (shorthand) |
| `np.sum(a, axis)` | `np.sum(a, axis=0)` | Sum elements |
| `np.mean(a, axis)` | `np.mean(a, axis=1)` | Mean |
| `np.std(a)` | `np.std(a)` | Standard deviation |
| `np.max(a, axis)` / `np.min()` | `np.max(a, axis=0)` | Max / Min |
| `np.argmax(a)` / `np.argmin()` | `np.argmax(a)` | Index of max / min |
| `np.cumsum(a)` | `np.cumsum(a)` | Cumulative sum |

---

### 🎯 **Indexing & Slicing**

| Syntax | Example | Description |
| --- | --- | --- |
| Standard slicing | `a[1:3, :]` | Rows 1 to 2, all columns |
| Boolean indexing | `a[a > 0]` | Filter elements |
| Fancy indexing | `a[[0, 2], [1, 3]]` | Specific row/col combinations |

---

### 📏 **Linear Algebra (important for ML & DL!)**

| Function | Example | Description |
| --- | --- | --- |
| `np.dot(a, b)` | `np.dot(a, b)` | Dot product |
| `np.matmul(a, b)` | `np.matmul(a, b)` | Matrix product |
| `np.linalg.inv(a)` | `np.linalg.inv(a)` | Inverse of matrix |
| `np.linalg.det(a)` | `np.linalg.det(a)` | Determinant |
| `np.linalg.eig(a)` | `np.linalg.eig(a)` | Eigenvalues & eigenvectors |
| `np.linalg.svd(a)` | `np.linalg.svd(a)` | Singular Value Decomposition |

---

### 🚩 **Commonly Used Functions in ML Pipelines**

| Function | Example | Description |
| --- | --- | --- |
| `np.unique()` | `np.unique(a)` | Unique elements, often used for labels |
| `np.clip()` | `np.clip(a, min, max)` | Limit values within range |
| `np.where(condition, x, y)` | `np.where(a > 0, 1, -1)` | Conditional replacement |
| `np.isnan(a)` | `np.isnan(a)` | Detect NaN values |
| `np.isinf(a)` | `np.isinf(a)` | Detect infinity |

---

### 💡 **Pro Tip: Vectorization over Loops**

- Replace slow Python loops with vectorized NumPy operations for speed.

```python

# Instead of:
result = []
for x in a:
    result.append(x ** 2)

# Do:
result = a ** 2

```

- 🚀**Mini-Project: Titanic Passenger Data (NumPy Only)**
    
    *(Simulating a simplified dataset without Pandas!)*
    
    ### **Scenario:**
    
    You have basic passenger data stored in NumPy arrays. Your tasks:
    
    ---
    
    ### 🔽 **Dataset (Simulated):**
    
    ```python
    python
    CopyEdit
    import numpy as np
    
    # Passenger data: [PassengerID, Age, Fare, Survived]
    data = np.array([
        [1, 22, 7.25, 0],
        [2, 38, 71.28, 1],
        [3, 26, 7.92, 1],
        [4, 35, 53.1, 1],
        [5, 35, 8.05, 0],
        [6, 29, 8.46, 0],
        [7, 2, 21.07, 1],
        [8, 27, 11.13, 1],
        [9, 14, 30.07, 0],
        [10, 4, 16.7, 1],
    ])
    
    ```
    
    ---
    
    ### 📋 **Tasks:**
    
    1. **Get basic stats:**
        - Compute mean, median, min, and max of Age and Fare columns.
    2. **Normalize Fare column (Min-Max scaling).**normalized_fare=max−minfare−min
        
        normalized_fare=fare−minmax−min\text{normalized\_fare} = \frac{\text{fare} - \text{min}}{\text{max} - \text{min}}
        
    3. **Find passengers under age 18.**
    4. **Count how many survived (Survived column = 1).**
    5. **Create a new column: Fare per Age ratio.**
    6. **Find index of passenger who paid the highest fare.**
    7. **Filter passengers who paid more than the average fare and survived.**
    8. **Apply vectorized operation: Increase Fare by 10% if age > 30.**
    
    ---
    
    ### ✅ **BONUS (Linear Algebra Challenge):**
    
    Given the simplified feature matrix:
    
    ```python
    python
    CopyEdit
    X = data[:, 1:3]  # Age & Fare
    
    ```
    
    And a weight vector:
    
    ```python
    python
    CopyEdit
    weights = np.array([0.4, 0.6])  # Random weights
    
    ```
    
    **Task:**
    
    Compute the weighted sum (dot product) for each passenger:
    
    ```python
    python
    CopyEdit
    result = np.dot(X, weights)
    
    ```
    
    ---
    
    ### 🌟 **Optional Next Step:**
    
    Wrap the workflow into reusable NumPy functions (simulate a pre-ML preprocessing pipeline).
