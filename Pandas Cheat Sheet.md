# Pandas Cheat Sheet

# Pandas Cheatsheet for Data Science

This cheat sheet provides code snippets for common operations in pandas—from loading and previewing data to cleaning, transforming, and aggregating it for further analysis or visualization.

---

## 1. Loading Data from Files

- **Read CSV:**
    
    ```python
    python
    Copy
    import pandas as pd
    df = pd.read_csv('data.csv')
    
    ```
    
- **Read Excel:**
    
    ```python
    python
    Copy
    df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
    
    ```
    

---

## 2. Quick Data Preview

- **View First & Last Rows:**
    
    ```python
    python
    Copy
    df.head()  # First 5 rows
    df.tail()  # Last 5 rows
    
    ```
    

---

## 3. Understanding DataFrame Structure

- **Get DataFrame Info:**
    
    ```python
    python
    Copy
    df.info()
    
    ```
    
- **Summary Statistics:**
    
    ```python
    python
    Copy
    df.describe()
    
    ```
    

---

## 4. Indexing and Selection

- **Label-based Selection with `loc`:**
    
    ```python
    python
    Copy
    subset = df.loc[0:5, ['Column1', 'Column2']]
    
    ```
    
- **Position-based Selection with `iloc`:**
    
    ```python
    python
    Copy
    subset = df.iloc[0:5, 0:2]
    
    ```
    

---

## 5. Handling Missing Data

- **Drop Missing Values:**
    
    ```python
    python
    Copy
    df_clean = df.dropna()
    
    ```
    
- **Fill Missing Values:**
    
    ```python
    python
    Copy
    df_filled = df.fillna({'Column1': df['Column1'].mean(), 'Column2': 0})
    
    ```
    

---

## 6. Removing Duplicates

- **Drop Duplicate Rows:**
    
    ```python
    python
    Copy
    df_unique = df.drop_duplicates()
    
    ```
    

---

## 7. Grouping and Aggregation

- **Simple Grouping (Sum):**
    
    ```python
    python
    Copy
    grouped = df.groupby('Country').sum().reset_index()
    
    ```
    
- **Custom Aggregation with `.agg()`:**
    
    ```python
    python
    Copy
    grouped = df.groupby('Country').agg({
        'Amount': 'sum',
        'SalesPersona': lambda x: ', '.join(x.unique()),
        'Product': lambda x: ', '.join(x.unique())
    }).reset_index()
    
    ```
    
    *Note: The `.agg()` method lets you define different aggregation functions for each column.*
    

---

## 8. Merging and Joining DataFrames

- **Merge DataFrames:**
    
    ```python
    python
    Copy
    merged_df = pd.merge(df1, df2, on='id', how='inner')
    
    ```
    
- **Join DataFrames (Using Index):**
    
    ```python
    python
    Copy
    df1.set_index('id', inplace=True)
    df2.set_index('id', inplace=True)
    joined_df = df1.join(df2, how='inner')
    
    ```
    

---

## 9. Concatenating DataFrames

- **Vertical Concatenation:**
    
    ```python
    python
    Copy
    vertical_concat = pd.concat([df1, df2], axis=0)
    
    ```
    
- **Horizontal Concatenation:**
    
    ```python
    python
    Copy
    horizontal_concat = pd.concat([df1, df2], axis=1)
    
    ```
    

---

## 10. Sorting Data

- **Sort by Column Values:**
    
    ```python
    python
    Copy
    df_sorted = df.sort_values(by='Column1', ascending=False)
    
    ```
    

---

## 11. Managing DataFrame Index

- **Reset the Index:**
    
    ```python
    python
    Copy
    df_reset = df.reset_index(drop=True)
    
    ```
    
- **Set a Column as Index:**
    
    ```python
    python
    Copy
    df_indexed = df.set_index('Column1')
    
    ```
    

---

## 12. Applying Functions

- **Apply a Custom Function:**
    
    ```python
    python
    Copy
    df['new_column'] = df['Column1'].apply(lambda x: x * 2)
    
    ```
    

---

## 13. Data Type Conversion

- **Convert Data Types:**
    
    ```python
    python
    Copy
    df['Column1'] = df['Column1'].astype(float)
    
    ```
    

---

## 14. Handling Date/Time Data

- **Convert Strings to Datetime:**
    
    ```python
    python
    Copy
    df['date'] = pd.to_datetime(df['date'])
    
    ```
    

---

## 15. Creating Pivot Tables

- **Pivot Table for Aggregation:**
    
    ```python
    python
    Copy
    pivot = df.pivot_table(values='Sales', index='Region', columns='Month', aggfunc='sum')
    
    ```
    

---

## 16. Counting Unique Values

- **Value Counts:**
    
    ```python
    python
    Copy
    counts = df['CategoryColumn'].value_counts()
    
    ```
    

---

## 17. Random Sampling

- **Sample Rows Randomly:**
    
    ```python
    python
    Copy
    sample_df = df.sample(n=5)
    
    ```
    

---

## 18. Replacing Values (Preprocessing)

- **Replace Specific Values:**
    
    ```python
    python
    Copy
    # Cleans Amount by removing $ and commas, then converts to float
    df['Amount'] = df['Amount'].replace('[\$,]', '', regex=True).astype(float)
    
    ```
    
- **String Replacement in a Column:**
    
    ```python
    python
    Copy
    df['Column2'] = df['Column2'].str.replace('old', 'new')
    
    ```
    

---

## 19. Checking DataFrame Dimensions

- **Check Shape:**
    
    ```python
    python
    Copy
    print(df.shape)  # Outputs (rows, columns)
    
    ```
    

---

## Putting It All Together: Sample Workflow

```python
python
Copy
import pandas as pd

# 1. Load data
df = pd.read_csv('data.csv')

# 2. Preview data
print(df.head())
print(df.tail())

# 3. Inspect structure
df.info()
print(df.describe())

# 4. Handle missing data
df = df.fillna({'numeric_col': df['numeric_col'].mean(), 'categorical_col': 'Unknown'})
df = df.dropna()

# 5. Clean data (e.g., remove '$' sign)
df['Amount'] = df['Amount'].replace('[\$,]', '', regex=True).astype(float)

# 6. Convert data types
df['numeric_col'] = df['numeric_col'].astype(float)
df['date'] = pd.to_datetime(df['date'])

# 7. Feature engineering
df['double_numeric'] = df['numeric_col'].apply(lambda x: x * 2)

# 8. Remove duplicates
df = df.drop_duplicates()

# 9. Group and aggregate with custom functions
grouped = df.groupby('Country').agg({
    'Amount': 'sum',
    'SalesPersona': lambda x: ', '.join(x.unique()),
    'Product': lambda x: ', '.join(x.unique())
}).reset_index()

# 10. Continue with further analysis/visualization...
print(grouped)

```

---
