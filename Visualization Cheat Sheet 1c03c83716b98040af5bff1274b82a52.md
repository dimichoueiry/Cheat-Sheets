# Visualization Cheat Sheet

# Plot Type Cheatsheet

### 1. **Line Plot**

- **When to Use:**
    
    Display trends over continuous data or time series.
    
- **Example Use Cases:**
    - Stock prices over time
    - Temperature or sales trends
- **Key Functions:**
    - Matplotlib: `plt.plot()`
    - Seaborn: `sns.lineplot()`

---

### 2. **Bar Plot**

- **When to Use:**
    
    Compare discrete, categorical values or aggregate measures.
    
- **Example Use Cases:**
    - Sales figures by region
    - Population counts per category
- **Key Functions:**
    - Matplotlib: `plt.bar()`
    - Seaborn: `sns.barplot()`

---

### 3. **Histogram (Histplot)**

- **When to Use:**
    
    Visualize the distribution of a numeric variable.
    
- **Example Use Cases:**
    - Frequency distribution of ages
    - Income distribution in a region
- **Key Functions:**
    - Matplotlib: `plt.hist()`
    - Seaborn: `sns.histplot()`

---

### 4. **Box Plot**

- **When to Use:**
    
    Summarize the distribution of a dataset (median, quartiles, and outliers).
    
- **Example Use Cases:**
    - Examining salary ranges
    - Comparing test scores across groups
- **Key Functions:**
    - Matplotlib: `plt.boxplot()`
    - Seaborn: `sns.boxplot()`

---

### 5. **Heatmap**

- **When to Use:**
    
    Visualize matrix-style data or correlations.
    
- **Example Use Cases:**
    - Correlation matrices between variables
    - Confusion matrices in classification tasks
- **Key Functions:**
    - Seaborn: `sns.heatmap()`

---

### 6. **Scatter Plot**

- **When to Use:**
    
    Explore relationships between two continuous variables.
    
- **Example Use Cases:**
    - Height vs. weight
    - Advertising spend vs. sales revenue
- **Key Functions:**
    - Matplotlib: `plt.scatter()`
    - Seaborn: `sns.scatterplot()`

---

### 7. **Violin Plot**

- **When to Use:**
    
    Show the distribution and density of data, combining box plot and KDE features.
    
- **Example Use Cases:**
    - Comparing distributions across several categories
- **Key Functions:**
    - Seaborn: `sns.violinplot()`

---

### 8. **Count Plot**

- **When to Use:**
    
    Visualize the frequency/count of observations in categorical variables.
    
- **Example Use Cases:**
    - Frequency of categories (e.g., product types, regions)
- **Key Functions:**
    - Seaborn: `sns.countplot()`

---

### 9. **Pair Plot**

- **When to Use:**
    
    Quickly explore pairwise relationships and distributions in multivariate data.
    
- **Example Use Cases:**
    - Comprehensive EDA of a dataset with multiple numeric variables
- **Key Functions:**
    - Seaborn: `sns.pairplot()`

---

## Summary

- **Line Plots** are ideal for trends and continuous data.
- **Bar Plots** work best for comparing aggregates across discrete categories.
- **Histograms** help you understand the distribution of a single numerical variable.
- **Box Plots** provide a quick summary of data distribution and highlight outliers.
- **Heatmaps** excel at visualizing correlations or matrix data.
- **Scatter Plots** are useful for revealing relationships between two continuous variables.
- **Violin Plots** offer a deeper look into distribution shapes across groups.
- **Count Plots** show how frequently each category occurs.
- **Pair Plots** are great for an overall look at relationships among several variables.

# Matplotlib & Seaborn Cheatsheet

## 1. Overview

- **Matplotlib.pyplot** is the core plotting library in Python. It provides functions to create and customize plots at a low level.
- **Seaborn** builds on top of matplotlib, offering a high-level interface for attractive statistical graphics. It sets up a nice default style and simplifies complex visualizations.

They work together seamlessly: you can use seaborn to create a plot and then further tweak it using matplotlib commands.

---

## 2. Matplotlib.pyplot Essentials

### Importing and Basic Setup

```python
import matplotlib.pyplot as plt

```

### Creating Figures and Axes

- **Create a Figure & Axes:**
    
    ```python
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ```
    
- **Set Title and Labels:**
    
    ```python
    ax.set_title("Plot Title")
    ax.set_xlabel("X-axis Label")
    ax.set_ylabel("Y-axis Label")
    
    ```
    

### Common Plot Types

- **Line Plot:**
    
    ```python
    plt.plot(x, y, label="Line", color="blue", linestyle='-', marker='o')
    
    ```
    
- **Scatter Plot:**
    
    ```python
    plt.scatter(x, y, color="red", marker="x")
    
    ```
    
- **Bar Chart:**
    
    ```python
    plt.bar(x_categories, y_values, color="green")
    
    ```
    
- **Histogram:**
    
    ```python
    plt.hist(data, bins=30, color="purple", edgecolor="black")
    
    ```
    
- **Box Plot:**
    
    ```python
    plt.boxplot(data)
    
    ```
    
- **Pie Chart:**
    
    ```python
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    
    ```
    

### Customization and Annotations

- **Legend:**
    
    ```python
    plt.legend(loc="upper right")
    
    ```
    
- **Grid:**
    
    ```python
    plt.grid(True)
    
    ```
    
- **Annotations:**
    
    ```python
    plt.annotate("Important point", xy=(x_val, y_val), xytext=(x_val+1, y_val+1),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    ```
    

### Layout and Saving

- **Adjust Layout:**
    
    ```python
    plt.tight_layout()
    
    ```
    
- **Saving Figures:**
    
    ```python
    plt.savefig("my_plot.png", dpi=300)
    
    ```
    
- **Display the Plot:**
    
    ```python
    plt.show()
    
    ```
    

---

## 3. Seaborn Essentials

### Importing and Setting Up

```python
import seaborn as sns
sns.set(style="whitegrid")  # Use a built-in theme, e.g., "whitegrid", "dark", etc.

```

### Common Plot Types

- **Line Plot (with Confidence Interval):**
    
    ```python
    sns.lineplot(x="time", y="value", data=df, marker="o")
    
    ```
    
- **Scatter Plot:**
    
    ```python
    sns.scatterplot(x="variable1", y="variable2", data=df, hue="category")
    
    ```
    
- **Bar Plot:**
    
    ```python
    sns.barplot(x="category", y="value", data=df, palette="viridis")
    
    ```
    
- **Histogram/Distribution Plot:**
    
    ```python
    sns.histplot(data=df, x="value", bins=30, kde=True)
    
    ```
    
- **Box Plot:**
    
    ```python
    sns.boxplot(x="category", y="value", data=df)
    
    ```
    
- **Violin Plot:**
    
    ```python
    sns.violinplot(x="category", y="value", data=df)
    
    ```
    
- **Heatmap (for Correlation Matrices):**
    
    ```python
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    
    ```
    
- **Pair Plot (for exploring relationships):**
    
    ```python
    sns.pairplot(df, hue="category")
    
    ```
    
- **Count Plot (Categorical Counts):**
    
    ```python
    sns.countplot(x="category", data=df)
    
    ```
    
- **Regression Plot (with lmplot):**
    
    ```python
    sns.lmplot(x="variable", y="target", data=df, ci=95)
    
    ```
    

### Customization with Seaborn

- **Changing Color Palette:**
    
    ```python
    sns.set_palette("pastel")
    
    ```
    
- **Context Settings (e.g., for presentations):**
    
    ```python
    sns.set_context("talk")  # Options: paper, notebook, talk, poster
    
    ```
    

---

## 4. How They Work Together

- **Seaborn Uses Matplotlib Behind the Scenes:**
    
    When you create a plot with seaborn, it generates a matplotlib figure and axes. You can use matplotlib functions to further customize these plots.
    
    **Example:**
    
    ```python
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create a seaborn plot
    ax = sns.scatterplot(x="variable1", y="variable2", data=df, hue="category")
    
    # Customize further with matplotlib
    plt.title("Customized Scatter Plot")
    plt.xlabel("Variable 1")
    plt.ylabel("Variable 2")
    plt.legend(title="Category")
    plt.show()
    
    ```
    
- **Mixing Both:**
    
    You can set the overall aesthetic with seaborn (e.g., sns.set()) and then add finer details with plt (e.g., annotations, custom legends, saving figures).
    
- **Subplots and Grids:**
    
    You can combine seaborn plots in matplotlib subplots:
    
    ```python
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.boxplot(x="category", y="value", data=df, ax=axes[0])
    sns.violinplot(x="category", y="value", data=df, ax=axes[1])
    plt.tight_layout()
    plt.show()
    
    ```
    

---

## 5. Real-World Scenario Examples

### Scenario 1: Exploratory Data Analysis (EDA)

1. **Distribution of a Numeric Variable:**
    
    ```python
    sns.histplot(data=df, x="age", bins=20, kde=True)
    plt.title("Age Distribution")
    plt.show()
    
    ```
    
2. **Relationship Between Two Variables:**
    
    ```python
    sns.scatterplot(x="income", y="spending", data=df, hue="region")
    plt.title("Income vs. Spending by Region")
    plt.xlabel("Income")
    plt.ylabel("Spending")
    plt.show()
    
    ```
    
3. **Comparing Categories:**
    
    ```python
    sns.boxplot(x="region", y="income", data=df)
    plt.title("Income Distribution by Region")
    plt.show()
    
    ```
    

### Scenario 2: Advanced Custom Visualizations

1. **Heatmap of Correlations:**
    
    ```python
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()
    
    ```
    
2. **Pair Plot for Multivariate Analysis:**
    
    ```python
    sns.pairplot(df, hue="region", diag_kind="kde")
    plt.show()
    
    ```
    
3. **Regression Analysis:**
    
    ```python
    sns.lmplot(x="experience", y="salary", data=df, aspect=1.5, ci=95)
    plt.title("Salary vs. Experience")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.show()
    
    ```
    

---

## Summary

- **Matplotlib.pyplot** is best for low-level customization and creating detailed layouts. Use it for:
    - Creating figures, subplots, and saving figures.
    - Basic plot types like line, scatter, bar, histogram, etc.
    - Fine-tuning with annotations, legends, titles, and labels.
- **Seaborn** simplifies statistical plotting:
    - Easily create attractive plots with less code.
    - Built-in themes and color palettes.
    - Functions like `sns.scatterplot()`, `sns.boxplot()`, `sns.heatmap()`, and `sns.pairplot()` streamline common visualization tasks.
- **Working Together:**
    
    Seaborn’s plots are matplotlib objects. You can start with seaborn for its high-level features and then use matplotlib functions to adjust layout and add details.
    

# Matplotlib & Seaborn Cheatsheet: Stateful vs. Object-Oriented Plotting

## 1. Overview

- **Matplotlib** is a powerful plotting library.
- **Seaborn** builds on matplotlib, simplifying statistical visualizations with attractive defaults.

**Two Main Plotting Styles:**

- **Stateful (Implicit) API:**
Uses global functions like `plt.figure()`, `plt.subplot()`, and `plt.plot()` that affect the current axes.
- **Object-Oriented (Explicit) API:**
Creates figures and axes objects directly (using `fig, ax = plt.subplots()`) and then calls methods on these objects for finer control.

---

## 2. Stateful Plotting

### Creating a Single Plot

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Sample Data
df = pd.DataFrame({
    "x": range(10),
    "y": [i**2 for i in range(10)],
    "z": [i**0.5 for i in range(10)]
})

# Create a figure
plt.figure(figsize=(8, 6))

# Create a line plot (matplotlib style)
plt.plot(df['x'], df['y'], label="Line Plot", color='blue', marker='o')

# Add labels and title
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Stateful: Line Plot")
plt.legend()
plt.grid(True)
plt.show()

```

### Creating Multiple Subplots (Stateful)

```python
# Set the overall figure size
plt.figure(figsize=(10, 8))

# Subplot 1: Line Plot using Seaborn (implicitly plots on current axes)
plt.subplot(2, 2, 1)  # 2 rows, 2 columns, subplot 1
sns.lineplot(x="x", y="y", data=df)
plt.title("Line Plot")

# Subplot 2: Scatter Plot using Seaborn
plt.subplot(2, 2, 2)
sns.scatterplot(x="x", y="z", data=df, color='red')
plt.title("Scatter Plot")

# Subplot 3: Histogram using Seaborn
plt.subplot(2, 2, 3)
sns.histplot(df['y'], bins=10, kde=True, color='green')
plt.title("Histogram")

# Subplot 4: Box Plot using matplotlib directly
plt.subplot(2, 2, 4)
plt.boxplot(df['y'])
plt.title("Box Plot")

plt.tight_layout()
plt.show()

```

---

## 3. Object-Oriented Plotting

### Creating a Single Plot with OO API

```python
# Create a figure and a single set of axes
fig, ax = plt.subplots(figsize=(8, 6))

# Plot using matplotlib on the specific axes
ax.plot(df['x'], df['y'], label="Line Plot", color='blue', marker='o')
ax.set_title("OO: Line Plot")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.legend()
ax.grid(True)
plt.show()

```

### Creating Multiple Subplots with OO API

```python
# Create a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Top-left subplot: Line Plot (Seaborn)
sns.lineplot(x="x", y="y", data=df, ax=axes[0, 0])
axes[0, 0].set_title("Line Plot")

# Top-right subplot: Scatter Plot (Seaborn)
sns.scatterplot(x="x", y="z", data=df, ax=axes[0, 1], color='red')
axes[0, 1].set_title("Scatter Plot")

# Bottom-left subplot: Histogram (Seaborn)
sns.histplot(df['y'], bins=10, kde=True, ax=axes[1, 0], color='green')
axes[1, 0].set_title("Histogram")

# Bottom-right subplot: Box Plot (matplotlib)
axes[1, 1].boxplot(df['y'])
axes[1, 1].set_title("Box Plot")

plt.tight_layout()
plt.show()

```

---

## 4. Combining Seaborn with Matplotlib (Both Approaches)

### Example: Enhancing a Seaborn Plot with Matplotlib Customization

**Stateful Approach:**

```python
plt.figure(figsize=(8, 6))
sns.scatterplot(x="x", y="z", data=df, hue="x", palette="viridis")
plt.title("Stateful: Customized Scatter Plot")
plt.xlabel("X-axis")
plt.ylabel("Z-axis")
plt.legend(title="X Values")
plt.show()

```

**Object-Oriented Approach:**

```python
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x="x", y="z", data=df, hue="x", palette="viridis", ax=ax)
ax.set_title("OO: Customized Scatter Plot")
ax.set_xlabel("X-axis")
ax.set_ylabel("Z-axis")
ax.legend(title="X Values")
plt.show()

```

---

## 5. Summary

- **Stateful API:**
    - Uses global functions that automatically target the "current" axes.
    - Quick for exploratory analysis.
    - Example functions: `plt.figure()`, `plt.subplot()`, `plt.plot()`, `plt.show()`
- **Object-Oriented API:**
    - Creates explicit `Figure` and `Axes` objects.
    - Offers detailed control over each subplot.
    - Example functions: `fig, ax = plt.subplots()`, then methods like `ax.plot()`, `ax.set_title()`, etc.
- **Mixing with Seaborn:**
    - Seaborn plots generate matplotlib axes objects.
    - You can pass an `ax` parameter to direct Seaborn plots to specific subplots.
    - After creating plots with Seaborn, further customize them using matplotlib’s methods.