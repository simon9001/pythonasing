# ------------------------------------------------------
# Analyzing Data with Pandas and Visualizing Results
# ------------------------------------------------------

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# -----------------------------
# Task 1: Load and Explore Data
# -----------------------------

try:
    # Load the Iris dataset from sklearn
    iris = load_iris(as_frame=True)
    df = iris.frame  # pandas DataFrame
    print("Dataset loaded successfully!\n")

    # Display first few rows
    print("First 5 rows of the dataset:")
    print(df.head(), "\n")

    # Explore dataset info
    print("Dataset info:")
    print(df.info(), "\n")

    # Check missing values
    print("Missing values per column:")
    print(df.isnull().sum(), "\n")

    # Clean dataset (no missing values in Iris, but demo with fillna)
    df = df.fillna(method="ffill")

except FileNotFoundError:
    print("Error: Dataset file not found.")
except Exception as e:
    print(f"An error occurred while loading dataset: {e}")

# ----------------------------------
# Task 2: Basic Data Analysis
# ----------------------------------

print("Basic statistics of numerical columns:")
print(df.describe(), "\n")

# Grouping: Average petal length by species
grouped = df.groupby("target")["petal length (cm)"].mean()
print("Average petal length per species:")
print(grouped, "\n")

# Identify patterns
print("Observation: Iris-setosa generally has smaller petal lengths,")
print("while Iris-virginica has the longest petals.\n")

# ----------------------------------
# Task 3: Data Visualization
# ----------------------------------

# Set style
sns.set(style="whitegrid")

# 1. Line chart - example with sepal length trend
plt.figure(figsize=(8,5))
plt.plot(df.index, df["sepal length (cm)"], label="Sepal Length")
plt.title("Line Chart: Sepal Length Trend")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.show()

# 2. Bar chart - Average petal length per species
plt.figure(figsize=(8,5))
grouped.plot(kind="bar", color=["skyblue","orange","green"])
plt.title("Bar Chart: Avg Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.xticks(ticks=[0,1,2], labels=iris.target_names, rotation=0)
plt.show()

# 3. Histogram - Sepal Width distribution
plt.figure(figsize=(8,5))
plt.hist(df["sepal width (cm)"], bins=15, color="purple", alpha=0.7)
plt.title("Histogram: Sepal Width Distribution")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter plot - Sepal length vs Petal length
plt.figure(figsize=(8,5))
sns.scatterplot(
    x="sepal length (cm)",
    y="petal length (cm)",
    hue="target",
    palette="deep",
    data=df
)
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species", labels=iris.target_names)
plt.show()
