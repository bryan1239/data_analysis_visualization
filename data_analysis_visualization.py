import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset from sklearn
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].apply(lambda x: iris.target_names[x])

# Display the first few rows
print(df.head())

# Check data types and missing values
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())

# Clean dataset (Iris has no missing values, but here's how you'd do it)
# df.dropna(inplace=True) or df.fillna(value, inplace=True)

# Basic statistics
print("\nStatistical Summary:\n", df.describe())

# Group by species and calculate mean
grouped = df.groupby('species').mean()
print("\nMean values by species:\n", grouped)

# Example finding
print("\nObservation: Iris-virginica generally has the largest petals.")

# Line Chart: Simulate time trend with index (not time-series data here)
plt.figure(figsize=(10, 5))
df[['sepal length (cm)', 'sepal width (cm)']].plot(
    title='Sepal Length vs. Width Over Index')
plt.xlabel("Index")
plt.ylabel("Length (cm)")
plt.grid(True)
plt.show()

# Bar Chart: Average petal length per species
grouped['petal length (cm)'].plot(
    kind='bar', title='Average Petal Length by Species', color='skyblue')
plt.ylabel("Petal Length (cm)")
plt.show()

# Histogram: Distribution of sepal length
df['sepal length (cm)'].plot(kind='hist', bins=10,
                             title='Distribution of Sepal Length', color='orange', edgecolor='black')
plt.xlabel("Sepal Length (cm)")
plt.show()

# Scatter Plot: Sepal length vs. petal length
plt.figure(figsize=(8, 6))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)',
                hue='species', data=df)
plt.title("Sepal vs. Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend()
plt.show()

try:
    data = pd.read_csv("your_dataset.csv")
except FileNotFoundError:
    print("Error: The file was not found. Please check the file path.")
except pd.errors.EmptyDataError:
    print("Error: The file is empty.")
except Exception as e:
    print("Unexpected error:", e)
