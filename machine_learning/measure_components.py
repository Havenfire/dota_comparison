import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Read data from CSV file
csv_file_path = 'hero_data.csv'  # Replace with the actual path to your CSV file
df = pd.read_csv(csv_file_path)

# Extract numerical columns for PCA
numerical_columns = df.columns[1:]  # Assuming the first column is heroId
data_for_pca = df[numerical_columns].values

# Perform PCA for different values of n_components
n_components_range = range(1, len(numerical_columns) + 1)
explained_variances = []

for n_components in n_components_range:
    pca = PCA(n_components=n_components)
    pca.fit(data_for_pca)
    explained_variances.append(np.sum(pca.explained_variance_ratio_))

# Plot explained variance for each n_components
plt.plot(n_components_range, explained_variances)
plt.xlabel('Number of Components (n_components)')
plt.ylabel('Explained Variance')
plt.title('Explained Variance vs. Number of Components')
plt.show()
