#2D PCA and 2D LDA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

# Load the data
data = np.loadtxt('MNIST_digits0-1-2.csv', delimiter=',')

# Split the data into features (X) and labels (y)
X, y = data[:, :-1], data[:, -1]

# Partition the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

# Calculate mean and standard deviation for each feature
mean = np.mean(X_train, axis=0)
std_dev = np.std(X_train, axis=0)
epsilon = 1e-10
std_dev[std_dev == 0] = epsilon

# Normalize the training and test data
X_train_normalized = (X_train - mean) / std_dev
X_test_normalized = (X_test - mean) / std_dev

# Run Linear Discriminant Analysis (LDA)
lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda_normalized = lda.fit_transform(X_train_normalized, y_train)
X_test_lda_normalized = lda.transform(X_test_normalized)

# Run 2D Principal Component Analysis (PCA)
pca = PCA(n_components=2)
X_train_pca_normalized = pca.fit_transform(X_train_normalized)
X_test_pca_normalized = pca.transform(X_test_normalized)

# Define a colormap with unique colors for each class
colors = ['r', 'g', 'b']

# Visualize the LDA projection
plt.figure(figsize=(12, 6))
plt.subplot(121)
for label in np.unique(y_train):
    plt.scatter(X_train_lda_normalized[y_train==label, 0], X_train_lda_normalized[y_train==label, 1], label=f'Training Class {int(label)}', color=colors[int(label)])
for label in np.unique(y_test):
    plt.scatter(X_test_lda_normalized[y_test==label, 0], X_test_lda_normalized[y_test==label, 1], label=f'Test Class {int(label)}', marker='*', color=colors[int(label)])
plt.title('LDA Projection (Normalized Data)')
plt.xlabel('LDA Component 1')
plt.ylabel('LDA Component 2')
plt.legend()
plt.grid(True)
# Limits
plt.xlim(-21, 21)
plt.ylim(-21, 22)

# Visualize the PCA projection
plt.subplot(122)
for label in np.unique(y_train):
    plt.scatter(X_train_pca_normalized[y_train==label, 0], X_train_pca_normalized[y_train==label, 1], label=f'Training Class {int(label)}', color=colors[int(label)])
#for label in np.unique(y_test):
#    plt.scatter(X_test_pca_normalized[y_test==label, 0], X_test_pca_normalized[y_test==label, 1], label=f'Test Class {int(label)}', marker='x')
plt.title('2D PCA Projection (Normalized Data)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()
