from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import sklearn.metrics as metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
names = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width', 'Class']
dataset = pd.read_csv("iris.csv", names=names)
X, y = dataset.iloc[:, :-1], dataset['Class'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

# Plotting
plt.figure(figsize=(14, 7))

# Real Plot
plt.subplot(1, 3, 1)
plt.title('Real')
plt.scatter(X.Petal_Length, X.Petal_Width, c=np.array(['red', 'lime', 'black'])[y])

# K-Means Plot
kmeans_model = KMeans(n_clusters=3, random_state=0).fit(X)
plt.subplot(1, 3, 2)
plt.title('KMeans')
plt.scatter(X.Petal_Length, X.Petal_Width, c=np.array(['red', 'lime', 'black'])[kmeans_model.labels_])

print('Accuracy score of K-Means:', metrics.accuracy_score(y, kmeans_model.labels_))
print('Confusion matrix of K-Means:\n', metrics.confusion_matrix(y, kmeans_model.labels_))

# Gaussian Mixture Model Plot
gmm_model = GaussianMixture(n_components=3, random_state=0).fit(X)
y_cluster_gmm = gmm_model.predict(X)
plt.subplot(1, 3, 3)
plt.title('GMM Classification')
plt.scatter(X.Petal_Length, X.Petal_Width, c=np.array(['red', 'lime', 'black'])[y_cluster_gmm])
plt.show()

print('Accuracy score of GMM:', metrics.accuracy_score(y, y_cluster_gmm))
print('Confusion matrix of GMM:\n', metrics.confusion_matrix(y, y_cluster_gmm))
