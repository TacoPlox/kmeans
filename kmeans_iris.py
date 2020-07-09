import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# Attribute information
#
# 1. sepal length in cm
# 2. sepal width in cm
# 3. petal length in cm
# 4. petal width in cm
# 5. class:
# -- Iris Setosa ('Iris-setosa')
# -- Iris Versicolour ('Iris-versicolor')
# -- Iris Virginica ('Iris-virginica')
#
# source: http://archive.ics.uci.edu/ml/datasets/Iris
#
# Note: actual iris class values in data are shown in parenthesis
df = pd.read_csv('iris.data', names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'iris_class'])

# Iris classes
# print(df.iris_class.unique())

# Reduce dimensionality with tSNE
tsne = TSNE(n_components=2, random_state=0)
tsne_obj = tsne.fit_transform(df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])

# Convert to DataFrame
tsne_df = pd.DataFrame(tsne_obj, columns=['x', 'y'])
# print(tsne_df)

kmeans = KMeans(n_clusters=3)

tsne_df['cluster'] = kmeans.fit_predict(tsne_df)

# map iris class to a color value
def mapcolor(lst):
    colors = []
    for l in lst:
        if l == 'Iris-setosa':
            colors.append(0)
        elif l == 'Iris-versicolor':
            colors.append(1)
        elif l == 'Iris-virginica':
            colors.append(2)
    return colors

colors = mapcolor(df['iris_class'])

# Uncomment one of the following two lines at a time

# To show predicted kmeans clusters, the next line should be uncommented
tsne_df.plot.scatter('x', 'y', c='cluster', colormap='viridis')

# To show the actual iris class, the next line should be uncommented
# tsne_df.plot.scatter('x', 'y', c=colors, colormap='viridis')

# Cluster centers
# print(kmeans.cluster_centers_)

# Cluster labels
# print(kmeans.labels_)

# Inclue cluster centers in scatter plot
plt.scatter(kmeans.cluster_centers_[0][0], kmeans.cluster_centers_[0][1], s=200, c='r')
plt.scatter(kmeans.cluster_centers_[1][0], kmeans.cluster_centers_[1][1], s=200, c='g')
plt.scatter(kmeans.cluster_centers_[2][0], kmeans.cluster_centers_[2][1], s=200, c='purple')

# Show scatter plot
plt.show()