import pandas as pd
import numpy as np
import collections
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt
from matplotlib import cm

# load and manipulate data
df_train = pd.read_csv('survey.csv')
print(df_train.head())
X = df_train.iloc[:,5:33] # we only want to train on survey questions (columns 5-33)

# reduce to 2 dimensions.  Will help visualize our k-means
pca = PCA(n_components = 2, random_state=1)
X_pca = pca.fit_transform(X)
print('Explained Variance Ratio : ' + str(pca.explained_variance_ratio_.cumsum()[1])) # Variability in data maintained

# variables for elbow method (for determining n_clusters)
distortions = []
K_to_try = range(1, 6)

# elbow method for finding optimal K
for i in K_to_try:
    model = KMeans(
            n_clusters=i,
            init='k-means++',
            random_state=1)
    model.fit(X_pca)
    distortions.append(model.inertia_)

# plot elbow
plt.plot(K_to_try, distortions, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Distortion')
plt.show()
plt.figure()

# use the best K from elbow method
model = KMeans(
    n_clusters=3,
    init='k-means++',
    random_state=1)

model = model.fit(X_pca)
y = model.predict(X_pca)

# plot k means cluster
plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], s = 50, c = 'yellow', label = 'Cluster 1') # clusters 1-3
plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], s = 50, c = 'green', label = 'Cluster 2')
plt.scatter(X_pca[y == 2, 0], X_pca[y == 2, 1], s = 50, c = 'red', label = 'Cluster 3')
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s = 100, c = 'blue', label = 'Centroids') # centroids
plt.title('Clusters of Students')
plt.xlabel('Principle Component 1')
plt.ylabel('Principle Component 2')
plt.legend()
plt.grid()
plt.show()
plt.figure()

# print number of observations in each cluster
print('K Means Result : ')
print(collections.Counter(y))

cluster_labels = np.unique(y)
n_clusters = cluster_labels.shape[0]

# initializing variables for creating silhouettes of clusters
silhouette_vals = silhouette_samples(X_pca, y, metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []

# generate silhouette for each cluster
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y == c] # store silhouette vals for each cluster
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters) # assign different color each loop
    plt.barh(range(y_ax_lower, y_ax_upper), # plot silhouette for each cluster
            c_silhouette_vals,
            height=1.0,
            edgecolor='none',
            color=color)
    yticks.append((y_ax_lower + y_ax_upper) / 2.)
    y_ax_lower += len(c_silhouette_vals)
silhouette_avg = np.mean(silhouette_vals) # avg for plotting

# plot silhouette coefficients (labels)
plt.axvline(silhouette_avg, color="red", linestyle="--") # plots line to show avg of each cluster
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.show()

# use best k(3) from Elbow method
model_k = KMeans(
    n_clusters=3,
    init='k-means++',
    random_state=1)

# fit with X instead of X_pca in order to compare with vs without
model_k = model_k.fit(X)
y_final = model_k.predict(X)

# print number of observations in each cluster without PCA to compare the two.
print('Final K Means Result (no PCA) : ')
print(collections.Counter(y_final)) # Of 5,820 observations only one was classified differently using PCA than without PCA.

# transfer data into pandas dataframes for easy manipulation:
# without PCA
y_final = pd.DataFrame(y_final, columns=['cluster'])
raw_result = pd.concat([X, y_final], axis=1)
# with PCA
y = pd.DataFrame(y, columns=['cluster'])
raw_result_pca = pd.concat([X, y], axis=1)

# compare mean score of each cluster
mean_by_student_1 = raw_result[raw_result['cluster']==0].iloc[:, 0:28].mean(axis = 1)
mean_by_student_2 = raw_result[raw_result['cluster']==1].iloc[:, 0:28].mean(axis = 1)
mean_by_student_3 = raw_result[raw_result['cluster']==2].iloc[:, 0:28].mean(axis = 1)

# print mean/S.D
print('Mean cluster 1 : ' + str(mean_by_student_1.mean()) + ',STD :' + str(mean_by_student_1.std()))
print('Mean cluster 2 : ' + str(mean_by_student_2.mean()) + ',STD :' + str(mean_by_student_2.std()))
print('Mean cluster 3 : ' + str(mean_by_student_3.mean()) + ',STD :' + str(mean_by_student_3.std()))


