import pandas as pd
import numpy as np
import collections
import scipy.cluster.hierarchy as hier
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt
from matplotlib import cm

# import and manipulate data
df_train = pd.read_csv('survey.csv')
X = df_train.iloc[:,5:33]

# reduce to 2 dimensions
pca = PCA(n_components = 2, random_state=1)
X_pca = pca.fit_transform(X)

# dendogram to graph iterative clustering process
dendrogram = hier.dendrogram(hier.linkage(X_pca, method ='ward'))
plt.title('Dendrogram')
plt.xlabel('questions')
plt.ylabel('Euclidean distances')
plt.show()
plt.figure()

# fit our model and choose n_clusters (chose 3 for satisfied, disatisfied, neutral.)
model = AgglomerativeClustering(n_clusters = 3, affinity ='euclidean', linkage ='ward')
y = model.fit_predict(X_pca)

# plot resulting clusters
plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], s = 50, c = 'yellow', label = 'Cluster 1')
plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], s = 50, c = 'green', label = 'Cluster 2')
plt.scatter(X_pca[y == 2, 0], X_pca[y == 2, 1], s = 50, c = 'blue', label = 'Cluster 2')
plt.title('Clusters of students')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.grid()
plt.show()
plt.figure()

# print number of observations in each cluster
print('AgglomerativeClustering w/ Ward Result : ')
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
plt.axvline(silhouette_avg, color="red", linestyle="--")
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.show()

# transfer data to pandas dataframe
y = pd.DataFrame(y, columns=['cluster'])
raw_result = pd.concat([X, y], axis=1)

# declare mean result of each cluster
mean_by_student_1 = raw_result[raw_result['cluster']==0].iloc[:, 0:28].mean(axis = 1)
mean_by_student_2 = raw_result[raw_result['cluster']==1].iloc[:, 0:28].mean(axis = 1)
mean_by_student_3 = raw_result[raw_result['cluster']==2].iloc[:, 0:28].mean(axis = 1)

# print mean/S.D of each cluster
print('Mean of cluster 1 : ' + str(mean_by_student_1.mean()) + ',STD :' + str(mean_by_student_1.std()))
print('Mean of cluster 2 : ' + str(mean_by_student_2.mean()) + ',STD :' + str(mean_by_student_2.std()))
print('Mean of cluster 2 : ' + str(mean_by_student_3.mean()) + ',STD :' + str(mean_by_student_3.std()))