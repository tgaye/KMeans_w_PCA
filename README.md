# KMeans_w_PCA
Utilizes k-means in order to visualize clusters of students, using survey answers. Compares results with and without PCA.  Tests a number of clustering methods.

Cluster Techniques used:

### K-means Clustering

### DBSCAN (density based scan)

### Agglomerative Clustering (link=ward)

### Agglomerative Clustering (link=complete)

First, lets take a look at our data.  Functions within DataExploration class will help with this.
```
df_train = pd.read_csv('survey.csv')
df_train.head()
```
Result:
```
  instr  class  nb.repeat  attendance  difficulty ...   Q24  Q25  Q26  Q27  Q28
0      1      2          1           0           4 ...     3    3    3    3    3
1      1      2          1           1           3 ...     3    3    3    3    3
2      1      2          1           2           4 ...     5    5    5    5    5
3      1      2          1           1           3 ...     3    3    3    3    3
4      1      2          1           0           1 ...     1    1    1    1    1
[5 rows x 33 columns]
```
We can see we have 33 dimensions to work with, however only 28 of these are survey questions; which are what we want to base clusters on.

Lets look at average response for each question in the survey.
```
hist_question_means():
```
![figure_1](https://user-images.githubusercontent.com/34739163/44390015-93466700-a4e9-11e8-8c11-dee4ff5b507c.png)
The scores seem roughly normally distributed, which tells us we shouldnt see top much variance in the data.

Now lets view survey participation, both by class and by instructor.  (13 classes, 3 instructors):

```
hist_total_response_by_class()
```
![figure_2](https://user-images.githubusercontent.com/34739163/44390025-96415780-a4e9-11e8-8579-096f82d2ec26.png)

```
hist_total_response_by_instr()
```
![figure_3](https://user-images.githubusercontent.com/34739163/44390033-99d4de80-a4e9-11e8-9474-0f3972c0604b.png)
We can see some classes had very low participation compared to others (more variance than our scores), and one instructor (#3)
had more students that filled out surveys than either other instructor combined.

## K-MEANS CLUSTERING

Now its time for clustering models.  Lets begin with the trusted K-means clustering technique. First we have to find our 
optimal K (# of clusters) using the "elbow method".  We plot the variance explained with each iteration of K, and choose the number
that represents the most variance, avoiding diminishing returns of adding more clusters.

![figure_4](https://user-images.githubusercontent.com/34739163/44390034-9b060b80-a4e9-11e8-93ee-1c98a1f179da.png)

It appears the elbow on the graph lies somewhere between 3 and 4, though closer to 3 I believe.  This works well because
this is a survey on satisfaction and its easy to hypothesize natural clusters for this data set, "Satisfied, Disatisfied, Neutral."

![figure_5](https://user-images.githubusercontent.com/34739163/44390037-9d686580-a4e9-11e8-95ea-07f1ece37478.png)

Wow, I really like how this model fits our data.  We can see 3 distinct groupings, and using this method it appears our principal 
components are in some way related to satisfaction of the students.  This is a compelling trait for our PC's to contain because it is 
easily interpretable (i.e not abstract)

We can also plot a silohouette of our clusters to help visualize outliars (I will be doing this for the remained of clustering methods 
we test against kmeans as well)

![figure_6](https://user-images.githubusercontent.com/34739163/44390041-9e999280-a4e9-11e8-94d2-a26a9b5d5920.png)

## DBSCAN
![figure_7](https://user-images.githubusercontent.com/34739163/44390043-9fcabf80-a4e9-11e8-8fa7-bf9ac6a70b12.png)
My first impression of the model is that it doesn't fit our given data set very well.  The way DBSCAN works is it assumes every observation is its own cluster, and then it groups them based on their proximity/density (hence density-based scan).  Given
our data spans a good length (disatisfied to satisfied) and has no real seperation to its 'clusters', this might not be the best model.

Our silhouette visualization doesn't look any better for this model:
![figure_8](https://user-images.githubusercontent.com/34739163/44390046-a0fbec80-a4e9-11e8-925e-bb204a009f9a.png)

## Agglomerative Clustering (Link = Ward)

We will try two variations of agg. clustering by playing with the hyperparameter that adjusts the linkage criterion we use.
From scikit learn docs on the AgglomerativeClustering() function:

```
linkage : {“ward”, “complete”, “average”}, optional, default: “ward”

Which linkage criterion to use. The linkage criterion determines which distance to use 
between sets of observation. The algorithm will merge the pairs of cluster that minimize this criterion.
```

The first graph I want to generate is a dendogram, which will help us better view how our error function decreases as we add clusters.
```
dendrogram = hier.dendrogram(hier.linkage(X_pca, method ='ward'))
```
![figure_9](https://user-images.githubusercontent.com/34739163/44390049-a2c5b000-a4e9-11e8-8567-c0ed0912e7cb.png)

Whats nice about dendograms is we can easily see what our model tells us our number of clusters should be, each cluster is its own distinct color.  Since red and green are our furthest children nodes, we have 2 clusters.

We train our model on what we found to be optimal clusters (n=2)
```
model = AgglomerativeClustering(n_clusters = 2, affinity ='euclidean', linkage ='ward')
y = model.fit_predict(X_pca)
```

![figure_10](https://user-images.githubusercontent.com/34739163/44390051-a5280a00-a4e9-11e8-8a33-e4b948a9342e.png)
Hmm, this is an interesting fit for our data.  I can see a potential use for this clustering method if our clients wanted to treat/view neutral and satisfied students in a similar light, and cared more about clustering disatisfied students.

The silhouette looks like a much cleaner fit than the DBSCAN's as well:
![figure_11](https://user-images.githubusercontent.com/34739163/44390054-a6593700-a4e9-11e8-905e-6848ca676261.png)

## Agglomerative Clustering (Link = Complete)
```
dendrogram = hier.dendrogram(hier.linkage(X_pca, method ='complete'))
```
![figure_12](https://user-images.githubusercontent.com/34739163/44390059-a822fa80-a4e9-11e8-885e-3d4610c5fb9a.png)

I see 4 distinct children node colors, so we will use n=4 for our clusters.

![figure_13](https://user-images.githubusercontent.com/34739163/44390062-a9ecbe00-a4e9-11e8-929b-a2b2c715da49.png)

This is an interesting fit as well if im being honest.   It looks very much like our k-means clusters, only with a rather small outliar group of high value principal component #2.  

If I had to draw a conclusion about this, I'd say it looks like our model is differentiating between satisfied students, and extremely statsfied students.  This might also be useful, if our client wants to make this distinction.

Lastly, a silhouette visual for our second variation of agg clustering model:
![figure_14](https://user-images.githubusercontent.com/34739163/44390068-ab1deb00-a4e9-11e8-9ca4-94b78523e519.png)

The small grouping of light blue that contains no negative outliars in our silhouette chart furthers my case that the model is grouping satisfied students and extremely satisfied students seperately.

If I had to pick my two favorite fits for our given data, from the clustering methods demo'd here; it's a toss up between K-means and Agg Clustering with complete linkage.  Both fit the data in intuitive, satisfying ways when paired with PCA.  Again though, the optimal model depends on the business use / statistic of interest at hand.  
