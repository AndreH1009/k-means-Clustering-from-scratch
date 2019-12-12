# Clustering: k-means from scratch
In this project I implement the k-means algorithm, which is an unsupervised learning algorithm for classification tasks. I avoid resorting to external libraries to really make sure I understand the algorithm. Functionality includes text extraction via regular expressions from the data file, normalization of the data set, initialisation of the k-means algorithm with a set of centers using the farthest traversal algorithm, generation of a clustering of the given data set where each step of the clustering process is optionally visualised, and the prediction of the optimal number of clusters for the given data set using the elbow method. Support is currently restricted to two-dimensional data sets. 

## Purpose
Deepen my knowledge of Python and the k-means Algorithm.

## The Data Set
I use the "Old Faithful" data set, which observes the duration of geyser eruptions and the waiting time between geyser eruptions. The source is https://www.stat.cmu.edu/~larry/all-of-statistics/=data/faithful.dat .

## Example
![Screenshot](/docs/images/prior.png)
![Screenshot](/docs/images/clustering.png)
![Screenshot](/docs/images/elbow.png)