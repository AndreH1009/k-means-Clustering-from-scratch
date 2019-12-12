# author: Andre Hoffmann

import re
import numpy as np
import matplotlib.pyplot as plt


# some general declarations:
data_source = 'geyser.txt'  # source: https://www.stat.cmu.edu/~larry/all-of-statistics/=data/faithful.dat
x_data = []  # eruption time in minutes
y_data = []  # waiting time till next eruption
dim = 2


# implement normalization routine for a list of values:
def normalize(data):
    ndata = []
    for value in data:
        ndata.append((value - min(data))/(max(data)-min(data)))
    return ndata


# implement distance measure:
# p1, p2 are tuples for the point coordinates
# works for dim dimensions
def sqrdist(p1, p2):
    dist = 0
    for i in range(dim):
        dist += (p2[i] - p1[i])**2
    return dist


# implement function to compute a centroid for a given cluster:
def getCentroid(cluster, dimension):  # cluster is a list of tuples
    coords = []
    centroid = ()  # is a tuple with 'dimension' coordinates
    for i in range(dimension):
        coords.append([])
    for point in cluster:
        for j in range(dimension):
            coords[j].append(point[j])
    for i in range(dimension):
        centroid = centroid + (sum(coords[i])/len(cluster),)
    return centroid


# implement function to read data from file:
def readDataFile(path, plot=False):
    file = open(path)
    for lines in file:
        parsed = re.search(r"\d+\s+([0-9\.]+)\s+(\d+)", lines)
        x_data.append(float(parsed.group(1)))
        y_data.append(int(parsed.group(2)))
    file.close()
    x = normalize(x_data)
    y = normalize(y_data)
    zdata = list(zip(x, y))
    # plot the dataset:
    if plot:
        plt.scatter(x, y)
        plt.xlabel('Eruption Time')
        plt.ylabel('Waiting Time')
        plt.title("The data set prior to clustering:")
        plt.show()
    return zdata


# implement function to visualize a given clustering:
def plotClustering(clustering, dimension=2, colors=['cyan', 'magenta'], centercol='blue', title=None):
    coords = []
    for i in range(len(clustering.keys())):
        coords.append([])  # for each cluster one list
        for j in range(dimension):  # for each dimension one list of coords
            coords[i].append([])
            for point in clustering[list(clustering.keys())[i]]:
                coords[i][j].append(point[j])  # for each point store the respective coordinate
    if len(colors) < dimension:  # prevent errors due to too few colors in the list
        for i in range(dim):
            colors.append(colors[0])
    # draw the points for each cluster:
    for sets in coords:
        plt.scatter(sets[0], sets[1], color=colors[0])
        colors = colors[1:]
    # draw the centers:
    for center in clustering.keys():
        plt.scatter(center[0], center[1], color=centercol, marker='x', s=200)
    plt.xlabel('Eruption Time')
    plt.ylabel('Waiting Time')
    plt.title(title)
    plt.show()


# implement Farthest Traversal to initialize Lloyd:
def fatra(k, data):  # k: number of centers
    # generate random first center:
    randex = np.random.randint(0, len(x_data))
    centers = [data[randex]]
    # append the farthest-away point from previous centers until k:
    while len(centers) < k:
        # compute and store distance from centers to each data point
        dists = []  # dists[i]: stores distance from point zdata[i] to all centers.
        for point in data:
            pcdist = []  # store distances from a point to all centers
            for center in centers:
                pcdist.append(sqrdist(point, center))
            dists.append(min(pcdist))
        centers.append(data[dists.index(max(dists))])
    return centers


# implement k-means/Lloyd's algorithm to return a clustering:
# optionally plots all stages of clustering for presentation purposes, set plotall = True.
def k_means(k, datapath, init='FT', dimension = 2, iterations = 5, plotall=False):
    clustering = {}  # maps cluster sets of points to centers given by coordinate tuples
    if plotall:
        data = readDataFile(path=datapath, plot=True)
    else:
        data = readDataFile(path=datapath)
    # get initial centers from specified algorithm init:
    if init == 'FT':
        cInit = fatra(k, data)
    else:
        print("choose valid initialisation!")
        return None
    # initialise the clusters as empty lists for each center:
    for center in cInit:
        clustering[center] = []
    for i in range(iterations):
        # empty all the clusters before reassignment:
        for center in clustering.keys():
            clustering[center] = []
        # for each data point compute distances to all centers,
        # assign each point to the center nearest to it
        for point in data:
            bestdist = np.inf
            bestc = None
            for center in clustering.keys():
                curdist = sqrdist(point, center)
                if (bestdist > curdist):
                    bestdist = curdist
                    bestc = center
            clustering[bestc].append(point)
        # plot the clustering after assignment the data points
        if plotall:
            plotClustering(clustering, title='Iteration %d, after assignment step:' % (i+1))
        # compute the centroids of the clusters and replace the former centers:
        for center in clustering.keys():
            centroid = getCentroid(clustering[center], dimension)
            clustering[centroid] = clustering.pop(center)
        # plot the clustering after centroid computation
        if plotall:
            plotClustering(clustering, title='Iteration %d, after centroid computation:' % (i+1))
    return clustering


# compute the sum of interclustural distances for all clusters as a goodness measure of the clustering:
def comp_wss(clustering):
    wss = 0
    for center in clustering.keys():
        for point in clustering[center]:
            wss += sqrdist(center, point)
    return wss


# return optimal number of clusters for given data set.
# optionally plot elbow curve. plot=False may save considerable computation time.
def elbow_method(min_, max_, iterations=5, plot=False):
    results = {}
    min_wss = np.infty
    best_k = None
    k_found = False
    for k in range(min_, max_):
        values = []
        for i in range(iterations):
            values.append(comp_wss(k_means(k, data_source)))
        results[k] = sum(values)/len(values)
        if results[k] < min_wss and not k_found:
            min_wss = results[k]
            best_k = k
        elif results[k] > min_wss:
            k_found = True
            if not plot:  # keep computing only for the sake of illustration; stop when no plot is desired anyway.
                break
    if plot:  # plots the number of clusters against the average goodness of the resulting clusters.
        y = []
        for k in range(min_, max_):
            y.append(results[k])
        plt.plot(results.keys(), y, "-ob")
        plt.xlabel("k")
        plt.ylabel("average wss on %d iterations" % iterations)
        plt.show()
    return best_k


# for demonstration: run k_means on optimal k as computed using the elbow method.
# alternatively just set k = 2 for the current data set.

# opt_k = elbow_method(1, 6)
k_means(k=2, datapath=data_source, iterations=4, plotall=True)
