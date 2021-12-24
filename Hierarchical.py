from sklearn.cluster import AgglomerativeClustering
import src
import numpy as np

def calcCentroids(data, labels, k):
    centroids = []
    for i in range(k):
        tmp = data[labels == i, :]
        avg = sum(tmp)
        avg = [np.array(avg[i])/len(tmp) for i in range(len(avg))]
        centroids.append(avg)
    return np.array(centroids)


data = src.X1
xlabel, ylabel, title = 'Income', 'Score', 'Hierarchical'
k = 5
hc = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward')
labels = hc.fit_predict(data)
centroids = calcCentroids(data, labels, k)
src.plotClusters(data, labels, centroids, '2d', xlabel, ylabel, title)
src.metrics(data, centroids, labels)



