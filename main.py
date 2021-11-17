from math import dist
import numpy as np
import matplotlib.pyplot as plt
import random
from numpy.lib.function_base import copy
from sklearn.datasets import make_blobs

def euclideanDist(x,y):
        return np.sqrt(sum([pow(x[i]-y[i], 2.0) for i in range(len(x))]))


def getClusters(membershipMatrix):
        n_points = len(membershipMatrix[0])
        cluster_labels = list()
        for i in range(n_points):
            max_val, idx = max((val, idx) for (idx, val) in enumerate(membershipMatrix[:,i]))
            cluster_labels.append(idx)
        return cluster_labels


# X = [[1,3], [2,5], [4,8], [7,9]]

# X = [np.array(e) for e in X]
# clustersLabels = [1,2]
# m = 2.0
# membershipMatrix = np.array([[0.8, 0.7, 0.2, 0.1], [0.2, 0.3, 0.8, 0.9]])
# c = list()
# copyMembershipMatrix = copy(membershipMatrix)

# for i in range(len(clustersLabels)):
#     coefs = [pow(coef,m) for coef in membershipMatrix[i]]
#     numerator = sum([coefs[j] * X[j] for j in range(len(X))])
#     denominator = sum(coefs)
#     c.append(numerator/denominator)
# for i in range(len(X)):
#     distances = list()
#     for j in range(len(clustersLabels)):
#         distances.append(euclideanDist(X[i], c[j]))
#     for j in range(len(clustersLabels)):
#         tmp = sum([pow(distances[j] / distances[k], 2.0/(m-1)) for k in range(len(clustersLabels))])
#         membershipMatrix[j][i] = 1.0 / tmp

# print(getClusters(membershipMatrix))
# print(membershipMatrix)

X = [[1,3], [2,5], [4,8], [7,9]]
y = sum(X,[])
print(y)


