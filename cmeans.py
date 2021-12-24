import numpy as np
import random
import src

class CMEANS:
    def __init__(self, k, n_points, m = 2, max_iterations = 1000):
        self.k = k
        self.m = m # fuzzy-parameter
        self.n_points = n_points
        self.max_iterations = max_iterations
        # add stop criterion

    # иницализация матрицы принадлежностей случайными числами (сумма в каждой строке 1)
    # строки матрицы = точки, а столбцы = кластеры => матрица N * K 
    def initMembershipMatrix(self):
        self.membershipMatrix = list()
        for i in range(self.k):
            randList = [random.random() for i in range(self.n_points)]
            summation = sum(randList)
            tmpList = [x / summation for x in randList]
            self.membershipMatrix.append(tmpList)
        self.membershipMatrix = np.array(self.membershipMatrix)

    # вычисление центров кластеров
    def calcClusterCenter(self, X):
        self.centerClusters = list()
        for i in range(self.k):
            coefs = [pow(coef, self.m) for coef in self.membershipMatrix[i]]
            numerator = sum([coefs[j] * np.array(X[j]) for j in range(self.n_points)])
            denominator = sum(coefs)
            self.centerClusters.append(numerator / denominator)
    
    def euclideanDist(self,x,y):
        return np.sqrt(sum([pow(x[i]-y[i],2) for i in range(len(x))]))

    # обновление матрицы принадлежностей
    def updateMembershipValue(self,  X):
        # надо посчитать расстояния от точек до центроидов
        for i in range(self.n_points):
            distances = list()
            for j in range(self.k):
                distances.append(self.euclideanDist(X[i], self.centerClusters[j]))
            for j in range(self.k):
                tmp = sum([pow(distances[j] / distances[k], 2.0/(self.m-1)) for k in range(self.k)])
                self.membershipMatrix[j][i] = 1.0 / tmp
    
    def getClusters(self):
        cluster_labels = list()
        for i in range(self.n_points):
            max_val, idx = max((val, idx) for (idx, val) in enumerate(self.membershipMatrix[:,i]))
            cluster_labels.append(idx)
        return cluster_labels

    def fit(self, data):
        self.initMembershipMatrix()
        curr = 0
        while curr <= self.max_iterations:
            self.calcClusterCenter(data)
            self.updateMembershipValue(data)
            curr += 1
        self.cluster_labels = self.getClusters()



def main():
    data = src.X1
    xlabel, ylabel, title = 'Income', 'Score', 'KMeans'
    cmeans = CMEANS(5, len(data))
    cmeans.fit(data)
    labels = cmeans.cluster_labels
    centroids = cmeans.centerClusters
    src.plotClusters(data, labels, centroids, '2d', xlabel, ylabel, title)
    src.metrics(data, centroids, labels)

   
if __name__ == "__main__":
    main()
