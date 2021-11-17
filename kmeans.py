from math import dist
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.datasets import make_blobs

class KMEANS:
    def __init__(self, k = 3, max_iterations = 1500):
        self.k = k
        self.max_iterations = max_iterations

    def fit(self, data):
        self.centroids = {}

        # инициализируем центроиды
        for i in range(self.k):
            self.centroids[i] = data[random.randint(0,len(data))]

        # делаем max_iterations итераций поиска
        for i in range(self.max_iterations):
            self.clusters = {}
            for j in range(self.k):
                self.clusters[j] = []
        
            # найдем расстояния от точек до центроидов
            for dot in data:
                distances = [np.linalg.norm(dot - self.centroids[centroid]) for centroid in self.centroids]
                bestCluster = distances.index(min(distances))
                self.clusters[bestCluster].append(dot)

            # пересчитаем центроиды(центр масс)
            for cluster in self.clusters:
                self.centroids[cluster] = np.average(self.clusters[cluster], axis = 0)

        # добавить условие останова : максимальное расстояние от новых центров до старых меньше эпсилон

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classsification = distances.index(min(distances))
        return classsification
            
def main():
    n_samples = 500 # размер обучающей выборки
    n_components = 3 # начальное количество кластеров

    # генерируем кластеры
    X, y_true = make_blobs(n_samples=n_samples, centers=n_components, cluster_std=0.75, random_state=0)
    X = X[:, ::-1]
    plt.figure(1)
    colors = ["m", 'blue', 'red', 'green']

    kmeans = KMEANS(n_components)
    kmeans.fit(X)
    for cluster in kmeans.clusters:
	    color = colors[cluster]
	    for features in kmeans.clusters[cluster]:
		    plt.scatter(features[0], features[1], color = color,s = 30)
	
    for centroid in kmeans.centroids:
	    plt.scatter(kmeans.centroids[centroid][0], kmeans.centroids[centroid][1], color='green', s = 130, marker = "x")

    plt.title("K-Means")
    plt.xticks([])
    plt.yticks([])
    plt.show()

if __name__ == "__main__":
    main()
