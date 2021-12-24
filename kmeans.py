import random
import numpy as np
import src



class KMEANS:
    def __init__(self, k=5, max_iterations=300):
        self.k = k
        self.max_iterations = max_iterations

    def fit(self, data):
        self.centroids = {}

        # инициализируем центроиды
        for i in range(self.k):
            self.centroids[i] = data[random.randint(0, len(data)-1)]

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
                self.centroids[cluster] = np.average(self.clusters[cluster], axis=0)

        # добавить условие останова : максимальное расстояние от новых центров до старых меньше эпсилон

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classsification = distances.index(min(distances))
        return classsification


def main():
    data = src.X2
    xlabel, ylabel, title = 'Income', 'Score', 'KMeans'
    kmeans = KMEANS(4)
    kmeans.fit(data)
    labels = [kmeans.predict(data[i]) for i in range(len(data))]
    centroids = [kmeans.centroids[i] for i in range(len(kmeans.centroids))]
    src.plotClusters(data, labels, centroids, '2d', xlabel, ylabel, title)
    src.metrics(data, centroids, labels)


if __name__ == "__main__":
    main()
