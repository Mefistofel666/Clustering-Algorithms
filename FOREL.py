import numpy as np
import src


def get_centroid(points):
    return np.array([np.mean(points[:, 0]), np.mean(points[:, 1])])


class FOREL:
    def __init__(self, d, r, tol=1e-1):
        self.radius = r
        self.tol = tol
        self.centroids = []
        self.data = d

    def fit(self):
        while len(self.data) != 0:
            current_point = self.get_random_point(self.data)
            neighbors = self.get_neighbors(current_point, self.data)
            centroid = get_centroid(neighbors)
            while np.linalg.norm(current_point - centroid) > self.tol:
                current_point = centroid
                neighbors = self.get_neighbors(current_point, self.data)
                centroid = get_centroid(neighbors)
            self.data = self.remove_points(neighbors, self.data)
            self.centroids.append(current_point)
        return self.centroids

    def get_neighbors(self, p, points):
        neighbors = [point for point in points if np.linalg.norm(p - point) < self.radius]
        return np.array(neighbors)

    def get_random_point(self, points):
        random_index = np.random.choice(len(points), 1)[0]
        return points[random_index]

    def remove_points(self, subset, points):
        points = [p for p in points if p not in subset]
        return points

    def predict(self, point):
        distances = [np.linalg.norm(point - centroid) for centroid in self.centroids]
        classsification = distances.index(min(distances))
        return classsification



data = src.X1
xlabel, ylabel, title = 'Income', 'Score', 'Hierarchical'
frl = FOREL(data, r=40)
frl.fit()
centroids = frl.centroids
labels = [frl.predict(point) for point in data]
src.plotClusters(data, labels, centroids, '2d', xlabel, ylabel, title)
src.metrics(data, centroids, labels)


