import numpy as np
import matplotlib.pyplot as plt
import random
from numpy.core.fromnumeric import size
from sklearn.datasets import make_blobs


class KohonenNN:
    def __init__(self, data, n_layers, layerSize, learningRate, epochs):
        self.learningRate = learningRate # темп обучения
        self.epochs = epochs # количество эпох обучения
        self.n_layers = n_layers # количество слоев
        self.layerSize = layerSize # размер слоя
        self.dim = len(data[0]) # размерность пространства
        self.map = self.initMap() # карта кохонена (матрица n_layers * layerSize)
        self.sigma = 7 # эффективная ширина
        self.tau_1 = 1000 / np.log(self.sigma) # для уменьшения топологической окрестности
        self.tau_2 = 1000 # для уменьшения темпа обучения
    
    # инициализация карты (задание весов нейронам)
    def initMap(self):
        matrix = list()
        for i in range(self.n_layers):
            layer = list()
            for j in range(self.layerSize):
                neuron = [random.uniform(-0.5,0.5) for k in range(self.dim)]
                layer.append(neuron)
            matrix.append(layer)
        return np.array(matrix)
                 
    # евклидово расстояние
    def euclideanDistance(self, x, y):
        return np.sqrt(sum([(x[i] - y[i])**2 for i in range(len(x))]))

    # латеральные расстояния
    def findLateralDistances(self, indeces):
        matrix = list()
        for i in range(self.n_layers):
            row = list()
            for j in range(self.layerSize):
                dist = self.euclideanDistance(indeces, [i,j])
                row.append(dist)
            matrix.append(row)
        return np.array(matrix)

    # топологическая окрестность
    def findTN(self, dist):
        matrix = list()
        for i in range(self.n_layers):
            row = list()
            for j in range(self.layerSize):
                h = np.exp(-dist[i][j]/(2*self.sigma**2))
                row.append(h)
            matrix.append(row)
        return np.array(matrix)
                

    # поиск ближайшего нейрона
    def findNearestNeuron(self, point):
        distances = list()
        for i in range(self.n_layers):
            for j in range(self.layerSize):
                distance = self.euclideanDistance(point, self.map[i][j])**2
                distances.append(distance)
        return distances.index(min(distances)) # 1-D index (need convert to 2-D coord for map)

    # обновление весов
    def updateMap(self, H, point):
        for i in range(self.n_layers):
            for j in range(self.layerSize):
                self.map[i][j] = self.map[i][j] + self.learningRate * H[i][j] * (point - self.map[i][j])
 
    # обучение
    def fit(self, data):
        for i in range(self.epochs):
            idx = random.randint(0, len(data)-1)
            point = data[idx]
            # процесс конкуренции
            indexOfBestNeuron = self.findNearestNeuron(point)
            layerNumber = indexOfBestNeuron // self.n_layers # номер слоя 
            neuronNumber = indexOfBestNeuron % self.layerSize # номер нейрона в слое
            # процесс кооперации
            coopDist = self.findLateralDistances([layerNumber, neuronNumber])
            H = self.findTN(coopDist)
            self.sigma = np.exp(-i/self.tau_1)
            # процесс адаптации
            self.updateMap(H, point)
            self.learningRate = np.exp(-i / self.tau_2)

    def predict(self):
        pass



def main():
    n_samples = 400 # размер обучающей выборки
    n_components = 5 # начальное количество кластеров
    n_layers = 4
    layerSize = 4
    lrate = 0.5
    epochs = 1500

    # генерируем кластеры
    X, y_true = make_blobs(n_samples=n_samples, centers=n_components, cluster_std=0.95, random_state=0)
    X = X[:, ::-1]
    plt.figure(1)
    colors = ["#fcc500", '#00fc89', '#ff68ed', '#ff713a', '#48aeff', '#c5ff1c']

    nn = KohonenNN(X, n_layers, layerSize, lrate, epochs)
    map = nn.map

    # Before learning NN      
    for i in range(len(X)):
        color = colors[y_true[i]]
        plt.scatter(X[i][0], X[i][1], color=colors[1], s=30)

    for layer in map:
        for neuron in layer:
            plt.scatter(neuron[0], neuron[1], color='#0600ed', s=100, marker='x')
    plt.show()



    nn.fit(X)
    
    # After learning NN
    for i in range(len(X)):
        color = colors[y_true[i]]
        plt.scatter(X[i][0], X[i][1], color=colors[1], s=30)

    for layer in map:
        for neuron in layer:
            plt.scatter(neuron[0], neuron[1], color='#0600ed', s=300, marker='x')


    plt.title("Cohonen's NN")
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
   
if __name__ == "__main__":
    main()