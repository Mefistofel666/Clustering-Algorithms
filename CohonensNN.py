import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.datasets import make_blobs


class CohonenNN:
    def __init__(self, data, numberOfNeurons, learningRate, epochs):
        self.numberOfNeurons = numberOfNeurons
        self.dimension = len(data[0])
        self.learningRate = learningRate
        self.epochs = epochs
        self.neurons = self.initMatrixOfWeights()
        self.matrixOfDistances = self.initMatrixOfDistances()
        self.sigma = 4
    def initMatrixOfWeights(self):
        matrix = list()
        for i in range(self.numberOfNeurons):
            w = [random.uniform(-1,1) for j in range(self.dimension)]
            list.append(w)
        return np.array(matrix)
    def initMatrixOfDistances(self):
        matrix = list()
        for i in range(self.numberOfNeurons):
            distances = [self.euclideanDistance(i,j) for j in range(self.numberOfNeurons)]
            matrix.append(distances)
        return matrix
                 

    def euclideanDistance(self, x, y):
        return np.sqrt(sum([(x[i] - y[i])**2 for i in range(len(x))]))

    def findNearestNeuron(self, point):
        distances = [self.euclideanDistance(point, neuron) for neuron in self.neurons]
        return distances.index(min(distances))
    def fit(self, data):
        for i in range(self.epochs):
            order = random.shuffle([j for j in range(len(data))])
            for idx in order:
                x = data[idx]
                indexOfNearestNeuron = self.findNearestNeuron()
                h = [np.exp(self.matrixOfDistances[indexOfNearestNeuron][j]/(self.sigma**2)) for j in range(self.numberOfNeurons)]
                for index, neuron in enumerate(self.neurons):
                    neuron = neuron + self.learningRate * h[index] * (x-neuron)
            self.learningRate *= 0.95
            self.sigma *= 0.97

        return 

        
    def predict(self):
        pass



def main():
    n_samples = 400 # размер обучающей выборки
    n_components = 5 # начальное количество кластеров

    # генерируем кластеры
    X, y_true = make_blobs(n_samples=n_samples, centers=n_components, cluster_std=0.95, random_state=0)
    X = X[:, ::-1]
    plt.figure(1)
    colors = ["magenta", 'blue', 'red', 'yellow', 'orange']

    



    plt.title("Cohonen's NN")
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
   
if __name__ == "__main__":
    main()