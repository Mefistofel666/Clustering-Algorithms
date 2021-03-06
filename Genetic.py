import numpy as np
import random
import src

class Genetic:
    def __init__(self, popSize, k, data, threshold = 0.9, maxIter = 70):
        self.populationSize = popSize
        self.k = k
        self.population = self.initPopulation(data)
        self.dimension = len(data[0])
        self.fitnesses = [0 for i in range(self.populationSize)]
        self.maxIterations = maxIter
        self.threshold = threshold
    
    # для каждой особи нужно пересчитать центры(изменить хромосомы)
    def calculateClustersCenters(self, data):
        for index, agent in enumerate(self.population):
            clusters = dict()
            for j in range(self.k):
                clusters[j] = []
            for point in data:
                distances = [self.euclideanDist(point, center) for center in agent]
                bestCluster = distances.index(min(distances))
                clusters[bestCluster].append(point)
            clusteringMetric = 0
            #  оформить в отдельную функцию надо бы
            for centroid in clusters:
                currentMetric = sum([self.euclideanDist(agent[centroid], point) for point in clusters[centroid]])
                clusteringMetric += currentMetric
            
            self.fitnesses[index] = 1.0 / clusteringMetric

    # евклидово расстояние
    def euclideanDist(self, x, y):
        return np.sqrt(sum([pow(x[i]-y[i],2) for i in range(len(x))]))

    # генерируем начальную популяцию из векторов с длиной равной количеству кластеров k
    def initPopulation(self, data):
        res = []
        for i in range(self.populationSize):
            tmp = [data[random.randint(0, len(data)-1)].tolist() for j in range(self.k)]
            res.append(tmp)
        return res

    # одноточечная мутация
    def mutation(self, agent):
        idx = random.randint(0, self.k-1)
        delta = random.random()
        flag = True if random.random() > 0.5 else False
        if flag:
            for j in range(self.dimension):
                agent[idx][j] = agent[idx][j] + agent[idx][j] * delta
        else:
            for j in range(self.dimension):
                agent[idx][j] = agent[idx][j] - agent[idx][j] * delta

    # Одноточечное скрещивание
    def onePointCross(self, indexOfAgentForCross):
        newPopulation = list()
        agentForCross = [self.population[i] for i in indexOfAgentForCross]
        chromosomeSize = self.k * self.dimension
        delimeter = random.randint(1, chromosomeSize - 2)

        for i in range(int(self.populationSize / 2)):
            firstParentIndex = random.randint(0, len(agentForCross)-1)
            secondParentIndex = random.randint(0, len(agentForCross)-1)

            while (firstParentIndex == secondParentIndex):
                firstParentIndex = random.randint(0, len(agentForCross) - 1)
                secondParentIndex = random.randint(0, len(agentForCross) - 1)
    
            firstParent = sum(agentForCross[firstParentIndex], []) 
            secondParent = sum(agentForCross[secondParentIndex], [])
            firstChild = firstParent[0:delimeter] + secondParent[delimeter:]
            secondChild = secondParent[0:delimeter] + firstParent[delimeter:]
            firstChild = [firstChild[i : i + self.dimension] for i in range(0, len(firstChild), self.dimension)]
            secondChild = [secondChild[i : i + self.dimension] for i in range(0, len(secondChild), self.dimension)]
            newPopulation.append(firstChild)
            newPopulation.append(secondChild)
        return newPopulation

        
    # рулеточный отбор особей для скрещивания (функция возваращает набор индексов особей для скрещивания)
    def proportionalSelection(self):
        bestAgents = list()
        s = sum(self.fitnesses)
        probs = [f / s for f in self.fitnesses]
        segments = list()
        segments.append([0, probs[0]])
        for i in range(1, len(probs)):
            segments.append([segments[i-1][1], segments[i-1][1] + probs[i]])
        for i in range(self.populationSize):
            rnd = random.random()
            for index, segment in enumerate(segments):
                if rnd >= segment[0] and rnd < segment[1]:
                    bestAgents.append(index)
        return bestAgents

    # процесс поиска
    def run(self, data):
        for i in range(self.maxIterations):
            self.calculateClustersCenters(data)
            agentForCross = self.proportionalSelection()
            newPopulation = self.onePointCross(agentForCross)
            for agent in newPopulation:
                flag = True if random.random() > self.threshold else False
                if flag:
                    self.mutation(agent)
            self.population = newPopulation
        best = self.getBestAgent()  
        return np.array(best), self.getClusters(best, data)

    # лучшая особь
    def getBestAgent(self):
        bestAgentIndex = self.fitnesses.index(max(self.fitnesses))
        return self.population[bestAgentIndex]

    # Присвоение точек ближайшим кластерам
    def getClusters(self, best, data):
        clusters = dict()
        for j in range(self.k):
                clusters[j] = []
        for point in data:
            distances = list()
            for centroid in best:
                distances.append(self.euclideanDist(centroid, point))
            bestCluster = distances.index(min(distances))
            clusters[bestCluster].append(point)
        return clusters


    def predict(self, point, centroids):
        distances = [np.linalg.norm(point - centroid) for centroid in centroids]
        classsification = distances.index(min(distances))
        return classsification


def main():
    data = src.X1
    xlabel, ylabel, title = 'Income', 'Score', 'KMeans'
    gen = Genetic(120, 5, data)
    centroids, clusters = gen.run(data)
    labels = [gen.predict(point, centroids) for point in data]
    src.plotClusters(data, labels, centroids, '2d', xlabel, ylabel, title)
    src.metrics(data, centroids, labels)


if __name__ == "__main__":
    main()



