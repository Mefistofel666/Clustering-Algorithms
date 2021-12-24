import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score
import plotly.graph_objects as go
import plotly

def wss(data, centroids, labels):
    s = 0
    for i in range(len(data)):
        s += np.linalg.norm(data[i] - centroids[labels[i]]) ** 2
    return s


def bss(centroids, data):
    s = data[0]
    for i in range(1, len(data)):
        s += data[i]
    s = [np.array(s[i])/len(data) for i in range(len(s))]
    return sum([np.linalg.norm(centroids[i] - s) ** 2 for i in range(len(centroids))])


colors = ["#fcc500", '#00fc89', '#ff68ed', '#ff713a', '#48aeff', '#c5ff1c']


def plot2Dclusters(data, labels, centroids, xlabel, ylabel, title):
    for idx, point in enumerate(data):
        color = colors[labels[idx]]
        plt.scatter(point[0], point[1], s=100, c=color)
    for centroid in centroids:
        plt.scatter(centroid[0], centroid[1], s=200, c='#0600ed', marker='x')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot3Dclusters(data, labels):
    fig = go.Figure(data=[go.Scatter3d(
        x=data[:, 0],
        y=data[:, 1],
        z=data[:, 2],
        mode='markers',
        marker=dict(
            size=12,
            color=labels,  # set color to an array/list of desired values
            colorscale='Viridis',  # choose a colorscale
            opacity=0.8
        )
    )])
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    plotly.offline.plot(fig)


def plotClusters(data, labels, centroids, t, xlabel, ylabel, title):
    if t != '3d':
        plot2Dclusters(data, labels, centroids, xlabel, ylabel, title)
    else:
        plot3Dclusters(data, labels)


def metrics(data, centroids, labels):
    w = wss(data, centroids, labels)
    b = bss(centroids, data)
    d = davies_bouldin_score(data, labels)
    print(f'wss = {w}')
    print(f'bss = {b}')
    print(f'dbs = {d}')


df = pd.read_csv('data.csv')
# income-score
X1 = df.iloc[:, [3, 4]].values
X2 = df.iloc[:, [2, 4]].values
X3 = df.iloc[:, [2, 3, 4]].values


