import random
import math
import numpy as np

def height(x, y, x0, y0, size, intensity):
    return math.sqrt(intensity)*math.pow(math.e, -intensity*((x/size-x0/size)**2+(y/size-y0/size)**2))

def createMap(size=100, mountains=20):
    tMap = [[0 for _ in range(size)] for __ in range(size)]
    peaks = []
    for _ in range(mountains):
        A = random.random()*40+10
        x0, y0 = random.randrange(0, size), random.randrange(0, size)
        for y in range(size):
            for x in range(size):
                tMap[y][x] += height(x, y, x0, y0, size, A)
        peaks.append(y0*size+x0)
    cleandata = np.array(tMap)
    noise = np.random.normal(0, 0.1, cleandata.shape)
    return cleandata + noise, peaks

def createSparseAdjacency(tMap):
    source = []
    target = []
    weights = []
    n = tMap.shape[0]
    for j in range(n):
        for i in range(n):
            for dj, di in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if 0 <= j+dj < n and 0 <= i+di < n:
                    source.append(j*n+i)
                    target.append((j+dj)*n+i+di)
                    weights.append(tMap[j][i]-tMap[j+dj][i+di])
    m = min(weights)
    for i in range(len(weights)):
        weights[i] -= m
    return weights, (source, target)

def createFeatures(tMap):
    features = []
    noise = np.random.normal(0, 0.1, tMap.shape)
    humidity = noise + 0.3*tMap
    n = tMap.shape[0]
    for j in range(n):
        for i in range(n):
            features.append([humidity[j][i]])
    return np.array(features)

def createLabels(peaks, n):
    labels = []
    for j in range(n):
        for i in range(n):
            if j*n+i in peaks:
                labels.append(1)
            else:
                labels.append(0)
    return np.array(labels)

if __name__ == "__main__":
    tMap, peaks = createMap(3, 2)
    print(tMap)
    print(peaks)
    weights, x = createSparseAdjacency(tMap)
    print(weights)
    print(createFeatures(tMap))