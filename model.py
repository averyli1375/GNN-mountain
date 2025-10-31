from tools import *
from sknetwork.gnn.gnn_classifier import GNNClassifier
from scipy.sparse import csr_matrix

def createDataset(size, mountains):
    tMap, peaks = createMap(size, mountains)
    print(tMap)
    adjacency = csr_matrix(*createSparseAdjacency(tMap), shape=(size*size, size*size))
    features = createFeatures(tMap)
    labels = createLabels(peaks, size)
    return adjacency, features, labels

def evalModel():
    size = 10
    mountains = 40
    adjacency, features, labels = createDataset(size, mountains)

    model = GNNClassifier(dims = [32, 2], learning_rate=1e-2,verbose = True)
    labels_pred = model.fit_predict(adjacency, features, labels, 
                                    n_epochs=2000,
                                    random_state=42)
    print(f"Train accuracy: {float(np.mean(labels_pred == labels))}")
    print(sum(labels_pred))
    model(*createDataset(size, mountains))
    #test_pred = model.predict(*testSet)
    #print(f"Test accuracy: {float(round(np.mean(testSet[2] == test_pred), 2))}")

if __name__ == "__main__":
    evalModel()