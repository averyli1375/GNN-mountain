import torch
import numpy as np
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from tools import *

from torch_geometric.data import Data

def createDataset(size, mountains):
    tMap, peaks = createMap(size, mountains)
    adjacency = createSparseAdjacency(tMap)
    x = torch.tensor(createFeatures(tMap), dtype=torch.float)
    labels = createLabels(peaks, size)
    edge_index = torch.tensor([*adjacency[-1]], dtype=torch.long)
    edge_weight = torch.tensor(adjacency[0], dtype=torch.float)
    data = Data(x=x, y=torch.tensor(labels), edge_index=edge_index, edge_weight=edge_weight)
    return data


class WeightedGCN(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden_feats)
        self.conv2 = GCNConv(hidden_feats, out_feats)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x


data = createDataset(10, 40)

num_zeros = (data.y == 0).sum().item()
num_ones = (data.y == 1).sum().item()
weight = torch.tensor([1.0, num_zeros*1.4 / num_ones], dtype=torch.float)

model = WeightedGCN(1, 16, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_weight)
    loss = F.cross_entropy(out, data.y, weight=weight)
    loss.backward()
    optimizer.step()
    if epoch%10 == 0:
        print(f"Epoch: {epoch}; Loss: {loss}")


from torchmetrics import Accuracy, F1Score, Precision, Recall
def evalMetrics(out, y):
    print(out)
    preds = torch.sigmoid(out).argmax(dim=1)
    print(preds)
    print(y)
    acc = Accuracy(task="binary")
    f1 = F1Score(task="binary")
    precision = Precision(task="binary")
    recall = Recall(task="binary")
    print(f"Accuracy: {acc(preds, y)}")
    print(f"F1 Score: {f1(preds, y)}")
    print(f"Precision: {precision(preds, y)}")
    print(f"Recall: {recall(preds, y)}")


# Evaluate
model.eval()
testData = createDataset(10, 10)
pred = model(testData.x, testData.edge_index, testData.edge_weight)
evalMetrics(pred, testData.y)