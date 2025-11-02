# Training a GNN to Recognize Mountain Peaks, a.k.a a shakespearean tragedy in parts
The goal of this onboarding project is to train a graph neural network to identify mountain peaks on synthetic data using Scikit-Learn. The mountains are synthetically generated, and all data generation code is contained within the tools.py file

## Varied Hyperparameters
The inherent problem with these datasets is that the number of peaks is so much less than that which is not peaks, so most models eventually learn to simply classify everything as "not peak", which still yields a fairly decent accuracy. Therefore, the metric of interest to be improved is recall, which measures the percentage of true peaks that we actually labelled as peaks. However, precision is still important as well, as we can't mindlessly label every point as a peak.
The baseline model evaluates on a test set as:
```
Grid Size: 20x20; Number of Mountains: 20
Accuracy: 0.949999988079071
F1 Score: 0.0
Precision: 0.0
Recall: 0.0
Accuracy if guessed all 0s: 0.95
```
### Model Size/Epoch Number
Model complexity can be improved by both adding nodes and increasing the depth of the model is a way to learn more information and prevent underfitting. Increasing number of epochs gives the model a chance to learn more, but also runs the risk of overfitting, so both metrics must be balanced to reach optimal state.
However, changing either parameter does nothing for precision or recall, so both are not worth considering for now.

### Weighting Scheme
Overweighting the minority class during training, especially with cross entropy loss, presents a unique approach at tackling the 0 recall problem. By counting the number of positive labels (peaks) and negative labels we weight the cross entropy loss with 
```
weight = torch.tensor([1.0, (num_zeros/num_ones)**0.5], dtype=torch.float)
```
Trying this weighting scheme yields
```
Grid Size: 10x10; Number of Mountains: 40
Accuracy: 0.6399999856948853
F1 Score: 0.5813953280448914
Precision: 0.4901960790157318
Recall: 0.7142857313156128
Accuracy if guessed all 0s: 0.6
_____________________________
Grid Size: 20x20; Number of Mountains: 20
Accuracy: 0.9375
F1 Score: 0.07407407462596893
Precision: 0.125
Recall: 0.05263157933950424
Accuracy if guessed all 0s: 0.95
```
For two differently sized test sets, for sake of scalability. We see in the case of the 10x10, the model begins predicting peaks too much, with high recall but low precision. This can be adjusted with the exponent in the weighting scheme. Increasing the exponent yields very high recall but very low accuracy/precision, and the best number for the exponent is still yet to be determined.

### Graph Weighting
Both sci-kit and pytorch don't take negative weights as values, so currently all weights are calculated, then the smallest weight is subtracted from each one, so all are non-negative. Taking the absolute value also is another solution, though does not seem to be performing better.
