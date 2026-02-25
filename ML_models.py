import os

import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

class PhotoZNet(nn.Module):
    def __init__(self, input_size, hidden_layers, dropout_rates):
        super(PhotoZNet, self).__init__()
        
        layers = []
        in_dim = input_size
        
        # Dynamic construction of hidden layers
        for h_dim, drop_rate in zip(hidden_layers, dropout_rates):
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(drop_rate))
            in_dim = h_dim
            
        # Linear output layer
        layers.append(nn.Linear(in_dim, 1))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class DeltaZLoss(nn.Module):
    def __init__(self):
        super(DeltaZLoss, self).__init__()
        
    def forward(self, y_pred, y_true):
        # Standard Photo-Z metric: |dz| / (1+z)
        numerator = torch.abs(y_pred - y_true)
        denominator = 1.0 + y_true
        return torch.mean(numerator / denominator)

class RandomForestPhotoZ:
    def __init__(self, n_estimators=100, max_depth=None):
        # Initialize the scikit-learn random forest
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=42
        )

    def fit(self, X, y):
        self.model.fit(X, y.ravel())

    def predict(self, X):
        # Calculate predictions from all individual trees to estimate uncertainty
        preds = np.array([tree.predict(X) for tree in self.model.estimators_])
        
        # Mean prediction across all trees
        mean_pred = np.mean(preds, axis=0)
        
        # Standard deviation across trees serves as our uncertainty measure
        std_pred = np.std(preds, axis=0) 
        
        return mean_pred, std_pred

    def save(self, path):
        joblib.dump(self.model, path)
        
    def load(self, path):
        self.model = joblib.load(path)