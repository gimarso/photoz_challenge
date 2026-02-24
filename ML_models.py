import os
import torch
import torch.nn as nn

class PhotoZNet(nn.Module):
    def __init__(self, input_size, hidden_layers, dropout_rates):
        super(PhotoZNet, self).__init__()
        
        layers = []
        in_dim = input_size
        
        # Construcción dinámica de capas oculta
        for h_dim, drop_rate in zip(hidden_layers, dropout_rates):
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(drop_rate))
            in_dim = h_dim
            
        # Capa de salida lineal
        layers.append(nn.Linear(in_dim, 1))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class DeltaZLoss(nn.Module):
    def __init__(self):
        super(DeltaZLoss, self).__init__()
        
    def forward(self, y_pred, y_true):
        # Métrica estándar en Photo-Z: |dz| / (1+z)
        numerator = torch.abs(y_pred - y_true)
        denominator = 1.0 + y_true
        return torch.mean(numerator / denominator)
