import os

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from ML_models import PhotoZNet, DeltaZLoss, RandomForestPhotoZ

# Custom Dataset class to handle features, targets, and object types
class PhotoZDataset(Dataset):
    def __init__(self, features, targets, types):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets).reshape(-1, 1)
        self.types = np.array(types)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx], self.types[idx]

# Helper function to load configuration from a YAML file
def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Helper function to compute the Median Absolute Deviation (MAD)
def get_mad(series):
    median = series.median()
    return (series - median).abs().median()

# Function to select features, handle missing values, and normalize data
def preprocess_data(df, config, train_stats=None):
    selected_cols = []
    for group in config['data']['selected_features']:
        selected_cols.extend(config['data']['inputs'][group])
    
    X = df[selected_cols].copy()
    
    # Extract the target variable Z 
    # (Since Z_GAL and Z_QSO are dropped, we solely rely on Z)
    y = df['Z'].values
    types = df['TYPE'].values

    # Replace NaN values with 0.0
    X = X.fillna(0.0)

    # Identify columns that need normalization
    cols_to_norm = []
    for group in config['data']['features_to_normalize']:
        if group in config['data']['selected_features']:
             cols_to_norm.extend(config['data']['inputs'][group])
    
    # Compute median and MAD only for the training set to prevent data leakage
    if train_stats is None:
        medians = X[cols_to_norm].median()
        mads = X[cols_to_norm].apply(get_mad)
        
        # Prevent division by zero
        mads = mads.replace(0, 1.0)
        train_stats = {'medians': medians, 'mads': mads}
    
    # Apply normalization using the training statistics
    X[cols_to_norm] = (X[cols_to_norm] - train_stats['medians']) / train_stats['mads']
    
    return X.values, y, types, train_stats, len(selected_cols)

# Main function to train and validate the model
def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Load training and validation data from HDF5 files
    print("Cargando datos...")
    df_train = pd.read_hdf(cfg['data']['train_path'], key='data')
    df_val = pd.read_hdf(cfg['data']['val_path'], key='data')

    # Preprocess datasets, ensuring validation data uses training statistics
    print("Preprocesando datos...")
    X_train, y_train, types_train, stats, input_dim = preprocess_data(df_train, cfg, train_stats=None)
    X_val, y_val, types_val, _, _ = preprocess_data(df_val, cfg, train_stats=stats)




    model_type = cfg['model'].get('type', 'neural_net')

    if model_type == 'neural_net':
        # Initialize DataLoaders for batch processing
        train_dataset = PhotoZDataset(X_train, y_train, types_train)
        val_dataset = PhotoZDataset(X_val, y_val, types_val)
        
        train_loader = DataLoader(train_dataset, batch_size=cfg['data']['batch_size'], shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg['data']['batch_size'], shuffle=False)
        
        # Instantiate the neural network model
        print(f"Inicializando modelo '{cfg['experiment']['group_name']}'...")
        model = PhotoZNet(
            input_size=input_dim,
            hidden_layers=cfg['model']['hidden_layers'],
            dropout_rates=cfg['model']['dropout_rates']
        ).to(device)

        # Set the loss function and optimizer
        if cfg['training']['loss_type'] == 'deltaz':
            criterion = DeltaZLoss()
        else:
            criterion = nn.MSELoss()
        
        optimizer = optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])

        # Training loop over the specified number of epochs
        epochs = cfg['training']['epochs']
        print(f"Comenzando entrenamiento por {epochs} épocas.")

        for epoch in range(epochs):
            model.train()
            train_loss_acc = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
            
            # Iterate over batches and update model weights
            for inputs, targets, _ in pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss_acc += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            avg_train_loss = train_loss_acc / len(train_loader)

            # Evaluate the model on the validation set
            model.eval()
            val_loss_acc = 0.0
            all_val_outputs = []
            all_val_targets = []
            all_val_types = []
            
            with torch.no_grad():
                for v_inputs, v_targets, v_types in val_loader:
                    v_inputs, v_targets = v_inputs.to(device), v_targets.to(device)
                    
                    v_outputs = model(v_inputs)
                    v_loss = criterion(v_outputs, v_targets)
                    
                    val_loss_acc += v_loss.item()
                    
                    # Store predictions and targets for type-specific metrics
                    all_val_outputs.append(v_outputs.cpu())
                    all_val_targets.append(v_targets.cpu())
                    all_val_types.extend(v_types)
                
                total_val_loss = val_loss_acc / len(val_loader)
                val_outputs_cat = torch.cat(all_val_outputs)
                val_targets_cat = torch.cat(all_val_targets)
                val_types_np = np.array(all_val_types)
                
                # Calculate metrics individually for GALAXY and QSO types dynamically
                unique_types = np.unique(val_types_np)
                type_metrics = {}
                
                for t in unique_types:
                    mask = (val_types_np == t)
                    if np.sum(mask) > 0:
                        p_t = val_outputs_cat[mask]
                        t_t = val_targets_cat[mask]
                        loss_t = criterion(p_t, t_t).item()
                        type_metrics[t] = loss_t

            # Output the epoch summary
            msg = f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Loss Global: {total_val_loss:.4f}"
            for t, l in type_metrics.items():
                msg += f" | {t}: {l:.4f}"
            print(msg)

        # Save the trained model weights to the specified directory
        os.makedirs(cfg['experiment']['save_dir'], exist_ok=True)
        save_path = os.path.join(cfg['experiment']['save_dir'], f"{cfg['experiment']['group_name']}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"\nModelo guardado en: {save_path}")

    elif model_type == 'random_forest':

        # Sample 25% of the training data for speed
        n_sample = max(1, int(0.1 * X_train.shape[0]))
        sample_indices = np.random.choice(X_train.shape[0], size=n_sample, replace=False)
        X_train = X_train[sample_indices]
        y_train = y_train[sample_indices]
        types_train = types_train[sample_indices]

        print(f"Inicializando modelo Random Forest '{cfg['experiment']['group_name']}'...")
        rf_model = RandomForestPhotoZ(
            n_estimators=cfg['model'].get('n_estimators', 100),
            max_depth=cfg['model'].get('max_depth', None)
        )
        
        # Train the Random Forest
        print("Entrenando Random Forest...")
        rf_model.fit(X_train, y_train)
        
        # Evaluate on the validation set
        print("Evaluando Random Forest en validacion...")
        mean_pred, std_pred = rf_model.predict(X_val)
        
        # Calculate DeltaZLoss manually for validation set
        y_val_flat = y_val.ravel()
        numerator = np.abs(mean_pred - y_val_flat)
        denominator = 1.0 + y_val_flat
        total_val_loss = np.mean(numerator / denominator)
        
        # Calculate metrics individually for types
        unique_types = np.unique(types_val)
        type_metrics = {}
        for t in unique_types:
            mask = (types_val == t)
            if np.sum(mask) > 0:
                loss_t = np.mean(np.abs(mean_pred[mask] - y_val_flat[mask]) / (1.0 + y_val_flat[mask]))
                type_metrics[t] = loss_t
        
        # Output summary metrics
        msg = f"Val Loss Global: {total_val_loss:.4f}"
        for t, l in type_metrics.items():
            msg += f" | {t}: {l:.4f}"
        print(msg)
        
        # Display a small sample of the calculated uncertainty
        print("\nSample uncertainty (std dev of tree predictions) for first 5 val items:")
        for i in range(5):
            print(f"True Z: {y_val_flat[i]:.4f} | Pred Z: {mean_pred[i]:.4f} | Uncertainty: {std_pred[i]:.4f}")
        
        # Save the Random Forest model
        os.makedirs(cfg['experiment']['save_dir'], exist_ok=True)
        save_path = os.path.join(cfg['experiment']['save_dir'], f"{cfg['experiment']['group_name']}_rf.joblib")
        rf_model.save(save_path)
        print(f"\nModelo guardado en: {save_path}")

if __name__ == "__main__":
    main()