import os

import yaml
import torch
import numpy as np
import pandas as pd
from ML_models import PhotoZNet, RandomForestPhotoZ

# Helper function to load configuration
def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Helper function to calculate Median Absolute Deviation for robust scaling
def get_mad(series):
    median = series.median()
    return (series - median).abs().median()

# Prepares features for inference without requiring target columns (Z or TYPE)
def preprocess_blind_data(df, config, train_stats):
    selected_cols = []
    for group in config['data']['selected_features']:
        selected_cols.extend(config['data']['inputs'][group])
    
    X = df[selected_cols].copy()
    
    # Fill missing values with 0.0
    X = X.fillna(0.0)

    cols_to_norm = []
    for group in config['data']['features_to_normalize']:
        if group in config['data']['selected_features']:
             cols_to_norm.extend(config['data']['inputs'][group])
    
    # Apply normalization using the provided training statistics
    X[cols_to_norm] = (X[cols_to_norm] - train_stats['medians']) / train_stats['mads']
    
    return X.values, len(selected_cols)

def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dynamic paths using config to avoid hardcoded absolute paths
    train_path = cfg['data']['train_path']
    
    # Infer the data directory from the train_path to locate the blind test set
    data_dir = os.path.dirname(train_path)
    blind_test_path = os.path.join(data_dir, 'blind_test_set.h5')
    
    print("Loading training data to calculate normalization statistics...")
    df_train = pd.read_hdf(train_path, key='data')
    
    # Calculate normalization statistics from training data
    cols_to_norm = []
    for group in cfg['data']['features_to_normalize']:
        if group in cfg['data']['selected_features']:
             cols_to_norm.extend(cfg['data']['inputs'][group])
             
    X_train_norm_base = df_train[cols_to_norm].fillna(0.0)
    medians = X_train_norm_base.median()
    mads = X_train_norm_base.apply(get_mad).replace(0, 1.0)
    train_stats = {'medians': medians, 'mads': mads}
    
    # Free up memory before loading the blind test set
    del df_train, X_train_norm_base

    print(f"Loading blind test data from {blind_test_path}...")
    df_blind = pd.read_hdf(blind_test_path, key='data')
    
    print("Pre-processing blind test data...")
    X_blind, input_dim = preprocess_blind_data(df_blind, cfg, train_stats)
    
    # Extract the group name and model type to correctly load the model
    group_name = cfg['experiment']['group_name']
    model_type = cfg['model'].get('type', 'neural_net')
    
    print(f"Initializing {model_type} model...")
    
    if model_type == 'neural_net':
        model = PhotoZNet(
            input_size=input_dim,
            hidden_layers=cfg['model']['hidden_layers'],
            dropout_rates=cfg['model']['dropout_rates']
        ).to(device)
        
        model_path = os.path.join(cfg['experiment']['save_dir'], f"{group_name}.pth")
        print(f"Loading weights from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        print("Running inference...")
        with torch.no_grad():
            inputs_tensor = torch.FloatTensor(X_blind).to(device)
            preds = model(inputs_tensor).cpu().numpy().flatten()
            # Neural networks do not output standard deviation in this setup
            preds_std = np.full(preds.shape, np.nan)
            
    elif model_type == 'random_forest':
        model = RandomForestPhotoZ()
        model_path = os.path.join(cfg['experiment']['save_dir'], f"{group_name}_rf.joblib")
        print(f"Loading weights from {model_path}...")
        model.load(model_path)
        
        print("Running inference...")
        preds, preds_std = model.predict(X_blind)

    # Create an output dataframe containing TARGETID, predicted Z, and uncertainty
    print("Preparing predictions for submission...")
    output_df = pd.DataFrame({
        'TARGETID': df_blind['TARGETID'],
        'Z_PRED': preds,
        'Z_PRED_STD': preds_std
    })
    
    # Save the results to the dynamic data folder including the group_name
    output_filename = f"predictions_{group_name}.csv"
    output_path = os.path.join(data_dir, output_filename)
    
    output_df.to_csv(output_path, index=False)
    print(f"Predictions successfully saved to: {output_path}")

if __name__ == "__main__":
    main()