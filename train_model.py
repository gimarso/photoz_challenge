import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from ML_models import PhotoZNet, DeltaZLoss

# --- CLASE DATASET ---
class PhotoZDataset(Dataset):
    def __init__(self, features, targets, types):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets).reshape(-1, 1)
        self.types = np.array(types) # Mantener como array de numpy strings para filtrado

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx], self.types[idx]

# --- FUNCIONES AUXILIARES ---
def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_mad(series):
    """Calcula la Median Absolute Deviation"""
    median = series.median()
    return (series - median).abs().median()

def preprocess_data(df, config, train_stats=None):
    """
    Selecciona variables, rellena NaNs y normaliza.
    Si train_stats se pasa, usa esas estadísticas (para validación/test).
    Si no, las calcula (para training).
    """
    
    # 1. Selección de columnas inputs
    selected_cols = []
    for group in config['data']['selected_features']:
        selected_cols.extend(config['data']['inputs'][group])
    
    X = df[selected_cols].copy()
    
    # Target (vamos a asumir que combinamos Z_GAL y Z_QSO en una columna 'Z' común si no existe)
    # En tu prompt anterior creamos columnas separadas, aquí unificamos para el target
    if 'Z' in df.columns:
        y = df['Z'].values
    else:
        # Fallback si Z está separado (ajustar según tu dataset real)
        y = np.where(df['SPECTYPE'] == 2.0, df['Z_QSO'], df['Z_GAL'])

    types = df['TYPE'].values

    # 2. Missing data (NaN -> 0.0)
    X = X.fillna(0.0)

    # 3. Normalización (Data - Median) / MAD
    cols_to_norm = []
    for group in config['data']['features_to_normalize']:
        if group in config['data']['selected_features']:
             cols_to_norm.extend(config['data']['inputs'][group])
    
    if train_stats is None:
        # Calcular estadísticas (Solo en Training)
        medians = X[cols_to_norm].median()
        mads = X[cols_to_norm].apply(get_mad)
        # Evitar división por cero
        mads = mads.replace(0, 1.0)
        train_stats = {'medians': medians, 'mads': mads}
    
    # Aplicar normalización
    X[cols_to_norm] = (X[cols_to_norm] - train_stats['medians']) / train_stats['mads']
    
    return X.values, y, types, train_stats, len(selected_cols)

# --- MAIN ---
def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # 1. Cargar Datos
    print("Cargando datos...")
    df_train = pd.read_hdf(cfg['data']['train_path'], key='data')
    df_val = pd.read_hdf(cfg['data']['val_path'], key='data')

    # 2. Preprocesamiento
    print("Preprocesando datos...")
    X_train, y_train, types_train, stats, input_dim = preprocess_data(df_train, cfg, train_stats=None)
    # Importante: Usar stats de train para validar
    X_val, y_val, types_val, _, _ = preprocess_data(df_val, cfg, train_stats=stats)

    # 3. DataLoaders
    train_dataset = PhotoZDataset(X_train, y_train, types_train)
    val_dataset = PhotoZDataset(X_val, y_val, types_val) # Necesario custom collate si types son strings? No, si no se empaquetan en tensor.
    
    # Nota: PyTorch DataLoader por defecto intenta convertir todo a tensor. 
    # Strings (types) darán error en el batch por defecto. 
    # Solución rápida: En el loop de validación pasamos el dataset completo o hacemos un collate custom.
    # Para simplificar este código preliminar, usaremos el DataLoader solo para X e y en training,
    # y para validación evaluaremos en tensores completos (si caben en memoria) o por batches con cuidado.
    
    train_loader = DataLoader(train_dataset, batch_size=cfg['data']['batch_size'], shuffle=True, drop_last=True)
    # Para validación, como necesitamos los TYPES para las métricas, iteraremos manualmente o usaremos batch size grande sin shuffle
    
    # 4. Modelo
    print(f"Inicializando modelo '{cfg['experiment']['group_name']}'...")
    model = PhotoZNet(
        input_size=input_dim,
        hidden_layers=cfg['model']['hidden_layers'],
        dropout_rates=cfg['model']['dropout_rates']
    ).to(device)

    # 5. Loss y Optimizador
    if cfg['training']['loss_type'] == 'deltaz':
        criterion = DeltaZLoss()
    else:
        criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])

    # 6. Bucle de Entrenamiento
    epochs = cfg['training']['epochs']
    print(f"Comenzando entrenamiento por {epochs} épocas.")

    for epoch in range(epochs):
        model.train()
        train_loss_acc = 0.0
        
        # Barra de progreso para Training
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        
        for inputs, targets, _ in pbar: # Ignoramos types en training loop
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss_acc += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss_acc / len(train_loader)

        # 7. Validación y Métricas por TYPE
        model.eval()
        val_loss_acc = 0.0
        
        # Para simplificar métricas por tipo, hacemos inferencia en todo el set de validación
        # (Si es muy grande, habría que hacerlo por batches y acumular)
        with torch.no_grad():
            val_inputs = torch.FloatTensor(X_val).to(device)
            val_targets = torch.FloatTensor(y_val).to(device)
            val_outputs = model(val_inputs)
            
            # Loss Global
            total_val_loss = criterion(val_outputs, val_targets).item()
            
            # Métricas por TYPE
            val_preds_np = val_outputs.cpu().numpy().flatten()
            val_targets_np = val_targets.cpu().numpy().flatten()
            
            unique_types = np.unique(types_val)
            type_metrics = {}
            
            for t in unique_types:
                mask = (types_val == t)
                if np.sum(mask) > 0:
                    # Calcular loss específica usando la clase de Loss manualmente o re-usando criterion
                    p_t = val_outputs[mask]
                    t_t = val_targets[mask]
                    loss_t = criterion(p_t, t_t).item()
                    type_metrics[t] = loss_t

        # Imprimir resultados de la época
        msg = f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Loss Global: {total_val_loss:.4f}"
        for t, l in type_metrics.items():
            msg += f" | {t}: {l:.4f}"
        print(msg)

    # 8. Guardar Modelo
    os.makedirs(cfg['experiment']['save_dir'], exist_ok=True)
    save_path = os.path.join(cfg['experiment']['save_dir'], f"{cfg['experiment']['group_name']}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\nModelo guardado en: {save_path}")

if __name__ == "__main__":
    main()
