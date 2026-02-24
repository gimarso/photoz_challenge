import os
import yaml
import argparse
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import gaussian_kde
from ML_models import PhotoZNet

# --- 1. CONFIGURACIÓN DE ENTORNO ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# --- 2. PLOTTING STYLE ---
plt.style.use('dark_background')
plt.rcParams.update({
    'font.size': 14,          
    'axes.labelsize': 18,     
    'axes.titlesize': 20,     
    'xtick.labelsize': 14,    
    'ytick.labelsize': 14,    
    'legend.fontsize': 14,    
    'lines.linewidth': 3      
})

# --- 3. CONSTANTS & CONFIG ---
EW_EMISSION_LINES = [
    'OII_3729_EW', 'OIII_5007_EW', 'NII_6584_EW', 'HBETA_4861_EW', 
    'HALPHA_6562_EW', 'MGII_2796_EW', 'CIV_1549_EW', 'LYALPHA_EW'
]

TYPE_COLORS = {
    'GALAXY_ID': 'tab:cyan',
    'QSO': 'tab:red',
    'GALAXY_OOD1': 'yellow',
    'GALAXY_OOD2': 'gold',
    'Passive': 'tab:orange',
    'ELG': 'tab:green'
}

# --- 4. HELPER FUNCTIONS & PROCESSING ---

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_mad(series):
    """Calculates Median Absolute Deviation"""
    median = series.median()
    return (series - median).abs().median()

def get_point_density(x, y):
    """
    Calculates point density for coloring scatter plots.
    """
    xy = np.vstack([x, y])
    # Subsample for speed if too many points
    if xy.shape[1] > 5000:
        indices = np.random.choice(xy.shape[1], 5000, replace=False)
        z = gaussian_kde(xy[:, indices])(xy)
    else:
        z = gaussian_kde(xy)(xy)
    
    idx = z.argsort()
    return x[idx], y[idx], z[idx]

def preprocess_data(df, config, train_stats=None):
    """
    Exact replication of training preprocessing logic.
    """
    # 1. Select inputs
    selected_cols = []
    for group in config['data']['selected_features']:
        selected_cols.extend(config['data']['inputs'][group])
    
    X = df[selected_cols].copy()
    
    # Target (Combine Z if necessary)
    if 'Z' in df.columns:
        y = df['Z'].values
    else:
        y = np.where(df['SPECTYPE'] == 2.0, df['Z_QSO'], df['Z_GAL'])

    types = df['TYPE'].values

    # 2. Missing data (NaN -> 0.0) BEFORE calculating stats
    X = X.fillna(0.0)

    # 3. Normalization
    cols_to_norm = []
    for group in config['data']['features_to_normalize']:
        if group in config['data']['selected_features']:
             cols_to_norm.extend(config['data']['inputs'][group])
    
    if train_stats is None:
        # Calculate stats (Only when passing Training set)
        medians = X[cols_to_norm].median()
        mads = X[cols_to_norm].apply(get_mad)
        mads = mads.replace(0, 1.0)
        train_stats = {'medians': medians, 'mads': mads}
    
    # Apply normalization
    X[cols_to_norm] = (X[cols_to_norm] - train_stats['medians']) / train_stats['mads']
    
    return X.values, y, types, train_stats, len(selected_cols)

def compute_metrics_binned(df, x_col, z_true_col='Z_TRUE', z_pred_col='Z_PRED', bins=None):
    """
    Computes Bias, Sigma_NMAD, and Outlier Fraction in bins.
    """
    if bins is None:
        return None, None, None, None

    centers = []
    bias_list = []
    sigma_list = []
    outlier_list = []

    # Normalized Delta Z
    dz = (df[z_pred_col] - df[z_true_col]) / (1 + df[z_true_col])
    
    for i in range(len(bins) - 1):
        mask = (df[x_col] >= bins[i]) & (df[x_col] < bins[i+1])
        subset_dz = dz[mask]
        
        if len(subset_dz) < 100: # Min objects per bin
            centers.append(np.nan)
            bias_list.append(np.nan)
            sigma_list.append(np.nan)
            outlier_list.append(np.nan)
            continue
            
        centers.append(0.5 * (bins[i] + bins[i+1]))
        
        # Bias
        bias_list.append(np.median(subset_dz))
        
        # Sigma NMAD
        nmad = 1.4826 * np.median(np.abs(subset_dz - np.median(subset_dz)))
        sigma_list.append(nmad)
        
        # Outlier Fraction (> 0.15)
        outlier_frac = np.sum(np.abs(subset_dz) > 0.15) / len(subset_dz)
        outlier_list.append(outlier_frac)

    return np.array(centers), np.array(bias_list), np.array(sigma_list), np.array(outlier_list)

def prepare_inference_data(config):
    """
    Loads Train to get correct stats, then processes TEST set for inference.
    """
    print("Loading datasets...")
    # Load Training just for normalization stats
    df_train = pd.read_hdf(config['data']['train_path'], key='data')
    
    # --- CHANGE: Load TEST set explicitly ---
    # Assuming test_set.h5 is in the same dir as training_set.h5
    train_dir = os.path.dirname(config['data']['train_path'])
    test_path = os.path.join(train_dir, "test_set.h5")
    
    print(f"Loading Test Set from: {test_path}")
    df_test = pd.read_hdf(test_path, key='data')

    print("Calculating normalization statistics from Training Set...")
    # Step 1: Process Train ONLY to get stats
    _, _, _, stats, input_dim = preprocess_data(df_train, config, train_stats=None)
    
    print("Normalizing Test set with Training statistics...")
    # Step 2: Process Test using Train stats
    X_test_norm, _, _, _, _ = preprocess_data(df_test, config, train_stats=stats)
    
    # --- INFERENCE ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = PhotoZNet(
        input_size=input_dim,
        hidden_layers=config['model']['hidden_layers'],
        dropout_rates=config['model']['dropout_rates']
    ).to(device)
    
    model_path = os.path.join(config['experiment']['save_dir'], f"{config['experiment']['group_name']}.pth")
    print(f"Loading model from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    print("Running inference on Test Set...")
    with torch.no_grad():
        inputs_tensor = torch.FloatTensor(X_test_norm).to(device)
        preds = model(inputs_tensor).cpu().numpy().flatten()
    
    df_test['Z_PRED'] = preds
    # Unified Z column
    if 'Z' not in df_test.columns:
        df_test['Z'] = np.where(df_test['SPECTYPE'] == 2.0, df_test['Z_QSO'], df_test['Z_GAL'])
        
    return df_test

# --- PLOTTING FUNCTIONS ---

def draw_page_1(pdf, df, types):
    """Page 1: Scatter plots (Pred vs True) - Updated for 4 Types"""
    
    # --- CHANGE: 4 Columns to accommodate GALAXY_OOD2 ---
    fig, axs = plt.subplots(2, 4, figsize=(28, 16)) 
    fig.suptitle('Page 1: Predicted vs True Redshift (Test Set)', fontsize=24, weight='bold')
    
    for i, t in enumerate(types):
        # Handle if we have fewer types than columns (though here we expect 4)
        if i >= 4: break
            
        subset = df[df['TYPE'] == t]
        if len(subset) == 0: 
            # Disable axis if no data for this type
            axs[0, i].axis('off')
            axs[1, i].axis('off')
            continue
        
        z_true = subset['Z'].values
        z_pred = subset['Z_PRED'].values
        mag_i = subset['MAG_i'].values
        
        # Row 1: Density
        ax_dens = axs[0, i]
        x_d, y_d, z_d = get_point_density(z_true, z_pred)
        sc1 = ax_dens.scatter(x_d, y_d, c=z_d, s=10, cmap='plasma') 
        ax_dens.plot([0, 1.8], [0, 1.8], 'w--', alpha=0.5, linewidth=3)
        
        ax_dens.set_title(f"{t} (Density)")
        ax_dens.set_xlabel('Z True'); ax_dens.set_ylabel('Z Predicted')
        ax_dens.set_xlim(0, 1.8); ax_dens.set_ylim(0, 1.8)
        
        # Add colorbar only on the last column
        if i == 3: plt.colorbar(sc1, ax=ax_dens, label='Point Density')

        # Row 2: MAG_i
        ax_mag = axs[1, i]
        sc2 = ax_mag.scatter(z_true, z_pred, c=mag_i, s=10, cmap='viridis_r', alpha=0.7)
        ax_mag.plot([0, 1.8], [0, 1.8], 'w--', alpha=0.5, linewidth=3)
        
        ax_mag.set_title(f"{t} (Color: MAG_i)")
        ax_mag.set_xlabel('Z True'); ax_mag.set_ylabel('Z Predicted')
        ax_mag.set_xlim(0, 1.8); ax_mag.set_ylim(0, 1.8)
        
        if i == 3: plt.colorbar(sc2, ax=ax_mag, label='MAG_i')

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

def draw_metric_page(pdf, df, title_prefix, filter_condition=None):
    """Generic function for Pages 2, 3, 4"""
    
    if filter_condition is not None:
        data = df[filter_condition].copy()
    else:
        data = df.copy()
        
    unique_types = np.sort(data['TYPE'].unique())
    
    fig, axs = plt.subplots(2, 3, figsize=(22, 16))
    fig.suptitle(f'{title_prefix}', fontsize=24, weight='bold')
    
    bins_mag = np.arange(17, 23.3 + 0.25, 0.25)
    bins_z = np.arange(0, 1.8 + 0.1, 0.1)
    
    for t in unique_types:
        subset = data[data['TYPE'] == t]
        if len(subset) < 10: continue
        lbl = f"{t} (N={len(subset)})"
        col = TYPE_COLORS.get(t, 'white')
        
        # --- Row 1: vs MAG_i ---
        x_m, bias_m, sig_m, out_m = compute_metrics_binned(subset, 'MAG_i', 'Z', 'Z_PRED', bins_mag)
        
        axs[0, 0].plot(x_m, bias_m, label=lbl, color=col, marker='o', markersize=6)
        axs[0, 1].plot(x_m, sig_m, label=lbl, color=col, marker='o', markersize=6)
        
        out_m_log = (out_m * 100) 
        axs[0, 2].plot(x_m, out_m_log, label=lbl, color=col, marker='o', markersize=6)
        
        # --- Row 2: vs Z True ---
        x_z, bias_z, sig_z, out_z = compute_metrics_binned(subset, 'Z', 'Z', 'Z_PRED', bins_z)
        
        axs[1, 0].plot(x_z, bias_z, label=lbl, color=col, marker='o', markersize=6)
        axs[1, 1].plot(x_z, sig_z, label=lbl, color=col, marker='o', markersize=6)
        
        out_z_log = (out_z * 100) 
        axs[1, 2].plot(x_z, out_z_log, label=lbl, color=col, marker='o', markersize=6)

    # --- Decoration ---
    for ax in axs[0, :]: 
        ax.set_xlabel('MAG_i'); ax.set_xlim(17, 23.3)
    
    for ax in axs[1, :]: 
        ax.set_xlabel('Z True'); ax.set_xlim(0, 1.8)
        
    axs[0, 0].set_ylabel('Bias $\Delta z$'); axs[1, 0].set_ylabel('Bias $\Delta z$')
    
    axs[0, 1].set_ylabel('$\sigma_{NMAD}$ (log)'); axs[1, 1].set_ylabel('$\sigma_{NMAD}$ (log)')
    axs[0, 1].set_yscale('log'); axs[1, 1].set_yscale('log')
    
    axs[0, 2].set_ylabel('Outlier %'); axs[1, 2].set_ylabel('Outlier %')
    
    # Legend
    axs[0, 0].legend(loc='lower left', frameon=True)
    
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

def main():
    cfg = load_config()
    
    # 1. Prepare Data & Inference (ON TEST SET)
    df_test = prepare_inference_data(cfg)
    
    # 2. Setup PDF
    output_dir = os.path.join(os.getcwd(), '..', 'pdf')
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f"{cfg['experiment']['group_name']}_test_evaluation.pdf")
    
    print(f"Generating Test report: {pdf_path}")
    
    with PdfPages(pdf_path) as pdf:
        # Page 1 - Added GALAXY_OOD2
        print("Drawing Page 1...")
        types_page1 = ['GALAXY_ID', 'QSO', 'GALAXY_OOD1', 'GALAXY_OOD2']
        draw_page_1(pdf, df_test, types_page1)
        
        # Page 2
        print("Drawing Page 2...")
        draw_metric_page(pdf, df_test, "Page 2: Test Metrics (All Objects)")
        
        # Page 3 & 4
        available_ews = [col for col in EW_EMISSION_LINES if col in df_test.columns]
        if available_ews:
            is_elg = (df_test[available_ews] > 10).any(axis=1)
            # Include OOD2 in galaxy definition
            is_gal = df_test['TYPE'].isin(['GALAXY_ID', 'GALAXY_OOD1', 'GALAXY_OOD2'])
            
            print("Drawing Page 3 (ELG)...")
            draw_metric_page(pdf, df_test, "Page 3: Test Metrics (ELG Only)", is_gal & is_elg)
            
            print("Drawing Page 4 (Passive)...")
            draw_metric_page(pdf, df_test, "Page 4: Test Metrics (Passive Only)", is_gal & (~is_elg))
        else:
            print("Warning: Emission line columns not found. Skipping Pages 3 & 4.")

    print("Test Evaluation Complete.")

if __name__ == "__main__":
    main()
