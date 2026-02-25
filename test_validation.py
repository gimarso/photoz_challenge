import os
import yaml
import argparse
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import gaussian_kde
# Import the RandomForestPhotoZ class along with the neural network
from ML_models import PhotoZNet, RandomForestPhotoZ

# --- 1. ENVIRONMENT CONFIGURATION ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# --- 2. PLOTTING STYLE ---
# Apply dark background and user-specified scaling preferences
plt.style.use('dark_background')
plt.rcParams.update({
    'font.size': 14,          # General font size
    'axes.labelsize': 18,     # Axis labels
    'axes.titlesize': 20,     # Titles
    'xtick.labelsize': 14,    # X-axis tick labels
    'ytick.labelsize': 14,    # Y-axis tick labels
    'legend.fontsize': 14,    # Legend text size
    'lines.linewidth': 3      # Width of plotted lines
})

# --- 3. CONSTANTS & CONFIG ---
EW_EMISSION_LINES = [
    'OII_3729_EW', 'OIII_5007_EW', 'NII_6584_EW', 'HBETA_4861_EW', 
    'HALPHA_6562_EW', 'MGII_2796_EW', 'CIV_1549_EW', 'LYALPHA_EW'
]

# Updated dictionary mapping each target type to a specific color
TYPE_COLORS = {
    'GALAXY': 'tab:cyan',
    'QSO': 'tab:red',
    'GALAXY_ID': 'tab:blue',
    'GALAXY_HIGH_Z': 'tab:purple',
    'GALAXY_MISSING_BANDS': 'tab:orange',
    'GALAXY_OFFSET': 'tab:pink',
    'Passive': 'tab:orange',
    'ELG': 'tab:green'
}

# --- 4. HELPER FUNCTIONS & PROCESSING ---

def load_config(config_path="config.yaml"):
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_mad(series):
    """Calculates Median Absolute Deviation for robust scaling."""
    median = series.median()
    return (series - median).abs().median()

def get_point_density(x, y):
    """Calculates point density for coloring dense scatter plots efficiently."""
    xy = np.vstack([x, y])
    if xy.shape[1] > 5000:
        indices = np.random.choice(xy.shape[1], 5000, replace=False)
        z = gaussian_kde(xy[:, indices])(xy)
    else:
        z = gaussian_kde(xy)(xy)
    
    idx = z.argsort()
    return x[idx], y[idx], z[idx]

def preprocess_data(df, config, train_stats=None):
    """
    Selects features, handles missing values, and normalizes the dataset.
    Uses training statistics to normalize validation/test data to prevent leakage.
    """
    # 1. Select inputs dynamically based on the configuration
    selected_cols = []
    for group in config['data']['selected_features']:
        selected_cols.extend(config['data']['inputs'][group])
    
    X = df[selected_cols].copy()
    
    # Target definition (Unified Z variable)
    if 'Z' in df.columns:
        y = df['Z'].values
    else:
        y = np.where(df['SPECTYPE'] == 2.0, df['Z_QSO'], df['Z_GAL'])

    types = df['TYPE'].values

    # 2. Impute missing data (NaN replaced with 0.0)
    X = X.fillna(0.0)

    # 3. Identify columns needing normalization
    cols_to_norm = []
    for group in config['data']['features_to_normalize']:
        if group in config['data']['selected_features']:
             cols_to_norm.extend(config['data']['inputs'][group])
    
    # Calculate statistics only if train_stats is not provided
    if train_stats is None:
        medians = X[cols_to_norm].median()
        mads = X[cols_to_norm].apply(get_mad)
        # Prevent division by zero
        mads = mads.replace(0, 1.0)
        train_stats = {'medians': medians, 'mads': mads}
    
    # Apply robust normalization
    X[cols_to_norm] = (X[cols_to_norm] - train_stats['medians']) / train_stats['mads']
    
    return X.values, y, types, train_stats, len(selected_cols)

def compute_metrics_binned(df, x_col, z_true_col='Z_TRUE', z_pred_col='Z_PRED', bins=None):
    """
    Computes Bias, Sigma_NMAD, and Outlier Fraction binned by a specified feature.
    """
    if bins is None:
        return None, None, None, None

    centers = []
    bias_list = []
    sigma_list = []
    outlier_list = []

    # Calculate Normalized Delta Z
    dz = (df[z_pred_col] - df[z_true_col]) / (1 + df[z_true_col])
    
    for i in range(len(bins) - 1):
        mask = (df[x_col] >= bins[i]) & (df[x_col] < bins[i+1])
        subset_dz = dz[mask]
        
        # Ensure a minimum number of objects per bin for statistical relevance
        if len(subset_dz) < 100: 
            centers.append(np.nan)
            bias_list.append(np.nan)
            sigma_list.append(np.nan)
            outlier_list.append(np.nan)
            continue
            
        centers.append(0.5 * (bins[i] + bins[i+1]))
        
        # Median Bias
        bias_list.append(np.median(subset_dz))
        
        # Sigma NMAD (Normalized Median Absolute Deviation)
        nmad = 1.4826 * np.median(np.abs(subset_dz - np.median(subset_dz)))
        sigma_list.append(nmad)
        
        # Outlier Fraction (Definition: |dz| > 0.15)
        outlier_frac = np.sum(np.abs(subset_dz) > 0.15) / len(subset_dz)
        outlier_list.append(outlier_frac)

    return np.array(centers), np.array(bias_list), np.array(sigma_list), np.array(outlier_list)

def prepare_inference_data(config):
    """
    Loads training and validation datasets, normalizes validation data 
    using training statistics, and runs inference.
    """
    print("Loading datasets...")
    df_train = pd.read_hdf(config['data']['train_path'], key='data')
    df_val = pd.read_hdf(config['data']['val_path'], key='data')

    print("Calculating normalization statistics from Training Set...")
    # Process Train data purely to extract normalization statistics
    _, _, _, stats, input_dim = preprocess_data(df_train, config, train_stats=None)
    
    print("Normalizing Validation set with Training statistics...")
    # Apply Train statistics to Validation data
    X_val_norm, _, _, _, _ = preprocess_data(df_val, config, train_stats=stats)
    
    # Check model type from config
    model_type = config['model'].get('type', 'neural_net')
    
    if model_type == 'neural_net':
        # Setup device and model for inference
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = PhotoZNet(
            input_size=input_dim,
            hidden_layers=config['model']['hidden_layers'],
            dropout_rates=config['model']['dropout_rates']
        ).to(device)
        
        model_path = os.path.join(config['experiment']['save_dir'], f"{config['experiment']['group_name']}.pth")
        print(f"Loading Neural Network model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        
        print("Running inference...")
        with torch.no_grad():
            inputs_tensor = torch.FloatTensor(X_val_norm).to(device)
            preds = model(inputs_tensor).cpu().numpy().flatten()
            
    elif model_type == 'random_forest':
        # Initialize and load the Random Forest model
        model = RandomForestPhotoZ()
        model_path = os.path.join(config['experiment']['save_dir'], f"{config['experiment']['group_name']}_rf.joblib")
        print(f"Loading Random Forest model from {model_path}...")
        model.load(model_path)
        
        print("Running inference...")
        # Get mean predictions and standard deviations (uncertainty)
        preds, std_preds = model.predict(X_val_norm)
        
        # Save standard deviation to dataframe for potential future use in plots
        df_val['Z_PRED_STD'] = std_preds
    
    # Append predictions to the validation dataframe
    df_val['Z_PRED'] = preds
    
    # Ensure a unified Z column exists
    if 'Z' not in df_val.columns:
        df_val['Z'] = np.where(df_val['SPECTYPE'] == 2.0, df_val['Z_QSO'], df_val['Z_GAL'])
        
    return df_val

# --- PLOTTING FUNCTIONS ---

def draw_page_1(pdf, df, types):
    """
    Generates Page 1: Scatter plots comparing Predicted vs True Redshift.
    Dynamically sizes the subplots based on the number of types provided.
    """
    n_cols = len(types)
    fig, axs = plt.subplots(2, n_cols, figsize=(22, 16)) 
    fig.suptitle('Page 1: Predicted vs True Redshift (by Type)', fontsize=24, weight='bold')
    
    for i, t in enumerate(types):
        subset = df[df['TYPE'] == t]
        if len(subset) == 0: continue
        
        z_true = subset['Z'].values
        z_pred = subset['Z_PRED'].values
        mag_i = subset['MAG_i'].values
        
        # Handle indexing correctly depending on whether n_cols > 1
        ax_dens = axs[0, i] if n_cols > 1 else axs[0]
        ax_mag = axs[1, i] if n_cols > 1 else axs[1]
        
        # Set dynamic plot limits based on type
        if t == 'GALAXY':
            z_lim = (0, 1)
        elif t == 'QSO':
            z_lim = (0, 4)
        else:
            z_lim = (0, 1.8)
        
        # Row 1: Density Scatter
        x_d, y_d, z_d = get_point_density(z_true, z_pred)
        sc1 = ax_dens.scatter(x_d, y_d, c=z_d, s=10, cmap='plasma')
        ax_dens.plot(z_lim, z_lim, 'w--', alpha=0.5, linewidth=3)
        
        ax_dens.set_title(f"{t} (Density)")
        ax_dens.set_xlabel('Z True'); ax_dens.set_ylabel('Z Predicted')
        ax_dens.set_xlim(z_lim); ax_dens.set_ylim(z_lim)
        
        # Append colorbar only to the final plot in the row
        if i == n_cols - 1: plt.colorbar(sc1, ax=ax_dens, label='Point Density')

        # Row 2: Colored by MAG_i (now labeled as $iSDSS$)
        sc2 = ax_mag.scatter(z_true, z_pred, c=mag_i, s=10, cmap='viridis_r', alpha=0.7)
        ax_mag.plot(z_lim, z_lim, 'w--', alpha=0.5, linewidth=3)
        
        ax_mag.set_title(f"{t} (Color: $iSDSS$)")
        ax_mag.set_xlabel('Z True'); ax_mag.set_ylabel('Z Predicted')
        ax_mag.set_xlim(z_lim); ax_mag.set_ylim(z_lim)
        
        if i == n_cols - 1: plt.colorbar(sc2, ax=ax_mag, label='$iSDSS$')

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

def draw_metric_page(pdf, df, title_prefix, filter_condition=None):
    """
    Generates generic performance metric pages (Bias, Sigma NMAD, Outliers).
    Plots are customized with larger fonts, log scales, and no grid lines.
    """
    if filter_condition is not None:
        data = df[filter_condition].copy()
    else:
        data = df.copy()
        
    unique_types = data['TYPE'].unique()
    
    fig, axs = plt.subplots(2, 3, figsize=(22, 16))
    fig.suptitle(f'{title_prefix}', fontsize=24, weight='bold')
    
    bins_mag = np.arange(17, 23.3 + 0.25, 0.25)
    bins_z = np.arange(0, 1.0 + 0.1, 0.1)
    
    for t in unique_types:
        subset = data[data['TYPE'] == t]
        if len(subset) < 10: continue
        lbl = f"{t} (N={len(subset)})"
        col = TYPE_COLORS.get(t, 'white')
        
        # --- Metrics vs MAG_i ---
        x_m, bias_m, sig_m, out_m = compute_metrics_binned(subset, 'MAG_i', 'Z', 'Z_PRED', bins_mag)
        
        # 1. Bias (Linear scale)
        axs[0, 0].plot(x_m, bias_m, label=lbl, color=col, marker='o', markersize=6)
        
        # 2. Sigma NMAD (Log scale)
        axs[0, 1].plot(x_m, sig_m, label=lbl, color=col, marker='o', markersize=6)
        
        # 3. Outlier Fraction (Converted to percentage, log scale, +1e-6)
        out_m_log = (out_m * 100) + 1e-6
        axs[0, 2].plot(x_m, out_m_log, label=lbl, color=col, marker='o', markersize=6)
        
        # --- Metrics vs Z True ---
        x_z, bias_z, sig_z, out_z = compute_metrics_binned(subset, 'Z', 'Z', 'Z_PRED', bins_z)
        
        # 1. Bias
        axs[1, 0].plot(x_z, bias_z, label=lbl, color=col, marker='o', markersize=6)
        
        # 2. Sigma NMAD (Log scale)
        axs[1, 1].plot(x_z, sig_z, label=lbl, color=col, marker='o', markersize=6)
        
        # 3. Outlier Fraction (Converted to percentage, log scale, +1e-6)
        out_z_log = (out_z * 100) + 1e-6
        axs[1, 2].plot(x_z, out_z_log, label=lbl, color=col, marker='o', markersize=6)

    # --- Axes formatting and labels ---
    
    # Configure X Axis (No gridlines as per preferences)
    for ax in axs[0, :]: 
        ax.set_xlabel('$iSDSS$')
        ax.set_xlim(17, 23.3)
    
    for ax in axs[1, :]: 
        ax.set_xlabel('Z True')
        ax.set_xlim(0, 1.0)
        
    # Configure Y Axis Labels & Scales
    axs[0, 0].set_ylabel('Bias $\Delta z$'); axs[1, 0].set_ylabel('Bias $\Delta z$')
    
    axs[0, 1].set_ylabel('$\sigma_{NMAD}$ (log)'); axs[1, 1].set_ylabel('$\sigma_{NMAD}$ (log)')
    axs[0, 1].set_yscale('log'); axs[1, 1].set_yscale('log') # Log scale for Sigma
    
    axs[0, 2].set_ylabel('Outlier % (log)'); axs[1, 2].set_ylabel('Outlier % (log)')
    axs[0, 2].set_yscale('log'); axs[1, 2].set_yscale('log') # Log scale for Outliers

    # Render Legend explicitly at the bottom left
    axs[0, 0].legend(loc='lower left', frameon=True)
    
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

def draw_nll_page(pdf, df, types):
    """
    Generates Page 3: Scatter plots showing Negative Log-Likelihood (NLL) vs True Redshift.
    Only executed if uncertainty (Z_PRED_STD) is available in the dataframe.
    """
    n_cols = len(types)
    fig, axs = plt.subplots(1, n_cols, figsize=(22, 8)) 
    if n_cols == 1:
        axs = [axs]
        
    fig.suptitle('Page 3: Negative Log-Likelihood (NLL) vs True Redshift', fontsize=24, weight='bold')
    
    # Calculate NLL assuming Gaussian distribution
    # Adding a small epsilon to variance to prevent division by zero or log(0)
    variance = np.maximum(df['Z_PRED_STD']**2, 1e-6)
    dz = df['Z'] - df['Z_PRED']
    df['NLL'] = 0.5 * np.log(2 * np.pi * variance) + (dz**2) / (2 * variance)
    
    for i, t in enumerate(types):
        subset = df[df['TYPE'] == t]
        if len(subset) == 0: continue
        
        z_true = subset['Z'].values
        nll = subset['NLL'].values
        
        # Set dynamic plot limits based on type for X-axis
        if t == 'GALAXY':
            z_lim = (0, 1)
        elif t == 'QSO':
            z_lim = (0, 4)
        else:
            z_lim = (0, 1.8)
            
        # Smart axis for Y-axis (NLL) by filtering extreme outliers
        p1 = np.percentile(nll, 1) if len(nll) > 0 else -5
        p99 = np.percentile(nll, 99) if len(nll) > 0 else 10
        nll_lim = (p1 - 1, p99 + 5)
        
        ax = axs[i]
        
        # Density Scatter for NLL
        x_d, y_d, z_d = get_point_density(z_true, nll)
        sc = ax.scatter(x_d, y_d, c=z_d, s=10, cmap='plasma')
        
        ax.set_title(f"{t} (NLL vs True Z)")
        ax.set_xlabel('Z True'); ax.set_ylabel('NLL')
        ax.set_xlim(z_lim); ax.set_ylim(nll_lim)
        
        # Append colorbar only to the final plot in the row
        if i == n_cols - 1: plt.colorbar(sc, ax=ax, label='Point Density')

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

def main():
    cfg = load_config()
    
    # 1. Prepare Data & Execute Inference
    df_val = prepare_inference_data(cfg)
    
    # 2. Setup PDF Generation
    output_dir = os.path.join(os.getcwd(), 'pdf')
    os.makedirs(output_dir, exist_ok=True)
    # The output filename remains exactly identical as requested
    pdf_path = os.path.join(output_dir, f"{cfg['experiment']['group_name']}_evaluation.pdf")
    
    print(f"Generating report: {pdf_path}")
    
    with PdfPages(pdf_path) as pdf:
        # Page 1: Passing the types present in our validation set (GALAXY and QSO)
        print("Drawing Page 1...")
        draw_page_1(pdf, df_val, ['GALAXY', 'QSO'])
        
        # Page 2: Overall Metrics restricted to GALAXY only
        print("Drawing Page 2...")
        draw_metric_page(pdf, df_val, "Page 2: Performance Metrics (GALAXY Only)", df_val['TYPE'] == 'GALAXY')
        
        # Page 3: Negative Log-Likelihood (only if uncertainty is available)
        if 'Z_PRED_STD' in df_val.columns:
            print("Drawing Page 3 (NLL)...")
            draw_nll_page(pdf, df_val, ['GALAXY', 'QSO'])
        
    print("Evaluation Complete.")

    # 3. Global Metrics Table Generation
    print("\n--- Global Metrics Table ---")
    metrics = []
    for t in ['GALAXY', 'QSO']:
        subset = df_val[df_val['TYPE'] == t]
        if len(subset) > 0:
            dz = (subset['Z_PRED'] - subset['Z']) / (1 + subset['Z'])
            bias = np.median(dz)
            nmad = 1.4826 * np.median(np.abs(dz - bias))
            outliers = np.sum(np.abs(dz) > 0.15) / len(dz)
            metrics.append({
                'TYPE': t, 
                'Bias Global': bias, 
                'Sigma_NMAD Global': nmad, 
                'Outlier Fraction Global': outliers
            })
    
    if metrics:
        df_metrics = pd.DataFrame(metrics)
        print(df_metrics.to_string(index=False))

if __name__ == "__main__":
    main()